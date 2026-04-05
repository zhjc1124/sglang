import logging
import os
import socket
import struct
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
    InitLoadBackParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.storage.flexkv.flexkv_ipc_utils import (
    EFD_SEMAPHORE,
    eventfd,
    eventfd_read,
    send_fds,
)
from sglang.srt.utils import broadcast_pyobj

try:
    from flexkv.common.request import KVResponseStatus
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.integration.config import FlexKVConfig
    from flexkv.kvmanager import KVManager
    from flexkv.server.client import KVTPClient
except ImportError as e:
    raise RuntimeError("FlexKV is not installed. Please install it.") from e

logger = logging.getLogger(__name__)


class FlexKVLoadOperation:
    def __init__(
        self,
        task_id: int,
        device_indices: torch.Tensor,
        node_id: int,
        node: Optional[TreeNode] = None,
    ):
        self.task_id = task_id
        self.device_indices = device_indices
        self.node_id = node_id
        self.node = node
        self.host_hit_length = device_indices.numel()

class FlexKVLayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        # EFD_SEMAPHORE: each read decrements counter by 1
        self.load_event_fds: List[int] = [
            eventfd(0, EFD_SEMAPHORE) for _ in range(num_layers)
        ]
        # Initially True so first update_producer() can proceed
        self._finished = True
        self._last_layer_wait_count = 0
        # Need 2 waits per layer (K and V)
        self.wait_remaining: List[int] = [2] * num_layers

    def reset_for_new_transfer(self):
        """Reset state for a new transfer."""
        self._finished = False
        self._last_layer_wait_count = 0
        self.wait_remaining = [2] * self._num_layers

    def wait(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        eventfd_read(self.load_event_fds[layer_index])
        # Mark finished when last layer waited twice (K and V)
        if layer_index == self._num_layers - 1:
            self._last_layer_wait_count += 1
            if self._last_layer_wait_count >= 2:
                self._finished = True

    def close(self):
        for fd in self.load_event_fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self.load_event_fds.clear()

    def __del__(self):
        self.close()


class FlexKVLayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.num_counters = 3  # Triple buffering
        self.events: List[FlexKVLayerLoadingEvent] = [
            FlexKVLayerLoadingEvent(num_layers) for _ in range(self.num_counters)
        ]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[self.producer_index]._finished, (
            "Producer event should be finished before reuse"
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        event = self.events[self.consumer_index]
        if event.wait_remaining[threshold] <= 0:
            return
        event.wait_remaining[threshold] -= 1
        event.wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1

    def __del__(self):
        for event in self.events:
            event.close()
        self.events.clear()


class FlexKVConnector:
    """Manages KV cache operations through FlexKV's distributed cache system."""

    def __init__(
        self,
        sgl_config: ModelConfig,
        page_size: int,
        tp_size: int,
        tp_rank: int,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.flexkv_config = FlexKVConfig.from_env()
        self.flexkv_config.post_init_from_sglang_config(
            sglang_config=sgl_config, tp_size=tp_size, page_size=page_size
        )

        self.sgl_config = sgl_config
        self.tp_size = tp_size
        self.rank = tp_rank
        self.k_pool = k_pool
        self.v_pool = v_pool
        self.tp_group = tp_group

        if self.rank == 0:
            self.kv_manager = KVManager(
                model_config=self.flexkv_config.model_config,
                cache_config=self.flexkv_config.cache_config,
                server_recv_port=self.flexkv_config.server_recv_port,
            )
            self.kv_manager.start()

        self.tp_client = KVTPClient(self.flexkv_config.gpu_register_port, 0, self.rank)
        self.register_to_server(self.k_pool, self.v_pool)

        # task_id -> req_id mapping (rank 0 only)
        self.inflight_taskid2reqid: Dict[int, int] = {}
        # Skipped req_ids for lock release (rank 0 only)
        self.inflight_skipped_reqids: List[int] = []

        # Layer-by-layer transfer config
        self.num_layers = sgl_config.num_hidden_layers if sgl_config else 0
        self.enable_layerwise_transfer = bool(
            int(os.getenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", 1))
        )
        self.layerwise_eventfd_socket = os.getenv(
            "FLEXKV_LAYERWISE_EVENTFD_SOCKET", "/tmp/flexkv_layerwise_eventfd.sock"
        )
        self.layerwise_eventfd_connect_max_retries = max(
            360,
            int(os.getenv("FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES", "0")),
        )

        self.layer_done_counter: Optional[FlexKVLayerDoneCounter] = None
        self._worker_connected = False

        self._init_layer_transfer_components()

        if self.rank == 0:
            while not self.kv_manager.is_ready():
                time.sleep(3)
                logger.info("[FlexKV] Waiting for FlexKV to be ready...")
            logger.info("[FlexKV] FlexKV is ready")

        logger.info(
            f"[FlexKV] Connector initialized for rank {self.rank}, "
            f"layerwise_transfer={self.enable_layerwise_transfer}"
        )

    def _init_layer_transfer_components(self):
        """Initialize layer-by-layer transfer components and send eventfds to worker."""
        if not self.enable_layerwise_transfer:
            self.layer_done_counter = None
            self._worker_connected = False
            logger.info(f"[FlexKV] Rank {self.rank}: Layerwise transfer disabled")
            return

        self.layer_done_counter = FlexKVLayerDoneCounter(self.num_layers)
        self._send_eventfds_to_worker()
        logger.info(f"[FlexKV] Rank {self.rank}: Initialized layerwise transfer")

    def _send_eventfds_to_worker(
        self,
        max_retries: Optional[int] = None,
        retry_interval: float = 1.0,
    ):
        """Connect to LayerwiseTransferWorker via Unix socket and send eventfds.

        Max connect attempts default to ``self.layerwise_eventfd_connect_max_retries``
        (env: ``FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES``, default 360).
        """
        if max_retries is None:
            max_retries = self.layerwise_eventfd_connect_max_retries
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Retry until worker is ready
        for attempt in range(max_retries):
            try:
                sock.connect(self.layerwise_eventfd_socket)
                logger.info(f"[FlexKV] Rank {self.rank}: Connected to worker socket")
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if attempt == max_retries - 1:
                    sock.close()
                    raise RuntimeError(
                        f"[FlexKV] Rank {self.rank}: Failed to connect after {max_retries} attempts"
                    )
                if attempt % 10 == 0:
                    logger.info(f"[FlexKV] Rank {self.rank}: Worker not ready, retrying...")
                time.sleep(retry_interval)

        try:
            # Send metadata: tp_rank, tp_size, num_layers, num_counters
            num_counters = self.layer_done_counter.num_counters
            metadata = struct.pack("iiii", self.rank, self.tp_size, self.num_layers, num_counters)
            sock.sendall(metadata)

            # Send all eventfds for each counter set
            for counter_id in range(num_counters):
                fds = self.layer_done_counter.events[counter_id].load_event_fds
                send_fds(sock, fds, struct.pack("i", counter_id))

            self._worker_connected = True
            logger.info(f"[FlexKV] Rank {self.rank}: Sent {num_counters} sets of eventfds")

        except Exception as e:
            sock.close()
            raise RuntimeError(f"[FlexKV] Rank {self.rank}: Failed to send eventfds: {e}")

    def register_layer_transfer_counter(self, kvcache) -> None:
        """Register layer done counter with the KV cache."""
        if self.layer_done_counter is not None:
            kvcache.register_layer_transfer_counter(self.layer_done_counter)

    def chunk_size(self) -> int:
        """Return the chunk size used by FlexKV."""
        return self.flexkv_config.cache_config.tokens_per_block

    def poll_store_tasks(self) -> tuple[List[int], List[int]]:
        """Non-blocking poll for completed/skipped store tasks (rank 0 only)."""
        if self.rank != 0:
            return [], []

        completed_req_ids = []
        if self.inflight_taskid2reqid:
            completed_dict = self.kv_manager.try_wait(
                task_ids=list(self.inflight_taskid2reqid.keys())
            )
            for task_id in completed_dict:
                completed_req_ids.append(self.inflight_taskid2reqid.pop(task_id))

        skipped_req_ids = self.inflight_skipped_reqids
        self.inflight_skipped_reqids = []
        return completed_req_ids, skipped_req_ids

    def store_kv_async(self, token_ids: List[int], kv_indices: torch.Tensor, req_id: int) -> int:
        """Store KV cache to FlexKV asynchronously. Returns task_id or -1 if skipped."""
        if self.rank != 0:
            return -1

        try:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            assert len(token_ids) == len(kv_indices)

            task_id, unmatched_mask = self.kv_manager.put_match(
                token_ids=token_ids_np, token_mask=None
            )

            if unmatched_mask.sum() > 0:
                filtered = kv_indices[unmatched_mask]
                slot_mapping = filtered.cpu() if filtered.is_cuda else filtered
                self.kv_manager.launch(task_ids=[task_id], slot_mappings=[slot_mapping])
                self.inflight_taskid2reqid[task_id] = req_id
                return task_id
            else:
                self.inflight_skipped_reqids.append(req_id)
                return -1
        except Exception as e:
            logger.error(f"[FlexKV] store_kv_async failed: {e}")
            return -1

    def wait_task(self, task_id: int, timeout: float = 20.0) -> bool:
        """Wait for a task to complete. Returns True on success."""
        if task_id < 0:
            return True

        # Only tp0 has real tasks to wait for
        if self.rank == 0:
            try:
                response = self.kv_manager.wait([task_id], timeout=timeout)
                if task_id in response and response[task_id].status == KVResponseStatus.SUCCESS:
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"FlexKV wait_task failed: {e}")
                return False
        else:
            # Other ranks don't have real tasks, so always return success
            return True

    def launch_layerwise_batch_transfer(
        self,
        load_queue: List[FlexKVLoadOperation],
        producer_id: int = 0,
    ):
        if self.rank != 0:
            return 0
        
        task_ids = []
        slot_mappings = []
        
        for op in load_queue:
            if op.task_id < 0:
                continue
            
            slot_mapping_cpu = op.device_indices.cpu() if op.device_indices.is_cuda else op.device_indices
            task_ids.append(op.task_id)
            slot_mappings.append(slot_mapping_cpu)
        
        if task_ids:
            self.kv_manager.launch(
                task_ids=task_ids,
                slot_mappings=slot_mappings,
                as_batch=True,
                layerwise_transfer=True,
                counter_id=producer_id,
            )
        

    def launch_non_layerwise_batch_transfer(
        self,
        load_queue: List[FlexKVLoadOperation],
    ):   
        if self.rank == 0:
            task_ids = []
            slot_mappings = []
            
            for op in load_queue:
                if op.task_id < 0:
                    continue
                
                slot_mapping_cpu = op.device_indices.cpu() if op.device_indices.is_cuda else op.device_indices
                task_ids.append(op.task_id)
                slot_mappings.append(slot_mapping_cpu)

            if task_ids:
                self.kv_manager.launch(
                    task_ids=task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=False,
                )
                response = self.kv_manager.wait(task_ids, timeout=30.0)
                if not all(
                    tid in response and response[tid].status == KVResponseStatus.SUCCESS
                    for tid in task_ids
                ):
                    logger.warning("[FlexKV] Some tasks failed in non-layerwise transfer")

        if self.tp_group is not None and self.tp_size > 1:
            torch.distributed.barrier(self.tp_group)

    def register_to_server(self, k_caches: List[torch.Tensor], v_caches: List[torch.Tensor]) -> None:
        """Register GPU KV cache buffers to FlexKV server."""
        assert len(k_caches) == len(v_caches)
        assert k_caches[0].ndim == 3, f"Expected 3D tensor, got shape={k_caches[0].shape}"

        num_layer = len(k_caches)
        num_blocks, num_kv_heads, head_size = k_caches[0].shape

        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=num_layer,
            num_block=num_blocks,
            tokens_per_block=1,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=False,
        )
        self.tp_client.register_to_server(k_caches + v_caches, gpu_layout)
        logger.info("[FlexKV] Registered KV caches to server")

    def shutdown(self) -> None:
        """Shutdown FlexKV connection."""
        self.kv_manager.shutdown()


class FlexKVRadixCache(RadixCache):
    def __init__(
        self,
        params,
        server_args,
        model_config: Optional[ModelConfig] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.rank = rank
        self.tp_group = tp_group
        self.tp_size = tp_size

        # req_id -> last node mapping (all ranks)
        self.inflight_reqid2node: Dict[int, TreeNode] = {}
        self.sts_total_seq_len = 0
        self.sts_gpu_cache_len = 0
        self.sts_flexkv_cache_len = 0

        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.flexkv_connector = FlexKVConnector(
            sgl_config=model_config,
            page_size=params.page_size,
            tp_size=tp_size,
            tp_rank=rank,
            tp_group=tp_group,
            k_pool=getattr(
                kvcache,
                "k_buffer",
                getattr(params.token_to_kv_pool_allocator._kvcache, "k_buffer"),
            ),
            v_pool=getattr(
                kvcache,
                "v_buffer",
                getattr(params.token_to_kv_pool_allocator._kvcache, "v_buffer"),
            ),
        )

        self.layer_done_counter = self.flexkv_connector.layer_done_counter
        self.flexkv_connector.register_layer_transfer_counter(kvcache)
        self.num_layers = model_config.num_hidden_layers if model_config else 0

        self.load_queue: List[FlexKVLoadOperation] = []
        self.pending_load_info: Dict[str, tuple] = {}  # rid -> (task_id, key, gpu_cached_len)
        self.ongoing_load_back: Dict[int, tuple] = {}  # node_id -> (node, producer_id)

        super().__init__(params)

    def reset(self):
        super().reset()
        if self.rank == 0:
            for task_id in list(self.flexkv_connector.inflight_taskid2reqid.keys()):
                self.flexkv_connector.wait_task(task_id)

        # Release all locks
        for node in self.inflight_reqid2node.values():
            self.dec_lock_ref(node)
        self.inflight_reqid2node.clear()

        for node, _ in self.ongoing_load_back.values():
            self.dec_lock_ref(node)
        self.ongoing_load_back.clear()

        self.load_queue.clear()
        self.pending_load_info.clear()       

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Match prefix against GPU cache and query FlexKV for uncached tokens."""
        key = params.key
        self.sts_total_seq_len += len(key)
        if self.disable or not key:
            return super().match_prefix(params)

        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]
            params = MatchPrefixParams(key=key, req=params.req, cow_mamba=params.cow_mamba)

        # First, match against GPU radix cache
        base_res = super().match_prefix(params)
        value: torch.Tensor = base_res.device_indices
        self.sts_gpu_cache_len += value.numel()

        last_node: TreeNode = base_res.last_device_node

        uncached_len = len(key) - value.numel()
        if uncached_len == 0:
            return base_res

        # Query FlexKV to see what tokens are available (match only, no transfer)
        flexkv_hit_length = 0
        flexkv_task_id = -1
        if self.rank == 0:
            token_ids_np = np.array(key.token_ids, dtype=np.int64)
            token_mask = torch.zeros(len(key), dtype=torch.bool)
            token_mask[value.numel():] = True  # Only check uncached tokens

            flexkv_task_id, matched_mask = self.flexkv_connector.kv_manager.get_match(
                token_ids=token_ids_np,
                token_mask=token_mask,
            )
            flexkv_hit_length = int(matched_mask.sum()) if matched_mask is not None else 0

        # Broadcast to all ranks in TP mode
        if self.tp_group is not None and self.flexkv_connector.tp_size > 1:
            broadcast_data = broadcast_pyobj([{
                'flexkv_hit_length': flexkv_hit_length,
                'flexkv_task_id': flexkv_task_id,
            }], self.rank, self.tp_group, src=0)[0]
            flexkv_hit_length = broadcast_data['flexkv_hit_length']
            flexkv_task_id = broadcast_data['flexkv_task_id']

        self.sts_flexkv_cache_len += flexkv_hit_length

        # Store pending load info for init_load_back, keyed by req.rid
        rid = params.req.rid if params.req is not None else None
        if rid is None:
            raise ValueError("req.rid is required for FlexKV match_prefix")
        if flexkv_hit_length > 0:
            self.pending_load_info[rid] = (flexkv_task_id, key, value.numel())

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            host_hit_length=flexkv_hit_length,
        )

    def init_load_back(
        self,
        params: InitLoadBackParams,
        **kwargs,
    ):
        """
        Allocate GPU memory, create new TreeNode, and add the load operation to load_queue.
        """
        last_node = params.last_host_node
        host_hit_length = params.host_hit_length
        mem_quota = params.mem_quota

        if host_hit_length <= 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        rid = params.req.rid
        # Get (task_id, key, gpu_cached_len) stored during match_prefix using rid as key
        if rid is None:
            raise ValueError("rid is required for FlexKV init_load_back")

        if rid not in self.pending_load_info:
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        
        task_id, key, gpu_cached_len = self.pending_load_info.pop(rid)
        
        # Check memory quota
        if mem_quota is not None and host_hit_length > mem_quota:
            logger.debug(f"[FlexKV] host_hit_length {host_hit_length} exceeds mem_quota {mem_quota}, skipping load")
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        
        # Allocate GPU memory for the tokens to load
        device_indices = self.token_to_kv_pool_allocator.alloc(host_hit_length)
        if device_indices is None:
            # Try eviction and retry
            self.evict(EvictParams(num_tokens=host_hit_length))
            device_indices = self.token_to_kv_pool_allocator.alloc(host_hit_length)
        
        if device_indices is None:
            logger.warning(f"[FlexKV] Failed to allocate {host_hit_length} GPU slots for load")
            return (
                torch.empty((0,), dtype=torch.int64, device=self.device),
                last_node,
            )
        
        #Create new TreeNode after alloc
        new_node = TreeNode()
        start = gpu_cached_len
        end = start + host_hit_length
        new_node.key = key[start:end]
        new_node.value = device_indices
        new_node.parent = last_node
        last_node.children[self.get_child_key_fn(new_node.key)] = new_node
        
        # Update evictable_size
        self.evictable_size_ += len(device_indices)
        
        # Lock new node to prevent eviction during transfer
        self.inc_lock_ref(new_node)
        
        # Create load operation with saved task_id (no need to re-call get_match)
        load_op = FlexKVLoadOperation(
            task_id=task_id,
            device_indices=device_indices,
            node_id=new_node.id,
            node=new_node,
        )
        self.load_queue.append(load_op)
        
        return device_indices, new_node
    
    def ready_to_load_host_cache(self) -> int:
        """
        Trigger KV cache transfer from FlexKV to GPU.
        
        Lyerwise transfer (enable_layerwise_transfer=True):
            - Returns consumer_index for layer-by-layer waiting
        Non-layerwise transfer (enable_layerwise_transfer=False):
            - Blocks until transfer completes, returns -1 (no layer waiting needed)
        """
        if not self.load_queue:
            return -1
        
        if (not self.flexkv_connector.enable_layerwise_transfer): # non-layerwise transfer mode
            
            logger.debug(f"[FlexKV] Using non-layerwise transfer for {len(self.load_queue)} operations")
            
            # Track nodes being loaded for later unlock (use -1 as producer_id indicator for non-layerwise)
            for op in self.load_queue:
                if op.node is not None:
                    self.ongoing_load_back[op.node_id] = (op.node, -1)
            
            # Launch and wait (non-layerwise)
            self.flexkv_connector.launch_non_layerwise_batch_transfer(self.load_queue)
            
            # Locks will be released in loading_check()
            
            self.load_queue.clear()

            return -1  # No layer-by-layer waiting needed
        
        else: # layerwise transfer mode

            producer_id = self.layer_done_counter.update_producer()
            # Reset event state for new transfer (marks as not finished and resets wait counter)
            self.layer_done_counter.events[producer_id].reset_for_new_transfer()
        
            # Track nodes being loaded with their producer_id for later unlock
            for op in self.load_queue:
                if op.node is not None:
                    self.ongoing_load_back[op.node_id] = (op.node, producer_id)
            
            # Launch layerwise batch transfer (rank 0 only, other ranks do nothing)
            self.flexkv_connector.launch_layerwise_batch_transfer(self.load_queue, producer_id)
            
            self.load_queue.clear()
            
            return producer_id

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        super().cache_finished_req(req, is_insert=is_insert)

        if req.req_pool_idx is None and not is_insert:
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        new_last_node = req.last_node
        if new_last_node is None:
            return

        self.inc_lock_ref(new_last_node)
        try:
            self.flexkv_connector.store_kv_async(
                token_ids=token_ids,
                kv_indices=kv_indices,
                req_id=req.req_pool_idx,
            )
        except Exception as e:
            logger.error(f"[FlexKV] Failed to store KV: {e}")
            return

        if req.req_pool_idx in self.inflight_reqid2node:
            self.dec_lock_ref(self.inflight_reqid2node[req.req_pool_idx])

        self.inflight_reqid2node[req.req_pool_idx] = new_last_node

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""
        super().cache_unfinished_req(req, chunked=chunked)

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        new_last_node = req.last_node
        self.inc_lock_ref(new_last_node)
        try:
            self.flexkv_connector.store_kv_async(
                token_ids=token_ids,
                kv_indices=kv_indices,
                req_id=req.req_pool_idx,
            )
        except Exception as e:
            logger.error(f"[FlexKV] Failed to store KV: {e}")
            return

        if req.req_pool_idx in self.inflight_reqid2node:
            self.dec_lock_ref(self.inflight_reqid2node[req.req_pool_idx])

        self.inflight_reqid2node[req.req_pool_idx] = new_last_node

    def evict(self, params: EvictParams) -> EvictResult:
        """
        Try non-blocking release of completed FlexKV store tasks, evict, and only
        if insufficient tokens are freed, block-wait remaining tasks and evict again.
        """
        if self.disable:
            return EvictResult()

        num_tokens = params.num_tokens

        # Step 1: Non-blocking poll to release completed/skimmed store locks
        try:
            self.writing_check()
        except Exception:
            pass

        # Step 2: Attempt eviction with currently evictable nodes
        result = super().evict(params)
        if result.num_tokens_evicted >= num_tokens:
            return result

        # Step 3: Not enough freed. Block-wait remaining FlexKV tasks, then release all locks and evict the rest.
        # Use tuple instead of list to avoid unnecessary copy overhead
        remaining_reqids = tuple(self.inflight_reqid2node.keys())

        if self.flexkv_connector.rank == 0:
            task_ids = tuple(self.flexkv_connector.inflight_taskid2reqid.keys())
            for task_id in task_ids:
                self.flexkv_connector.wait_task(task_id)
            self.flexkv_connector.inflight_taskid2reqid.clear()

        for req_id in remaining_reqids:
            node = self.inflight_reqid2node[req_id]
            self.dec_lock_ref(node)
        self.inflight_reqid2node.clear()

        remaining_to_evict = num_tokens - result.num_tokens_evicted
        if remaining_to_evict > 0:
            extra = super().evict(EvictParams(num_tokens=remaining_to_evict))
            return EvictResult(
                num_tokens_evicted=result.num_tokens_evicted + extra.num_tokens_evicted
            )
        return result
    
    def pretty_print(self):
        super().pretty_print()
        logger.debug(
            "evictable=%d protected=%d", self.evictable_size_, self.protected_size_
        )

    def loading_check(self):
        """
        Check for completed load operations and release corresponding locks.
        """
        if len(self.ongoing_load_back) == 0:
            return
        
        completed_nodes = []
        for node_id, (node, producer_id) in tuple(self.ongoing_load_back.items()):
            if producer_id == -1:
                # Non-layerwise mode: transfer is synchronous, already complete
                completed_nodes.append(node_id)
            elif self.layer_done_counter is not None:
                # Layerwise mode: check if the loading event for this producer_id has finished
                # Note: layer_done_counter is not None implies layerwise transfer is enabled
                if self.layer_done_counter.events[producer_id]._finished:
                    completed_nodes.append(node_id)
        
        for node_id in completed_nodes:
            node, producer_id = self.ongoing_load_back.pop(node_id)
            self.dec_lock_ref(node)

    def check_kv_events(self):
        self.writing_check()
        self.loading_check()
        # Avoid f-string construction when log level is not enabled
        if self.rank == 0 and self.sts_total_seq_len > 0 and logger.isEnabledFor(logging.INFO):
            logger.info(
                "[FlexKV stats] total_seq_len=%d, gpu_cache_len=%d, flexkv_cache_len=%d, "
                "gpu_cache_ratio=%.4f, flexkv_cache_ratio=%.4f",
                self.sts_total_seq_len,
                self.sts_gpu_cache_len,
                self.sts_flexkv_cache_len,
                self.sts_gpu_cache_len / self.sts_total_seq_len,
                self.sts_flexkv_cache_len / self.sts_total_seq_len,
            )

    def writing_check(self) -> None:
        """Poll for completed store tasks and release corresponding locks."""
        completed, skipped = self.flexkv_connector.poll_store_tasks()

        # Broadcast to all ranks if using tensor-parallel group
        if self.tp_group is not None and self.tp_size > 1:
            payload = broadcast_pyobj(
                [{'completed': completed, 'skipped': skipped}] if self.rank == 0 else [None],
                self.rank, self.tp_group, src=0
            )[0]
            to_release = payload['completed'] + payload['skipped']
        else:
            to_release = completed + skipped

        for req_id in to_release:
            if req_id in self.inflight_reqid2node:
                self.dec_lock_ref(self.inflight_reqid2node.pop(req_id))
