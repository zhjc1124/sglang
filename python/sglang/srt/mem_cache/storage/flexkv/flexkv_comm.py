import ctypes
import errno
import logging
import os
import pickle
import socket
import struct
from collections import deque
from datetime import timedelta
from typing import Any, Deque, Dict, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper

logger = logging.getLogger(__name__)

# ---- PP control command constants ----
CMD_PUT_META = 2
CMD_LAYERWISE = 3
CMD_STORE_COMPLETE = 5


class FlexKVComm:
    """FlexKV hierarchical communication on 4D topology
    (PP × ATTN_DP × ATTN_CP × ATTN_TP).

    Public API: scatter, scatter_pp, barrier, all_reduce_min.
    Public read-only attributes: is_sync_leader, needs_sync, is_pp_active,
    is_pp_sender, is_pp_receiver.

    Communication hierarchy (fan-out / aggregate):

        Scatter (async):   PP_leader --isend--> PP_stage_leaders
                                     --isend--> ATTN_CP_ranks  (per PP stage)
                                     --isend--> ATTN_TP_ranks  (per CP group)

        AllReduce (sync):  ATTN_TP group all_reduce
                           -> ATTN_CP group all_reduce
                           -> PP P2P reduce (stage leaders)
                           -> bcast result back down

        scatter_pp (async): PP0 stage leader --isend--> PP1+ stage leaders
        barrier (sync):     hierarchical barrier: ATTN_TP -> ATTN_CP -> PP
    """

    # ---- Tags for P2P scatter on world_cpu_group ----
    _TAG_SCATTER = int.from_bytes(b"FxSc", byteorder="big")
    _TAG_PP      = int.from_bytes(b"FxPP", byteorder="big")
    _TAG_CP      = int.from_bytes(b"FxCP", byteorder="big")
    _TAG_TP      = int.from_bytes(b"FxTP", byteorder="big")
    # ---- Tags for PP P2P all-reduce / barrier on world_cpu_group ----
    _TAG_PP_AR_MIN = int.from_bytes(b"FxA2", byteorder="big")
    _TAG_PP_BARRIER = int.from_bytes(b"FxB2", byteorder="big")
    _TAG_PP_BARRIER_BCAST = int.from_bytes(b"FxB3", byteorder="big")
    _TAG_AR_BCAST = int.from_bytes(b"FxAR", byteorder="big")

    # ---- Async-work reaper tunables ----
    # Background: gloo isend Work objects do not auto-advance their
    # "completed" state on poll, so a pure poll-based reaper leaks. We
    # actively wait() the oldest works with a tiny timeout. The watermark
    # adapts: it grows on stuck reaps (peer slow / asymmetric) and shrinks
    # back on clean reaps. Empty scatter payloads (~50B over loopback /
    # LAN) complete in <1ms, so PROBE=1ms is comfortable.
    #
    # Each call to ``scatter`` enqueues O(pp + cp + tp) Work objects, and
    # the leader is the only producer per dimension, so backlog grows on
    # the leader rank when peers post recv() late. The numbers below cap
    # the reaper's main-thread cost while still bounding the queue:
    #
    #   * Steady-state ``len(_async_works) <= _REAP_HIGH_BASE``
    #     means O(64KiB) of Python overhead per leader rank, negligible.
    #   * Worst-case stuck ``len(_async_works) ~= _REAP_HIGH_MAX`` keeps
    #     the queue under O(1MiB), which is fine for any real workload.
    #   * Per call: at most ``_REAP_MAX_DRAIN`` wait() probes; happy-path
    #     each is microseconds, stuck-path bails out after the first 1ms
    #     timeout, so worst-case main-thread cost is ~1ms per scatter.
    _REAP_HIGH_BASE = 1024            # initial / minimum trigger watermark
    _REAP_HIGH_MAX  = 8192            # cap on the adaptive watermark
    _REAP_MAX_DRAIN = 256             # bound on works popped per reap call
    _REAP_PROBE     = timedelta(milliseconds=1)
    _REAP_LOG_EVERY = 256             # sample-log every N reap calls

    def __init__(
        self,
        rank_info,
        world_rank: int,
        pp_group=None,
        attn_tp_group=None,
        attn_cp_group=None,
    ):
        model_config = rank_info.model_config
        self.world_rank = world_rank
        # FIFO of pending isend Work objects. Use deque so that the
        # head-of-queue popleft() done by ``_reap_completed_async_works``
        # is O(1) instead of O(n) on a 8K-entry list.
        self._async_works: Deque = deque()
        # Adaptive watermark for async-work reaping. Grows on stuck reaps
        # (peer asymmetric / slow), shrinks back to base on clean reaps.
        self._reap_high: int = self._REAP_HIGH_BASE
        # Counters for sampled debug logging.
        self._reap_calls: int = 0
        self._reap_stuck_total: int = 0
        self._reap_drained_total: int = 0

        # ---- Extract cpu_group from wrapper objects if present ----
        self.pp_cpu_group = (
            getattr(pp_group, "cpu_group", pp_group) if pp_group is not None else None
        )
        self.attn_tp_cpu_group = (
            getattr(attn_tp_group, "cpu_group", attn_tp_group)
            if attn_tp_group is not None
            else None
        )
        self.attn_cp_cpu_group = (
            getattr(attn_cp_group, "cpu_group", attn_cp_group)
            if attn_cp_group is not None
            else None
        )

        # ---- Dimension sizes ----
        self.pp_size = model_config.pp_size
        # dp_size: true DP shard count in both plain-DP and DP-Attention modes.
        # plain DP:       dp_size == sglang dp_size (each shard is an independent
        #                 scheduler process with its own KVManager).
        # DP Attention:   dp_size == attn_dp_size == sglang dp_size (each attn-dp
        #                 shard shares one CPU KV block, identified by dp_client_id).
        # In both cases FlexKVComm does NOT scatter/barrier across dp shards;
        # the dp dimension is handled at the KVManager routing level.
        self.dp_size = model_config.dp_size
        self.attn_tp_size = model_config.tp_size
        self.attn_cp_size = model_config.cp_size

        # ---- 4D coordinate ----
        self.pp_rank = rank_info.pp_rank
        self.dp_rank = rank_info.dp_rank
        self.attn_tp_rank = rank_info.tp_rank
        self.attn_cp_rank = rank_info.cp_rank

        # ---- Role resolution ----
        self.is_pp_stage_leader = (self.attn_tp_rank == 0 and self.attn_cp_rank == 0)
        self.is_sync_leader = (
            self.pp_rank == 0 and self.is_pp_stage_leader
        )
        self.is_pp_leader = (self.pp_rank == 0 and self.is_pp_stage_leader)
        self.is_attn_cp_leader = (self.attn_cp_rank == 0)
        self.is_attn_tp_leader = (self.attn_tp_rank == 0)

        # ---- Rank mapping for point-to-point scatter ----
        # ATTN_TP group ranks (pre-computed once)
        if self.attn_tp_size > 1:
            if self.attn_tp_cpu_group is None:
                raise RuntimeError(
                    f"[FlexKV] attn_tp_size={self.attn_tp_size} > 1 but "
                    f"attn_tp_cpu_group is None — ATTN_TP group is required "
                    f"for scatter/collective operations"
                )
            self._attn_tp_group_ranks = [
                dist.get_global_rank(self.attn_tp_cpu_group, i)
                for i in range(self.attn_tp_cpu_group.size())
            ]
        else:
            self._attn_tp_group_ranks = []

        # PP group ranks for scatter_pp (pre-computed once).
        # pp_cpu_group contains all ranks sharing the same (attn_dp, attn_cp, attn_tp)
        # coordinate across all PP stages — i.e. the full PP column for this rank.
        self._pp_group_global_ranks = (
            [dist.get_global_rank(self.pp_cpu_group, i)
             for i in range(self.pp_cpu_group.size())]
            if self.pp_size > 1 and self.pp_cpu_group is not None else []
        )

        # PP stage leader ranks: attn_tp=0, attn_cp=0 of *this* ATTN_DP shard,
        # one per PP stage.
        #
        # For is_pp_stage_leader ranks (attn_tp=0, attn_cp=0):
        #   pp_cpu_group contains exactly the stage leaders of all PP stages
        #   (same attn_dp/attn_cp/attn_tp coords across all PP stages), so
        #   _pp_stage_leader_ranks == _pp_group_global_ranks.
        #
        # For non-stage-leader ranks (attn_tp>0 or attn_cp>0):
        #   _pp_stage_leader_ranks is only used in _bcast_to_stage_members via
        #   _pp_stage_leader_ranks[self.pp_rank] to find the stage leader to
        #   receive from.  We compute this via _my_stage_leader_rank below.
        if self._pp_group_global_ranks:
            self._pp_stage_leader_ranks = list(self._pp_group_global_ranks)
        else:
            # pp_size == 1: only one stage, leader is self (if stage leader)
            # or derived below (if not stage leader)
            self._pp_stage_leader_ranks = [world_rank]

        # _my_stage_leader_rank: world rank of the stage leader (attn_tp=0,
        # attn_cp=0) in the same PP stage as this rank.
        # - If this rank IS the stage leader, it's just world_rank.
        # - Otherwise, find it via the attn_tp and attn_cp groups:
        #   attn_tp_cpu_group rank 0 = attn_tp=0 peer in same (pp, dp, cp) group
        #   attn_cp_cpu_group rank 0 = attn_cp=0 peer in same (pp, dp, tp=0) group
        #   The stage leader is attn_cp_group.rank0's attn_tp_group.rank0.
        if self.is_pp_stage_leader:
            self._my_stage_leader_rank = world_rank
        elif self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            # attn_cp_group is scoped to attn_tp=0 ranks within this PP stage.
            # rank 0 of attn_cp_group is the stage leader (attn_cp=0, attn_tp=0).
            self._my_stage_leader_rank = dist.get_global_rank(
                self.attn_cp_cpu_group, 0
            )
        elif self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            # attn_cp_size == 1, so attn_cp=0 is always satisfied.
            # attn_tp_group rank 0 is the stage leader (attn_tp=0).
            self._my_stage_leader_rank = dist.get_global_rank(
                self.attn_tp_cpu_group, 0
            )
        else:
            # Both attn_tp_size == 1 and attn_cp_size == 1: this rank IS the
            # stage leader (covered by is_pp_stage_leader above).
            self._my_stage_leader_rank = world_rank

        # ATTN_CP leader ranks within this PP stage's ATTN_DP shard.
        # attn_cp_cpu_group contains all CP ranks for this (pp, attn_dp, attn_tp=0)
        # slice.  The leader of each CP group is the rank with attn_tp_rank == 0,
        # which is the rank in attn_cp_cpu_group itself (since attn_cp_group is
        # scoped to attn_tp=0 ranks).  We derive these from attn_cp_cpu_group.
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            self._attn_cp_leader_ranks = [
                dist.get_global_rank(self.attn_cp_cpu_group, i)
                for i in range(self.attn_cp_cpu_group.size())
            ]
        else:
            self._attn_cp_leader_ranks = []

        # PP stage member ranks: ranks that the stage leader directly sends to
        # in _bcast_to_stage_members.  We use a two-level fan-out:
        #   Level 1 (stage leader sends to):
        #     - attn_tp_group_ranks (same CP group, attn_tp > 0)
        #     - attn_cp_leader_ranks (other CP groups, attn_tp = 0)
        #   Level 2 (each CP leader sends to its own attn_tp members):
        #     - handled inside _bcast_to_stage_members by CP leaders
        # So _pp_stage_member_ranks = attn_tp_group_ranks + attn_cp_leader_ranks
        # (excluding self).
        _direct_members: set = set()
        if self._attn_tp_group_ranks:
            _direct_members.update(self._attn_tp_group_ranks)
        if self._attn_cp_leader_ranks:
            _direct_members.update(self._attn_cp_leader_ranks)
        _direct_members.discard(world_rank)
        self._pp_stage_member_ranks = sorted(_direct_members)

        # ---- Whether sync is needed (any dimension > 1) ----
        # NOTE: dp_size is intentionally excluded here.
        # In both plain-DP and DP-Attention modes, each DP shard runs its own
        # independent KVManager (identified by
        #   dp_client_id = instance_id * dp_size + dp_rank
        # ) via server_client_mode.  The DP dimension is therefore handled at
        # the KVManager routing level, not via FlexKVComm scatter/barrier.
        # FlexKVComm only synchronises within a single DP shard across the
        # (PP × CP × TP) sub-topology.
        self.needs_sync = (self.pp_size > 1 or self.attn_tp_size > 1 or self.attn_cp_size > 1)

        # ==================================================================
        # Communication group strategy
        # ----------------------------
        # P2P operations (send/recv/isend/irecv) on CPU tensors:
        #   -> use world_group.cpu_group (gloo-backed, full TCP mesh).
        #      Sub-group cpu_groups have unreliable TCP pairs for P2P.
        #
        # Collective operations (all_reduce/barrier):
        #   -> use sglang's sub-group _cpu_groups (fine for collectives).
        # ==================================================================

        self._world_cpu_group = get_world_group().cpu_group

        # ---- PP scatter role flags (used by flexkv_connector) ----
        self.pp_group = self.pp_cpu_group if (self.pp_size > 1 and self.is_pp_stage_leader) else None
        self.is_pp_active = self.pp_size > 1
        self.is_pp_sender = self.is_pp_leader
        self.is_pp_receiver = self.is_pp_stage_leader and not self.is_pp_leader

        self.is_cross_node_pp = (self.pp_size > rank_info.pp_size_per_node)
        self.should_send_slot_mapping_to_remote = (
            self.is_pp_receiver and self.is_cross_node_pp
        )

        # Ensure _DpGatheredBufferWrapper._dp_max_padding has a default value
        # before any collective operation (e.g. all_reduce_min called during
        # FlexKV connector init) can indirectly trigger is_allocation_symmetric()
        # → is_dp_max_padding() → cls._dp_max_padding on PP non-first-stage
        # schedulers, where set_dp_buffer_len() has not yet been called.
        # Default True is safe: it makes is_allocation_symmetric() return True
        # (symmetric allocation assumed), which is the correct conservative
        # behaviour before the first forward pass sets the real value.
        if not hasattr(_DpGatheredBufferWrapper, "_dp_max_padding"):
            _DpGatheredBufferWrapper._dp_max_padding = True

        logger.info(
            f"[FlexKV] Comm init: rank={world_rank}, "
            f"pp={self.pp_rank}/{self.pp_size}, "
            f"dp={self.dp_rank}/{self.dp_size}, "
            f"attn_tp={self.attn_tp_rank}/{self.attn_tp_size}, "
            f"attn_cp={self.attn_cp_rank}/{self.attn_cp_size}, "
            f"sync_leader={self.is_sync_leader}, "
            f"stage_leader={self.is_pp_stage_leader}, "
            f"is_cross_node_pp={self.is_cross_node_pp}, "
            f"should_send_slot_mapping_to_remote={self.should_send_slot_mapping_to_remote}"
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def scatter(self, data: Any, blocking: bool = False) -> Any:
        """Hierarchical scatter: PP -> ATTN_CP -> ATTN_TP (async isend).

        Fan-out in 3 stages, each uses isend/irecv on world_cpu_group:
          1. PP: sync_leader -> each PP stage's stage_leader
          2. ATTN_CP: each stage's attn_cp_leader -> other ATTN_CP ranks
          3. ATTN_TP: each CP group's attn_tp_leader -> other ATTN_TP ranks
        """
        # Stage 1: PP scatter (sync_leader -> stage leaders)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            data = self._scatter_group(
                data, self._pp_stage_leader_ranks,
                self.is_pp_leader, self._TAG_PP, blocking,
            )

        # Stage 2: ATTN_CP scatter (attn_cp_leader -> other CP ranks in same PP stage)
        if self._attn_cp_leader_ranks:
            data = self._scatter_group(
                data, self._attn_cp_leader_ranks,
                self.is_attn_cp_leader, self._TAG_CP, blocking,
            )

        # Stage 3: ATTN_TP scatter (attn_tp_leader -> other TP ranks)
        if self._attn_tp_group_ranks:
            data = self._scatter_group(
                data, self._attn_tp_group_ranks,
                self.is_attn_tp_leader, self._TAG_TP, blocking,
            )

        return data

    def scatter_pp(self, data: Any) -> Any:
        """Fan-out across PP stages (PP0 stage leader -> PP1+ stage leaders).

        Only PP stage leaders participate. Non-leaders are no-ops.
        """
        if not self._pp_group_global_ranks or not self.is_pp_stage_leader:
            return data
        is_leader = (self._pp_group_global_ranks[0] == self.world_rank)
        return self._scatter_group(
            data, self._pp_group_global_ranks,
            is_leader, self._TAG_SCATTER, blocking=False,
        )

    def all_reduce_min(self, value: int) -> int:
        """Hierarchical all-reduce MIN across ATTN_TP, ATTN_CP, and PP.

        Every rank participates in each collective layer it belongs to:
          Layer 1  ATTN_TP all_reduce   all attn_tp group members
          Layer 2  ATTN_CP all_reduce   all attn_cp group members
          Layer 3  PP P2P reduce        only PP stage leaders
          Layer 4  bcast result         stage leaders -> non-stage-leaders
        """
        logger.debug(
            f"[FlexKV] all_reduce_min rank={self.world_rank} value={value}"
        )

        tensor = torch.tensor(value, dtype=torch.int64)

        # Layer 1: ATTN_TP all_reduce
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_tp_cpu_group)

        # Layer 2: ATTN_CP all_reduce
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_cp_cpu_group)

        # Layer 3: PP all_reduce via P2P (stage leaders only)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            self._pp_all_reduce_min_p2p(tensor)

        # Layer 4: broadcast result to non-stage-leaders.
        # Required whenever attn_cp_size > 1 (attn_tp>0 leaf ranks are outside
        # the attn_cp_cpu_group and never see the CP-level all_reduce result)
        # or pp_size > 1 (non-stage-leader ranks need the PP-reduced result).
        if self.attn_cp_size > 1 or self.pp_size > 1:
            self._bcast_to_stage_members(tensor, self._TAG_AR_BCAST)

        result = tensor.item()
        logger.debug(
            f"[FlexKV] all_reduce_min rank={self.world_rank} "
            f"value={value} -> {result}"
        )
        return result

    def barrier(self):
        """Hierarchical global barrier: ATTN_TP -> ATTN_CP -> PP -> bcast."""
        logger.debug(f"[FlexKV] barrier ENTER rank={self.world_rank}")

        # Layer 1: ATTN_TP barrier
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            dist.barrier(group=self.attn_tp_cpu_group)

        # Layer 2: ATTN_CP barrier
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            dist.barrier(group=self.attn_cp_cpu_group)

        # Layer 3: PP barrier (stage leaders, P2P)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            self._pp_barrier_p2p()

        # Layer 4: broadcast barrier completion to non-stage-leaders.
        # Same condition as all_reduce_min Layer 4: needed when attn_cp_size > 1
        # so that attn_tp>0 leaf ranks are unblocked after the CP-level barrier.
        if self.attn_cp_size > 1 or self.pp_size > 1:
            dummy = torch.tensor([0], dtype=torch.int64)
            self._bcast_to_stage_members(dummy, self._TAG_PP_BARRIER_BCAST)

        logger.debug(f"[FlexKV] barrier EXIT  rank={self.world_rank}")

    # ==================================================================
    # Unified scatter helper
    # ==================================================================

    def _scatter_group(
        self,
        data: Any,
        group_ranks: List[int],
        is_leader: bool,
        tag: int,
        blocking: bool = False,
    ) -> Any:
        """Scatter within a group: leader sends to all others, followers recv.

        If current rank is not in group_ranks, returns data unchanged.
        """
        if not group_ranks or self.world_rank not in group_ranks:
            return data

        if is_leader:
            dsts = [r for r in group_ranks if r != self.world_rank]
            works = []
            for dst in dsts:
                works.extend(self._isend(dst, data, tag, self._world_cpu_group))
            if blocking:
                for w in works:
                    w.wait()
            else:
                # Reap completed works to bound list growth, but never
                # block on a peer that hasn't posted recv yet — the
                # reaper uses a tiny timeout and bails out on stuck.
                self._reap_completed_async_works()
                self._async_works.extend(works)
            return data
        else:
            return self._recv(group_ranks[0], tag, self._world_cpu_group)

    # ==================================================================
    # Async work management
    # ==================================================================

    def _reap_completed_async_works(self):
        """Drain oldest completed isends with bounded main-thread cost.

        gloo's ``Work.is_completed()`` does not auto-advance on poll, so a
        pure-poll reaper leaks. Here we actively ``wait()`` the oldest works
        with a tiny timeout: on a symmetric channel the head of the
        queue has been in flight for many milliseconds and its matching
        ``recv`` is long posted, so ``wait()`` returns in microseconds. On
        timeout (peer slow / asymmetric) we bail out so the main thread is
        never blocked, and widen the trigger watermark via exponential
        backoff. On a clean reap we shrink the watermark back toward the
        base. When the peer recovers we converge back to steady-state.

        Safety net: once backlog exceeds ``_REAP_HIGH_MAX`` we stop
        backing off and simply keep probing every call so memory cannot
        grow without bound.
        """
        n = len(self._async_works)
        if n <= self._reap_high:
            return

        drained = 0
        stuck = False
        for _ in range(self._REAP_MAX_DRAIN):
            if not self._async_works:
                break
            w = self._async_works[0]
            try:
                w.wait(self._REAP_PROBE)
            except RuntimeError:
                # Oldest work still pending → newer ones are even less
                # likely to be ready. Bail out; next reap will retry.
                stuck = True
                break
            self._async_works.popleft()
            drained += 1

        # Update counters (used for sampled summary log below).
        self._reap_calls += 1
        self._reap_drained_total += drained
        if stuck:
            self._reap_stuck_total += 1

        # Adapt watermark. Once we hit MAX, stop growing it any further;
        # the reaper will then try to drain on every call, keeping the
        # backlog bounded even with a permanently-slow peer.
        prev_high = self._reap_high
        if stuck:
            self._reap_high = min(self._REAP_HIGH_MAX, self._reap_high * 2)
        else:
            self._reap_high = max(self._REAP_HIGH_BASE, self._reap_high // 2)
        if self._reap_high != prev_high:
            logger.debug(
                f"[FlexKV] reap watermark rank={self.world_rank} "
                f"{prev_high}->{self._reap_high} "
                f"(stuck={stuck} drained={drained} backlog={n})"
            )

        # Loud warning if we hit the safety ceiling — usually means the
        # peer is permanently asymmetric and someone should investigate.
        if n >= self._REAP_HIGH_MAX:
            logger.warning(
                f"[FlexKV] reap backlog at safety ceiling rank={self.world_rank} "
                f"backlog={n} drained={drained} stuck={stuck}"
            )

        # Sampled summary every N calls so steady-state behavior is
        # observable without flooding the log.
        if self._reap_calls % self._REAP_LOG_EVERY == 0:
            logger.debug(
                f"[FlexKV] reap stats rank={self.world_rank} "
                f"calls={self._reap_calls} "
                f"drained_total={self._reap_drained_total} "
                f"stuck_total={self._reap_stuck_total} "
                f"backlog={len(self._async_works)} "
                f"high={self._reap_high}"
            )

    # ==================================================================
    # Low-level send / recv
    # ==================================================================

    def _isend(self, dst: int, data: Any, tag: int = 0, group=None) -> list:
        """Non-blocking send via gloo-backed cpu_group (CPU tensor P2P)."""
        serialized = bytearray(pickle.dumps(data))
        t_size = torch.tensor([len(serialized)], dtype=torch.long)
        t_data = torch.frombuffer(serialized, dtype=torch.uint8)
        return [
            dist.isend(t_size, dst=dst, tag=tag, group=group),
            dist.isend(t_data, dst=dst, tag=tag, group=group),
        ]

    def _recv(self, src: int, tag: int = 0, group=None) -> Any:
        """Blocking recv via gloo-backed cpu_group (CPU tensor P2P)."""
        t_size = torch.tensor([0], dtype=torch.long)
        dist.irecv(t_size, src=src, tag=tag, group=group).wait()
        size = t_size.item()
        if size == 0:
            return []
        t_data = torch.empty(size, dtype=torch.uint8)
        dist.irecv(t_data, src=src, tag=tag, group=group).wait()
        return pickle.loads(t_data.numpy().tobytes())

    def _send_tensor(self, tensor: torch.Tensor, dst: int, tag: int = 0, group=None):
        dist.send(tensor, dst=dst, tag=tag, group=group)

    def _recv_tensor(self, tensor: torch.Tensor, src: int, tag: int = 0, group=None):
        dist.recv(tensor, src=src, tag=tag, group=group)

    # ==================================================================
    # Intra-stage broadcast (stage leader -> non-leaders in same PP stage)
    # ==================================================================

    def _bcast_to_stage_members(self, tensor: torch.Tensor, tag: int):
        """Broadcast tensor from PP stage leader to all non-leader ranks.

        Two-level fan-out to handle the full (ATTN_CP × ATTN_TP) space:
          Level 1: stage leader (attn_cp=0, attn_tp=0) sends to:
            - attn_tp peers (same CP group, attn_tp > 0)
            - CP leaders of other CP groups (attn_cp > 0, attn_tp = 0)
          Level 2: each CP leader (attn_cp > 0, attn_tp = 0) sends to:
            - its own attn_tp peers (attn_tp > 0)

        Non-stage-leader ranks receive from their respective leader:
          - attn_tp > 0, attn_cp = 0: receive from stage leader
          - attn_cp > 0, attn_tp = 0: receive from stage leader
          - attn_cp > 0, attn_tp > 0: receive from their CP leader
        """
        if self.is_pp_stage_leader:
            # Level 1: send to direct members (attn_tp peers + CP leaders)
            for rank in self._pp_stage_member_ranks:
                self._send_tensor(
                    tensor, dst=rank, tag=tag, group=self._world_cpu_group,
                )
        elif not self.is_pp_stage_leader and self.is_attn_tp_leader and self.attn_cp_size > 1:
            # CP leader (attn_cp > 0, attn_tp = 0):
            # First receive from stage leader (Level 1), then forward to
            # own attn_tp peers (Level 2).
            self._recv_tensor(
                tensor, src=self._my_stage_leader_rank,
                tag=tag, group=self._world_cpu_group,
            )
            # Forward to own attn_tp peers
            for rank in self._attn_tp_group_ranks:
                if rank != self.world_rank:
                    self._send_tensor(
                        tensor, dst=rank, tag=tag, group=self._world_cpu_group,
                    )
        elif not self.is_attn_tp_leader and self.attn_cp_size > 1:
            # attn_cp > 0, attn_tp > 0: receive from own CP leader.
            # _my_stage_leader_rank points to the stage leader (attn_cp=0),
            # but we need the CP leader (attn_cp=this, attn_tp=0).
            # The CP leader is attn_tp_group rank 0 (attn_tp=0 in same CP group).
            # The CP leader is attn_tp_group rank 0 (attn_tp=0 in same CP group).
            # attn_tp_size > 1 is guaranteed here: if attn_tp_size == 1 then
            # attn_tp_rank == 0 always, so is_attn_tp_leader is True and this
            # branch (not is_attn_tp_leader) is unreachable.
            _cp_leader = dist.get_global_rank(self.attn_tp_cpu_group, 0)
            self._recv_tensor(
                tensor, src=_cp_leader,
                tag=tag, group=self._world_cpu_group,
            )
        else:
            # attn_tp > 0, attn_cp = 0: receive from stage leader.
            self._recv_tensor(
                tensor, src=self._my_stage_leader_rank,
                tag=tag, group=self._world_cpu_group,
            )

    # ==================================================================
    # PP-level P2P collectives (cross-node safe on world_cpu_group)
    # ==================================================================

    def _pp_all_reduce_min_p2p(self, tensor: torch.Tensor):
        """PP all_reduce MIN via star-pattern P2P on world_cpu_group."""
        leader_rank = self._pp_stage_leader_ranks[0]
        other_leaders = self._pp_stage_leader_ranks[1:]
        tag = self._TAG_PP_AR_MIN

        if self.world_rank == leader_rank:
            result = tensor.item()
            for src in other_leaders:
                other = torch.tensor(0, dtype=torch.int64)
                self._recv_tensor(other, src=src, tag=tag, group=self._world_cpu_group)
                result = min(result, other.item())
            tensor.fill_(result)
            for dst in other_leaders:
                self._send_tensor(tensor, dst=dst, tag=tag, group=self._world_cpu_group)
        else:
            self._send_tensor(tensor, dst=leader_rank, tag=tag, group=self._world_cpu_group)
            self._recv_tensor(tensor, src=leader_rank, tag=tag, group=self._world_cpu_group)

    def _pp_barrier_p2p(self):
        """PP barrier via star-pattern P2P on world_cpu_group."""
        leader_rank = self._pp_stage_leader_ranks[0]
        other_leaders = self._pp_stage_leader_ranks[1:]
        tag = self._TAG_PP_BARRIER
        dummy = torch.tensor([1], dtype=torch.int64)

        if self.world_rank == leader_rank:
            for src in other_leaders:
                self._recv_tensor(dummy, src=src, tag=tag, group=self._world_cpu_group)
            for dst in other_leaders:
                self._send_tensor(dummy, dst=dst, tag=tag, group=self._world_cpu_group)
        else:
            self._send_tensor(dummy, dst=leader_rank, tag=tag, group=self._world_cpu_group)
            self._recv_tensor(dummy, src=leader_rank, tag=tag, group=self._world_cpu_group)


# ===================================================================
# libc / eventfd helpers (module-private)
# ===================================================================

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
_libc.eventfd.restype = ctypes.c_int
_libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.read.restype = ctypes.c_ssize_t
_libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.write.restype = ctypes.c_ssize_t

EFD_SEMAPHORE = 0x1
EFD_NONBLOCK = 0x800


def eventfd(initval=0, flags=0):
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def eventfd_write(fd, val):
    v = ctypes.c_uint64(val)
    n = _libc.write(fd, ctypes.byref(v), ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd write failed: {os.strerror(err)}")


def eventfd_read(fd):
    v = ctypes.c_uint64()
    n = _libc.read(fd, ctypes.byref(v), ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        if err == errno.EAGAIN:
            return 0
        raise OSError(err, f"eventfd read failed: {os.strerror(err)}")
    return v.value


def send_fds(sock: socket.socket, fds: list, extra_data: bytes = b"x"):
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


# ===================================================================
# Layer-wise transfer components
# ===================================================================


class FlexKVLayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_event_fds: List[int] = [
            eventfd(0, EFD_SEMAPHORE) for _ in range(num_layers)
        ]
        self._finished = True
        self.wait_remaining: List[int] = [1] * num_layers

    def reset_for_new_transfer(self):
        self._finished = False
        self.wait_remaining = [1] * self._num_layers

    def wait(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        eventfd_read(self.load_event_fds[layer_index])
        if layer_index == self._num_layers - 1:
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
    """Triple-buffered layer-wise transfer counter using eventfds."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.num_counters = 3
        self.events: List[FlexKVLayerLoadingEvent] = [
            FlexKVLayerLoadingEvent(num_layers) for _ in range(self.num_counters)
        ]
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer: Dict[int, int] = {}

    def register_task(self, task_id: int, producer_id: int):
        self._task_to_producer[task_id] = producer_id

    def register_task_with_explicit_counter_id(self, task_id: int, counter_id: int):
        if counter_id < 0 or counter_id >= self.num_counters:
            raise ValueError(
                f"Invalid counter_id={counter_id}, must be in [0, {self.num_counters})"
            )
        self._task_to_producer[task_id] = counter_id
        self.events[counter_id].reset_for_new_transfer()

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ]._finished, "Producer event should be finished before reuse"
        return self.producer_index

    def set_consumer(self, task_id: int):
        if task_id < 0:
            self.consumer_index = -1
            return
        producer_id = self._task_to_producer.pop(task_id, None)
        if producer_id is not None:
            self.consumer_index = producer_id
        else:
            self.consumer_index = -1

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
        self._task_to_producer.clear()

    def __del__(self):
        for event in self.events:
            event.close()
        self.events.clear()
