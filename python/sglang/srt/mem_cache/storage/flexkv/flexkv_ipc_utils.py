import ctypes
import os
import socket
import struct
import sys

# ---- libc / eventfd ----
libc = ctypes.CDLL("libc.so.6", use_errno=True)

libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
libc.eventfd.restype = ctypes.c_int

libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.read.restype = ctypes.c_ssize_t

libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.write.restype = ctypes.c_ssize_t

EFD_SEMAPHORE = 0x1
EFD_NONBLOCK = 0x800


def eventfd(initval=0, flags=0):
    # create and return an eventfd object
    fd = libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd

def eventfd_write(fd, val):
    v = ctypes.c_uint64(val)
    buf = ctypes.byref(v)
    n = libc.write(fd, buf, ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd write failed: {os.strerror(err)}")

def eventfd_read(fd):
    """Blocking read from eventfd."""
    v = ctypes.c_uint64()
    buf = ctypes.byref(v)
    n = libc.read(fd, buf, ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        # EAGAIN means non-blocking read with no data, which is fine in some cases
        # but here we usually expect blocking reads.
        if err == 11: # EAGAIN
             return 0
        raise OSError(err, f"eventfd read failed: {os.strerror(err)}")
    return v.value


def send_fds(sock: socket.socket, fds: list, extra_data: bytes = b"x"):
    """Send multiple fds + extra_data via Unix domain socket."""
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], anc)

def recv_fds(sock: socket.socket, num_fds: int):
    """Receive multiple fds + extra_data via Unix domain socket."""
    data_buf = bytearray(256)
    # Ancillary buffer size: need space for num_fds integers
    anc_buf_size = socket.CMSG_SPACE(num_fds * struct.calcsize("i"))
    
    nbytes, ancdata, flags, addr = sock.recvmsg_into([data_buf], anc_buf_size, anc_buf_size)
    data = bytes(data_buf[:nbytes])

    fds = []
    for level, ctype, cdata in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            num_received = len(cdata) // struct.calcsize("i")
            fds = list(struct.unpack(f"{num_received}i", cdata[:num_received * struct.calcsize("i")]))
            break
    if not fds:
        raise RuntimeError("did not receive fds via SCM_RIGHTS")
    return fds, data

# ---- CUDA Runtime (via ctypes) ----
# Try to find libcudart.so
def load_cudart():
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    for lib in candidates:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    return None

cudart = load_cudart()

if cudart:
    cudart.cudaLaunchHostFunc.argtypes = [
        ctypes.c_void_p, # stream (pointer)
        ctypes.CFUNCTYPE(None, ctypes.c_void_p), # fn
        ctypes.c_void_p  # userData
    ]
    cudart.cudaLaunchHostFunc.restype = ctypes.c_int
else:
    print("Warning: libcudart.so not found. CUDA callbacks will not work.", file=sys.stderr)