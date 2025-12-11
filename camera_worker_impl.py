#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Worker Backend - Isolated Subprocess
This script runs in a completely separate Python process.
Imports here will NEVER contaminate the calling code.
"""

# Force UTF-8 output encoding on Windows
import sys

if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import av
import time
import numpy as np
from multiprocessing import shared_memory, Value
from enum import Enum
import argparse


# Define Enum here
class StreamProtocol(Enum):
    MJPEG = "MJPEG"
    RTSP_ULAW = "rtsp_ulaw"
    RTSP_PCM = "rtsp_pcm"


def camera_worker_loop(ip, protocol, shm_name, time_shm_name, shape, downsample_scale):
    """
    Completely isolated worker.
    Does NOT import torch, cv2, or any project-specific modules.
    """
    print("Worker: RUNNING.... (Immediate Start)")
    sys.stdout.flush()

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        frame_size = int(np.prod(shape) * np.dtype(np.uint8).itemsize)

        buffers = [
            np.ndarray(shape, dtype=np.uint8, buffer=shm.buf[i * frame_size:(i + 1) * frame_size])
            for i in range(3)
        ]

        time_shm = shared_memory.SharedMemory(name=time_shm_name)
        timestamps = np.ndarray((3,), dtype=np.int64, buffer=time_shm.buf)

        # Create shared index using memory-mapped approach
        # Parent passes this via shared memory or we create a simple counter
        # For simplicity, we'll use a file-based approach similar to events
        import tempfile
        idx_path = os.path.join(tempfile.gettempdir(), f"camera_idx_{shm_name}.dat")

        def get_idx():
            try:
                with open(idx_path, 'rb') as f:
                    return int.from_bytes(f.read(4), 'little')
            except:
                return 0

        def set_idx(val):
            with open(idx_path, 'wb') as f:
                f.write(val.to_bytes(4, 'little'))

        set_idx(0)  # Initialize

    except Exception as e:
        print(f"Worker: Critical Shared Memory Error: {e}")
        sys.stdout.flush()
        return

    # Options for low latency
    options = {
        'fflags': 'nobuffer', 'flags': 'low_delay', 'framedrop': '1',
        'strict': 'experimental', 'probesize': '32', 'analyzeduration': '0',
        'threads': '1'
    }
    if protocol != StreamProtocol.MJPEG:
        options['rtsp_transport'] = 'tcp'
        options['rtsp_flags'] = 'prefer_tcp'

    url = f"http://{ip}/video"
    print(f"Worker: Connecting to {url}...")
    sys.stdout.flush()

    while True:
        container = None
        try:
            container = av.open(url, options=options, timeout=5.0)
            stream = container.streams.video[0]
            stream.thread_type = 'NONE'  # Fastest decoding

            print("Worker: Connection Successful. Starting stream.")
            sys.stdout.flush()

            for packet in container.demux(stream):
                for frame in packet.decode():
                    current_idx = get_idx()
                    next_idx = (current_idx + 1) % 3

                    # Decode directly to numpy
                    img = frame.to_ndarray(format='rgb24')

                    # Slice/Downsample
                    if downsample_scale > 1:
                        img = img[::downsample_scale, ::downsample_scale]

                    buffers[next_idx][:] = img
                    timestamps[next_idx] = time.time_ns()
                    set_idx(next_idx)
        except Exception as e:
            print(f"Worker: Stream Error: {e}") # Optional: reduce noise
            time.sleep(0.5)
        finally:
            if container:
                try:
                    container.close()
                except:
                    pass


def main():
    parser = argparse.ArgumentParser(description='Camera Worker Backend Subprocess')
    parser.add_argument('--ip', required=True, help='Camera IP and port')
    parser.add_argument('--protocol', required=True, help='Stream protocol')
    parser.add_argument('--shm-name', required=True, help='Shared memory name for frames')
    parser.add_argument('--time-shm-name', required=True, help='Shared memory name for timestamps')
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--downsample-scale', type=int, required=True)

    args = parser.parse_args()

    # Reconstruct protocol enum
    protocol = StreamProtocol(args.protocol)

    # Shape is (H, W, 3)
    shape = (args.height, args.width, 3)

    # Run worker loop
    try:
        camera_worker_loop(
            args.ip,
            protocol,
            args.shm_name,
            args.time_shm_name,
            shape,
            args.downsample_scale
        )
    except KeyboardInterrupt:
        print("Worker: Interrupted")
    except Exception as e:
        print(f"Worker: ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Worker: Shutting down.")


if __name__ == "__main__":
    import os

    main()