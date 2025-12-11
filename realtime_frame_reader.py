from multiprocessing import shared_memory, Value
import subprocess
from enum import Enum
import time
import os
import sys
from enum import Enum
import numpy as np


# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
# TODO: SET TO FALSE
DEBUG_IGNORE_CAMERA = False


# Define Enum here for frontend use
class StreamProtocol(Enum):
    MJPEG = "MJPEG"
    RTSP_ULAW = "rtsp_ulaw"
    RTSP_PCM = "rtsp_pcm"


class RealtimeCamera:
    def __init__(
        self, height, width, ip_and_port="10.26.208.31:8080", protocol=StreamProtocol.MJPEG, device='cuda',
        downsample_scale=3, dtype=None
    ):
        # Lazy Import Torch only in the main process
        import torch
        if dtype is None:
            dtype = torch.float16

        self.ip_and_port = ip_and_port
        self.downsample_scale = downsample_scale
        self.protocol = protocol
        self.device = device
        self.dtype = dtype
        self.device = device
        self.worker = None  # Initialize to avoid AttributeError

        print(f"Main: Probing {self.ip_and_port}...")
        # --- 1. Probe & Config ---
        self.width, self.height = width, height
        print(f"Main: Resolution {self.width}x{self.height}")

        self.shape = (self.height // downsample_scale, self.width // downsample_scale, 3)
        self.read_count = 0

        # --- 2. Allocate SHARED MEMORY (CPU) ---
        # Frame Buffer (H * W * 3 * 3 frames)
        frame_bytes = int(np.prod(self.shape) * np.dtype(np.uint8).itemsize)
        self.shm = shared_memory.SharedMemory(create=True, size=frame_bytes * 3)

        # Timestamp Buffer (8 bytes * 3 frames) - For int64 nanoseconds
        self.time_shm = shared_memory.SharedMemory(create=True, size=8 * 3)

        # Atomic Integer for triple-buffer synchronization (file-based for subprocess isolation)
        import tempfile
        self.idx_path = os.path.join(tempfile.gettempdir(), f"camera_idx_{self.shm.name}.dat")
        with open(self.idx_path, 'wb') as f:
            f.write((0).to_bytes(4, 'little'))  # Initialize to 0

        print("Main: START.... (Spawning Worker)")
        # --- MOVED UP: Start Worker BEFORE GPU Init ---
        if not DEBUG_IGNORE_CAMERA:
            # Find the backend script (should be in same directory)
            backend_script = os.path.join(os.path.dirname(__file__), 'camera_worker_impl.py')
            if not os.path.exists(backend_script):
                backend_script = 'camera_worker_impl.py'  # Fallback to relative

            # Build command line arguments
            args = [
                sys.executable,
                backend_script,
                '--ip', self.ip_and_port,
                '--protocol', self.protocol.value,
                '--shm-name', self.shm.name,
                '--time-shm-name', self.time_shm.name,
                '--height', str(self.shape[0]),
                '--width', str(self.shape[1]),
                '--downsample-scale', str(downsample_scale),
            ]

            # Launch subprocess
            self.worker = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

            # Start output monitoring thread
            import threading
            def monitor_output():
                try:
                    for line in self.worker.stdout:
                        print(f"[Worker] {line.rstrip()}")
                except:
                    pass

            self.monitor_thread = threading.Thread(target=monitor_output, daemon=True)
            self.monitor_thread.start()

        # --- 3. Pre-Create CPU Tensor Views ---
        # Frame Views
        self.cpu_buffers = [
            torch.from_numpy(
                np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf[i * frame_bytes:(i + 1) * frame_bytes])
            ) for i in range(3)
        ]

        # Timestamp View (Numpy is easier for scalar reading)
        self.timestamp_buffer = np.ndarray((3,), dtype=np.int64, buffer=self.time_shm.buf)

        # --- 4. PRE-ALLOCATE GPU MEMORY (SLOW STEP) ---
        # This creates ONE chunk of VRAM. We never allocate again.
        # 0-255 range, float16
        self.gpu_buffer = torch.zeros(
            self.shape,  # (H, W, 3)
            dtype=self.dtype,
            device=self.device
        )

        if DEBUG_IGNORE_CAMERA:
            import cv2
            img = cv2.resize(cv2.imread("img_1.png"), (self.width // self.downsample_scale, self.height // self.downsample_scale))[..., [2, 1, 0]]
            img = torch.from_numpy(img)
            for buf in self.cpu_buffers:
                buf[:] = img
            self.gpu_buffer.copy_(self.cpu_buffers[0], non_blocking=True)

    def _probe_dimensions(self):
        return 1920, 1080

    def read(self):
        """
        Returns: ret (bool), data (dict)
        """
        if not DEBUG_IGNORE_CAMERA:
            # 1. Atomic Read of Index (file-based)
            with open(self.idx_path, 'rb') as f:
                idx = int.from_bytes(f.read(4), 'little')

            # 2. Async Copy to GPU (Zero Allocation)
            ts = int(self.timestamp_buffer[idx])
            self.read_count += 1
            while ts == 0:
                self.read_count += 1
                time.sleep(0.1)
                if self.read_count % 20 == 0:
                    print("Camera is in an invalid state")
                if self.read_count % 500 == 0:
                    return False, {
                        'frame': self.gpu_buffer,  # API Compat
                        'timestamp_ns': ts
                    }
                ts = int(self.timestamp_buffer[idx])
            self.gpu_buffer.copy_(self.cpu_buffers[idx], non_blocking=True)
        else:
            ts = time.time_ns()
        return True, {
            'frame': self.gpu_buffer,  # API Compat
            'timestamp_ns': ts
        }

    def release(self):
        if self.worker:
            try:
                self.worker.terminate()
                self.worker.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.worker.kill()
            except:
                pass

        # Cleanup index file
        try:
            os.remove(self.idx_path)
        except:
            pass

        self.shm.close()
        self.time_shm.close()
        try:
            self.shm.unlink()
        except:
            print('Failed to unlink [self.shm]')

        #################################################

        try:
            self.time_shm.unlink()
        except:
            print('Failed to unlink [self.time_shm]')


import time
from enum import Enum
import numpy as np
import sounddevice as sd
import threading  # Reintroducing threading for the asynchronous beep

# Lazy Import Torch only in the main process
import torch
import cv2


# Define Enum here for frontend use
class StreamProtocol(Enum):
    MJPEG = "MJPEG"
    RTSP_ULAW = "rtsp_ulaw"
    RTSP_PCM = "rtsp_pcm"


# Configuration for the beep sound
BEEP_FREQUENCY = 1000  # Hz (A4)
BEEP_DURATION_S = 3.0  # seconds (Changed to 3.0s as requested)
PAUSE_DURATION_S = 0.5  # seconds (New constant for pause duration)
BEEP_SAMPLE_RATE = 44100  # samples per second


class SingleShotCamera:
    """
    A single-process camera reader that retries upon failure.
    - Plays a continuous, interruptible beep sequence in a separate thread on failure.
    - Uses a short delay to control the retry frequency.
    """

    def __init__(
            self, height, width, ip_and_port="10.26.208.31:8080", protocol=StreamProtocol.MJPEG, device='cuda',
            downsample_scale=3, dtype=None
    ):
        if dtype is None:
            dtype = torch.float16

        self.ip_and_port = ip_and_port
        self.downsample_scale = downsample_scale
        self.protocol = protocol
        self.device = device
        self.dtype = dtype

        # Threading and Audio Management
        self._stop_beeping_event = threading.Event()
        self._beep_thread = None

        # Determine the full stream URL based on protocol
        if self.protocol == StreamProtocol.MJPEG:
            self.stream_url = f"http://{self.ip_and_port}/video"
        elif 'rtsp' in self.protocol.value:
            self.stream_url = f"rtsp://{self.ip_and_port}/stream"
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        # --- 1. Config ---
        self.width, self.height = width, height
        self.target_shape = (
            self.height // downsample_scale,
            self.width // downsample_scale,
            3
        )

        # --- 2. PRE-ALLOCATE GPU MEMORY ---
        self.gpu_buffer = torch.zeros(
            self.target_shape,
            dtype=self.dtype,
            device=self.device
        )

        # --- 3. Pre-generate the beep audio signal ---
        t = np.linspace(0, BEEP_DURATION_S, int(BEEP_SAMPLE_RATE * BEEP_DURATION_S), False)
        # Generate sine wave: A4 (440 Hz) and scale amplitude (0.3)
        note = np.sin(BEEP_FREQUENCY * t * 2 * np.pi)
        self._beep_audio = (note*0.2).astype(np.float32)

    def _continuous_beep_task(self):
        """
        The target function for the beeping thread. Loops continuously until
        the stop event is set.
        """
        while not self._stop_beeping_event.is_set():
            # Play 3-second beep
            sd.play(self._beep_audio, samplerate=BEEP_SAMPLE_RATE)

            # Use wait() with a timeout matching BEEP_DURATION_S. If the stop event
            # is set during playback, wait() returns True, and we exit immediately.
            if self._stop_beeping_event.wait(BEEP_DURATION_S):
                sd.stop()
                return

            # 0.5 second pause (interruptible)
            if self._stop_beeping_event.wait(PAUSE_DURATION_S):
                sd.stop()
                return

    def _manage_beeping(self, start):
        """
        Manages the start and stop state of the continuous beeping thread.
        """
        if start:
            # Start the thread only if it's not running
            if self._beep_thread is None or not self._beep_thread.is_alive():
                self._stop_beeping_event.clear()  # Clear any previous stop signal
                self._beep_thread = threading.Thread(target=self._continuous_beep_task, daemon=True)
                self._beep_thread.start()
        else:  # Stop
            # Stop the thread immediately
            if self._beep_thread and self._beep_thread.is_alive():
                self._stop_beeping_event.set()  # Signal the thread to exit its loop
                sd.stop()  # Crucial: Immediately silence any playing audio
                # Rely on the thread to exit cleanly on its next check

    def read(self):
        """
        Reads a single frame, retrying indefinitely upon failure.
        - Starts continuous beeping on failure.
        - Stops beeping immediately on success.

        Returns: ret (bool), data (dict)
        """
        retry_delay_s = 0.5  # Pacing the connection attempts (small delay for robustness)
        retry_count = 0
        if DEBUG_IGNORE_CAMERA:
            return True, {
                'frame': self.gpu_buffer,
                'timestamp_ns': time.time_ns()
            }

        while True:
            retry_count += 1

            # --- 1. Capture Time Stamp (Crucial: BEFORE the potentially slow cv2.VideoCapture/read) ---
            timestamp_ns = time.time_ns()

            # --- 2. Setup Capture ---
            cap = cv2.VideoCapture(self.stream_url)

            if not cap.isOpened():
                cap.release()
                self._manage_beeping(start=True)
                time.sleep(retry_delay_s)
                continue

            # --- 3. Read Frame ---
            ret, frame = cap.read()

            # --- 4. Cleanup (Crucial: immediately release) ---
            cap.release()

            if not ret:
                self._manage_beeping(start=True)
                time.sleep(retry_delay_s)
                continue

            # --- 5. Success: Stop Beeping and Process Frame ---
            self._manage_beeping(start=False)

            # Downsample
            if self.downsample_scale > 1:
                frame = cv2.resize(frame, (self.target_shape[1], self.target_shape[0]), interpolation=cv2.INTER_LINEAR)

            # Convert BGR (OpenCV default) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create CPU tensor view (no memory copy here yet)
            cpu_tensor = torch.from_numpy(frame)

            # Async Copy to GPU
            self.gpu_buffer.copy_(cpu_tensor, non_blocking=True)

            return True, {
                'frame': self.gpu_buffer,
                'timestamp_ns': timestamp_ns
            }

    def release(self):
        """
        Ensures the beeping thread is stopped when the camera object is destroyed.
        """
        self._manage_beeping(start=False)
        # Explicitly wait for the thread to finish for clean shutdown
        if self._beep_thread and self._beep_thread.is_alive():
            self._beep_thread.join(timeout=1.0)


if __name__ == "__main__":
    # Import these ONLY in the main process
    import torch
    import cv2

    # Initialize
    cam = SingleShotCamera(height=1080, width=1440)

    # Warmup CUDA
    print("Main: Warming up CUDA...")
    torch.ones(1).cuda()

    print("Main: Camera started. Press 'q' to quit.")

    try:
        while True:
            ret, data = cam.read()

            # Don't preview if we have no data yet
            if not ret:
                time.sleep(0.01)
                continue

            # --- VERIFICATION & PREVIEW ---

            # 1. Take the FLOAT16 GPU frame (0-255)
            gpu_frame = data['frame']  # shape (3, H, W)

            # 2. Convert to Byte/CPU/Numpy for visualization
            # We must permute back to HWC for OpenCV: (3, H, W) -> (H, W, 3)
            # .byte() casts float16 -> uint8
            # .cpu() moves to RAM
            # .numpy() converts to numpy
            preview_frame = gpu_frame.byte().cpu().numpy()

            # 3. Convert RGB to BGR for OpenCV
            preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR)

            # 4. Add Timestamp Overlay
            ts_str = f"TS: {data['timestamp_ns']}"
            cv2.putText(preview_frame, ts_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 5. Show
            cv2.imshow("Float16 GPU -> CPU Verification", preview_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()