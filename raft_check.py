import torch
import numpy as np
import cv2
import time
import multiprocessing
from multiprocessing import shared_memory
import atexit
import subprocess
import sys
import os
import tempfile

# ============================================================================
# VISUALIZATION TOOLS (TENSOR OPTIMIZED)
# ============================================================================
class PointRenderer:
    """
    High-performance point renderer using vectorized tensor operations.
    """

    def __init__(self, max_h, max_w, device='cuda'):
        # Ensure device is correctly set up
        if str(device).lower() == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda', torch.cuda.current_device())
        else:
            self.device = torch.device('cpu')

        self.H = max_h
        self.W = max_w
        print(f"✓ PointRenderer initialized on {self.device}")

    @torch.no_grad()
    def draw(self, frame_tensor, points, valid_mask):
        """
        Draws points onto the frame_tensor.
        frame_tensor: (3, H, W) BGR (or RGB) tensor, float (on vis_dev)
        points: (N, 2) [x, y] coordinates (on vis_dev)
        valid_mask: (N, ) boolean mask (on vis_dev)
        """
        if frame_tensor.device != self.device: frame_tensor = frame_tensor.to(self.device)

        vis_frame = frame_tensor.clone()
        coords = points.long()

        # Clamp points to prevent out-of-bounds indexing (W-2, H-2 to draw 2x2 box)
        coords[:, 0] = coords[:, 0].clamp(0, self.W - 2)
        coords[:, 1] = coords[:, 1].clamp(0, self.H - 2)

        # Extract coordinates and mask
        x = coords[:, 0]
        y = coords[..., 1]

        # Filter for valid points
        idx_x = x[valid_mask]
        idx_y = y[valid_mask]

        if len(idx_x) > 0:
            # Green dots (2x2) BGR format (0=B, 1=G, 2=R)

            # Center pixel
            vis_frame[0, idx_y, idx_x] = 0
            vis_frame[1, idx_y, idx_x] = 255
            vis_frame[2, idx_y, idx_x] = 0

            # Corner pixel (for a 2x2 visibility boost)
            vis_frame[0, idx_y + 1, idx_x + 1] = 0
            vis_frame[1, idx_y + 1, idx_x + 1] = 255
            vis_frame[2, idx_y + 1, idx_x + 1] = 0

        return vis_frame

    @torch.no_grad()
    def draw(self, frame_tensor, points, valid_mask):
        """
        frame_tensor: (3, H, W)
        points: (N, 2)
        valid_mask: (N,)
        Draws 5×5 GREEN squares centered at each point.
        """

        if frame_tensor.device != self.device:
            frame_tensor = frame_tensor.to(self.device)

        vis = frame_tensor.clone()

        # Filter + convert
        pts = points[valid_mask].long()
        if pts.numel() == 0:
            return vis

        # We want a 5×5 block centered on the point:
        # Offsets from -2 to +2
        offsets = torch.stack(
            torch.meshgrid(
                torch.arange(-2, 3, device=self.device),
                torch.arange(-2, 3, device=self.device),
                indexing="ij"
            ),
            dim=-1
        ).reshape(-1, 2)  # shape: (25, 2)

        # Clamp so 5×5 fits in the image
        pts[:, 0].clamp_(2, self.W - 3)
        pts[:, 1].clamp_(2, self.H - 3)

        # Apply offsets → (P, 25, 2)
        pts25 = pts[:, None, :] + offsets[None, :, :]

        # Flatten all points → (P*25, 2)
        flat = pts25.reshape(-1, 2)
        ys = flat[:, 1]
        xs = flat[:, 0]

        # GREEN (BGR)
        green = torch.tensor([0.0, 255.0, 0.0], device=self.device)[:, None]

        # Single fused write (very fast)
        vis[:, ys, xs] = green

        return vis

# ============================================================================
# SIMPLE CROSS-PROCESS SYNCHRONIZATION
# ============================================================================
class SimpleEvent:
    """File-based event for cross-process signaling (works on Windows/Unix)"""

    def __init__(self, name):
        self.name = name
        self.path = os.path.join(tempfile.gettempdir(), f"raft_event_{name}.flag")
        self.clear()

    def set(self):
        """Signal the event"""
        with open(self.path, 'wb') as f:
            f.write(b'\x01')

    def clear(self):
        """Clear the event"""
        try:
            with open(self.path, 'wb') as f:
                f.write(b'\x00')
        except:
            pass

    def wait(self, timeout=None):
        """Wait for the event to be set"""
        import time
        start = time.time()
        while True:
            try:
                with open(self.path, 'rb') as f:
                    if f.read(1) == b'\x01':
                        return True
            except:
                pass

            if timeout and (time.time() - start) > timeout:
                return False
            time.sleep(0.001)  # 1ms polling

    def is_set(self):
        """Check if event is set"""
        try:
            with open(self.path, 'rb') as f:
                return f.read(1) == b'\x01'
        except:
            return False

    def cleanup(self):
        """Remove the event file"""
        try:
            os.remove(self.path)
        except:
            pass


# ============================================================================
# API: SUBPROCESS CLIENT (COMPLETE ISOLATION)
# ============================================================================
class RaftTracker:
    """
    Client-side API that communicates with the RAFT model subprocess.
    ZERO knowledge of ONNX/DML - complete import isolation.
    """

    def __init__(self, H, W, num_landmarks, dtype=torch.float16, device='cuda', model_type='small'):
        self.H = H
        self.W = W
        self.num_landmarks = num_landmarks
        self.dtype = dtype
        self.first_frame = True
        self.proc = None  # Initialize to avoid AttributeError in cleanup

        # --- DEVICE LOGIC (MATCHING ORIGINAL CLASS) ---
        self.device_str = str(device).lower()
        if self.device_str in ['igpu', 'dml', 'directml']:
            self.exec_device = torch.device('cpu')
        elif 'cuda' in self.device_str and torch.cuda.is_available():
            self.exec_device = torch.device('cuda', torch.cuda.current_device())
        else:
            self.exec_device = torch.device('cpu')

        print("[API] Initializing Isolated Subprocess RAFT Tracker...")

        # 1. Setup Shared Memory
        d_size = 2 if dtype == torch.float16 else 4

        # Input Layout: [Img1 (3,H,W)][Img2 (3,H,W)][Points (N,2)]
        self.img_size = 3 * H * W * d_size
        self.pts_size = num_landmarks * 4 * d_size
        input_bytes = self.img_size + self.img_size + self.pts_size

        self.shm_in = shared_memory.SharedMemory(create=True, size=input_bytes)

        # Output Layout: [NewPoints (N,2)][Mask (N,)]
        mask_size = num_landmarks * 1  # bool
        output_bytes = self.pts_size + mask_size

        self.shm_out = shared_memory.SharedMemory(create=True, size=output_bytes)

        # 2. Sync Primitives (file-based for cross-process compatibility)
        import uuid
        session_id = str(uuid.uuid4())[:8]
        self.evt_start = SimpleEvent(f"{session_id}_start")
        self.evt_done = SimpleEvent(f"{session_id}_done")
        self.evt_stop = SimpleEvent(f"{session_id}_stop")

        # 3. Launch ISOLATED subprocess (separate Python interpreter)
        dtype_str = 'float16' if dtype == torch.float16 else 'float32'

        # Find the backend script (should be in same directory)
        backend_script = os.path.join(os.path.dirname(__file__), 'raft_tracker_backend.py')
        if not os.path.exists(backend_script):
            backend_script = 'raft_tracker_backend.py'  # Fallback to relative

        # Build command line arguments
        args = [
            # We dont use sys.executable this because anaconda messes with onnx
            "python",  # Current Python interpreter
            backend_script,
            '--shm-in', self.shm_in.name,
            '--shm-out', self.shm_out.name,
            '--height', str(H),
            '--width', str(W),
            '--num-points', str(num_landmarks),
            '--dtype', dtype_str,
            '--device', self.device_str,
            '--model-type', model_type,
            '--evt-start', self.evt_start.name,
            '--evt-done', self.evt_done.name,
            '--evt-stop', self.evt_stop.name,
        ]

        # Launch subprocess
        self.proc = subprocess.Popen(
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
                for line in self.proc.stdout:
                    print(f"[Backend] {line.rstrip()}")
            except:
                pass

        self.monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        self.monitor_thread.start()

        # 4. Async Init - Don't wait here
        print("[API] Backend subprocess launched (loading asynchronously...)")
        self._initialized = False

        atexit.register(self.cleanup)

    def track_points(self, img1, img2, points):
        """
        img1, img2: (3, H, W) Tensor
        points: (N, 2) Tensor
        """
        n_actual = points.shape[0]

        # 1. Data Transfer
        cpu_img1 = img1.cpu() if img1.device.type != 'cpu' else img1
        cpu_img2 = img2.cpu() if img2.device.type != 'cpu' else img2
        cpu_pts = points.cpu() if points.device.type != 'cpu' else points

        # Write Img1
        np_img1 = np.ndarray((3, self.H, self.W), dtype=np.float16 if self.dtype == torch.float16 else np.float32,
                             buffer=self.shm_in.buf, offset=0)
        np_img1[:] = cpu_img1.numpy()[:]

        # Write Img2
        np_img2 = np.ndarray((3, self.H, self.W), dtype=np.float16 if self.dtype == torch.float16 else np.float32,
                             buffer=self.shm_in.buf, offset=self.img_size)
        np_img2[:] = cpu_img2.numpy()[:]

        # Write Points
        np_pts = np.ndarray((self.num_landmarks, 2), dtype=np.float32,
                            buffer=self.shm_in.buf, offset=self.img_size + self.img_size)
        np_pts[:n_actual] = cpu_pts.numpy()[:]

        # 2. Signal Process
        self.evt_start.set()

        # 3. Wait
        if not self.evt_done.wait(timeout=205):
            raise RuntimeError("Backend subprocess timed out")
        self.evt_done.clear()

        # 4. Read Result
        np_out_pts = np.ndarray((self.num_landmarks, 2),
                                dtype=np.float32,
                                buffer=self.shm_out.buf, offset=0)
        new_points = torch.from_numpy(np_out_pts.copy())[:n_actual]

        np_out_mask = np.ndarray((self.num_landmarks,), dtype=np.bool_, buffer=self.shm_out.buf, offset=self.pts_size)
        valid_mask = torch.from_numpy(np_out_mask.copy())[:n_actual]

        # Return on requested device
        if self.exec_device.type != 'cpu':
            new_points = new_points.to(self.exec_device)
            valid_mask = valid_mask.to(self.exec_device)
        if self.first_frame:
            self.first_frame = False
            return self.track_points(img1, img2, points)
        return new_points, valid_mask, None, None

    def cleanup(self):
        print("[API] Cleaning up resources...")
        try:
            self.evt_stop.set()
            self.evt_start.set()
        except Exception:
            pass

        # Wait for subprocess to exit gracefully
        if self.proc:
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

        # Cleanup events
        try:
            self.evt_start.cleanup()
            self.evt_done.cleanup()
            self.evt_stop.cleanup()
        except:
            pass

        # Cleanup shared memory
        try:
            self.shm_in.close()
            self.shm_in.unlink()
            self.shm_out.close()
            self.shm_out.unlink()
        except:
            pass

    def __del__(self):
        self.cleanup()


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Support Windows Multiprocessing
    multiprocessing.freeze_support()

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    WIDTH = 640//2
    WIDTH = (WIDTH // 8) * 8
    HEIGHT = 360//2
    HEIGHT = (HEIGHT // 8) * 8
    # Use extreme downscaling ?
    NUM_POINTS = 512

    # Hardware Selection: 'dml' (iGPU), 'cuda' (Nvidia), 'cpu'
    DEVICE_REQ = 'dml'

    # Precision: torch.float16 or torch.float32
    DTYPE = torch.float32

    # Model Selection: 'small' or 'large'
    MODEL_TYPE = 'small'

    print(f"=== Hybrid RAFT Live | {WIDTH}x{HEIGHT} | Mode: {DEVICE_REQ} | Dtype: {DTYPE} ===")

    # Warmup
    if torch.cuda.is_available():
        torch.ones(1).cuda()

    # Determine visualization device
    vis_dev = 'cpu'
    renderer = PointRenderer(HEIGHT, WIDTH, device=vis_dev)

    # Initialize Tracker
    tracker = RaftTracker(HEIGHT, WIDTH, NUM_POINTS, dtype=DTYPE, device=DEVICE_REQ, model_type=MODEL_TYPE)

    # Video Setup
    cap = cv2.VideoCapture(r"C:\Users\cjlyn\Downloads\Phone Link\20251015_094321.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Init Points Grid
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, HEIGHT - 1, int(np.sqrt(NUM_POINTS))),
        torch.linspace(0, WIDTH - 1, int(np.sqrt(NUM_POINTS))),
        indexing='ij'
    )
    initial_pts = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1).float()

    current_points = initial_pts
    initial_points_dev = current_points.clone()

    print("\n>>> RUNNING. Press 'q' to quit. Press 'r' to reset.")
    t_last = time.time()
    frames = 0
    prev_frame_t = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Convert HWC -> CHW
        t_in = torch.from_numpy(frame).permute(2, 0, 1).float()
        rgb_in = t_in.flip(0)  # BGR to RGB

        if prev_frame_t is None:
            prev_frame_t = rgb_in.clone()
            continue

        # Track
        t0 = time.perf_counter()
        new_points, valid_mask, _, _ = tracker.track_points(prev_frame_t, rgb_in, current_points)
        dt = time.perf_counter() - t0

        # Render
        vis_points = new_points.to(vis_dev).float()
        vis_mask = valid_mask.to(vis_dev)
        vis_frame = renderer.draw(t_in.to(vis_dev).float(), vis_points, vis_mask)
        vis_np = vis_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Update
        current_points = new_points[valid_mask]#torch.where(valid_mask.unsqueeze(-1), new_points, initial_points_dev)
        prev_frame_t = rgb_in.clone()

        frames += 1
        if time.time() - t_last > 1.0:
            print(f"FPS: {frames} | Track: {dt * 1000:.2f}ms")
            frames = 0
            t_last = time.time()

        cv2.imshow("Hybrid RAFT", vis_np)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            current_points = initial_points_dev.clone()
            print("Points Reset")

    cap.release()
    cv2.destroyAllWindows()


def main():
    multiprocessing.freeze_support()

    WIDTH = 640*2
    # WIDTH = (WIDTH // 8) * 8
    HEIGHT = 360*2
    # HEIGHT = (HEIGHT // 8) * 8
    NUM_POINTS = 512
    DEVICE_REQ = 'dml'
    DTYPE = torch.float32
    MODEL_TYPE = 'small'

    print(f"=== Hybrid RAFT Live | {WIDTH}x{HEIGHT} | Mode: {DEVICE_REQ} | Dtype: {DTYPE} ===")

    if torch.cuda.is_available():
        torch.ones(1).cuda()

    vis_dev = 'cpu'
    renderer = PointRenderer(HEIGHT, WIDTH, device=vis_dev)
    tracker = RaftTracker(HEIGHT, WIDTH, NUM_POINTS, dtype=DTYPE, device=DEVICE_REQ, model_type=MODEL_TYPE)

    # --- VIDEO SOURCE ---
    cap = cv2.VideoCapture(r"C:\Users\cjlyn\Downloads\Phone Link\20251015_094321.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # --- NEW: SAVE VISUALIZATION OUTPUT ---
    output_path = "raft_visualization_output.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1: fps = 30   # fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

    # --- POINT GRID ---
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, HEIGHT - 1, int(np.sqrt(NUM_POINTS))),
        torch.linspace(0, WIDTH - 1, int(np.sqrt(NUM_POINTS))),
        indexing='ij'
    )
    initial_pts = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=1).float()
    current_points = initial_pts
    initial_points_dev = current_points.clone()

    print("\n>>> RUNNING. Press 'q' to quit. Press 'r' to reset.")
    t_last = time.time()
    frames = 0
    prev_frame_t = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        t_in = torch.from_numpy(frame).permute(2, 0, 1).float()
        rgb_in = t_in.flip(0)

        if prev_frame_t is None:
            prev_frame_t = rgb_in.clone()
            continue

        # Track
        t0 = time.perf_counter()
        new_points, valid_mask, _, _ = tracker.track_points(prev_frame_t, rgb_in, current_points)
        dt = time.perf_counter() - t0

        # Render
        vis_points = new_points.to(vis_dev).float()
        vis_mask = valid_mask.to(vis_dev)
        vis_frame = renderer.draw(t_in.to(vis_dev).float()*0.0, vis_points, vis_mask)
        vis_np = vis_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # --- NEW: WRITE TO OUTPUT VIDEO ---
        writer.write(vis_np)

        # Update for next frame
        current_points = new_points[valid_mask]
        prev_frame_t = rgb_in.clone()

        frames += 1
        if time.time() - t_last > 1.0:
            print(f"FPS: {frames} | Track: {dt * 1000:.2f}ms")
            frames = 0
            t_last = time.time()

        cv2.imshow("Hybrid RAFT", vis_np)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            current_points = initial_points_dev.clone()
            print("Points Reset")

    # --- CLEANUP ---
    cap.release()
    writer.release()     # << REQUIRED
    cv2.destroyAllWindows()

    print(f"\n✔ Saved visualization: {output_path}")


if __name__ == "__main__":
    main()