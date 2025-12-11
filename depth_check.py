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
# VISUALIZATION TOOLS (CUDA OPTIMIZED)
# ============================================================================
class DepthMapColorizer:
    def __init__(self, dtype=torch.float32, device='cuda', lut_size=4096):
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())

        self.lut_size = lut_size
        self.log_min = -2.0
        self.log_max = 4.699
        self.log_range = self.log_max - self.log_min

        lut_distances = torch.logspace(self.log_min, self.log_max, self.lut_size, dtype=dtype, device=self.device)
        self.color_lut_rgb = self.build_color_lut_rgb(lut_distances).flip(dims=[-1])
        print(f"✓ Colorizer initialized: {self.lut_size} entries on {self.device}")

    def build_color_lut_rgb(self, z_values):
        t = (torch.log10(torch.clamp(z_values, 0.01, 50000.0)) - self.log_min) / self.log_range
        t = torch.clamp(t, 0.0, 1.0)
        colors = torch.zeros((len(z_values), 3), dtype=self.dtype, device=self.device)

        # Vectorized Color Gradient Logic
        mask = t <= 0.10
        lt = (t[mask] - 0.0) / 0.10
        colors[mask] = torch.stack((torch.full_like(lt, 255), 255 * (1 - lt), 255 * (1 - lt)), dim=1)

        mask = (t > 0.10) & (t <= 0.25)
        lt = (t[mask] - 0.10) / 0.15
        colors[mask] = torch.stack((torch.full_like(lt, 255), 127 * lt, torch.zeros_like(lt)), dim=1)

        mask = (t > 0.25) & (t <= 0.35)
        lt = (t[mask] - 0.25) / 0.10
        colors[mask] = torch.stack((torch.full_like(lt, 255), 127 + 128 * lt, torch.zeros_like(lt)), dim=1)

        mask = (t > 0.35) & (t <= 0.50)
        lt = (t[mask] - 0.35) / 0.15
        colors[mask] = torch.stack((255 * (1 - lt), torch.full_like(lt, 255), torch.zeros_like(lt)), dim=1)

        mask = (t > 0.50) & (t <= 0.65)
        lt = (t[mask] - 0.50) / 0.15
        colors[mask] = torch.stack((torch.zeros_like(lt), torch.full_like(lt, 255), 255 * lt), dim=1)

        mask = (t > 0.65) & (t <= 0.75)
        lt = (t[mask] - 0.65) / 0.10
        colors[mask] = torch.stack((torch.zeros_like(lt), 255 * (1 - lt), torch.full_like(lt, 255)), dim=1)

        mask = (t > 0.75) & (t <= 0.85)
        lt = (t[mask] - 0.75) / 0.10
        colors[mask] = torch.stack((127 * lt, torch.zeros_like(lt), torch.full_like(lt, 255)), dim=1)

        mask = t > 0.85
        lt = (t[mask] - 0.85) / 0.15
        colors[mask] = torch.stack((127 * (1 - lt), torch.zeros_like(lt), 255 * (1 - lt)), dim=1)
        return colors

    @torch.no_grad()
    def colorize(self, depth_map):
        if depth_map.device != self.device: depth_map = depth_map.to(self.device)
        H, W = depth_map.shape
        depth_flat = depth_map.reshape(-1)
        z_clamped = torch.clamp(depth_flat, 0.01, 50000.0)

        indices = (((torch.log10(z_clamped) - self.log_min) / self.log_range) * (self.lut_size - 1)).long()
        indices = torch.clamp(indices, 0, self.lut_size - 1)
        return self.color_lut_rgb[indices].reshape(H, W, 3).to(torch.uint8)


class HighPerformanceLegendRenderer:
    def __init__(self, H, W, dtype=torch.float32, device='cuda'):
        self.H, self.W, self.dtype = H, W, dtype
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())

        self.legend_overlay = torch.zeros((H, W, 3), dtype=torch.uint8, device=self.device)
        self.legend_mask = torch.zeros((H, W, 1), dtype=dtype, device=self.device)
        self._create_legend_bgr()
        print(f"✓ Legend Renderer initialized on {self.device}")

    def _create_legend_bgr(self):
        dists = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 50000.0]
        ov_cpu = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        mk_cpu = np.zeros((self.H, self.W), dtype=np.float32)

        z = torch.tensor(dists, dtype=torch.float32)
        t = torch.clamp((torch.log10(torch.clamp(z, 0.01, 50000.0)) + 2.0) / 6.699, 0.0, 1.0)
        colors_bgr = []
        for val in t:
            if val <= 0.1:
                c = (255, int(255 * (1 - (val / 0.1))), int(255 * (1 - (val / 0.1))))
            elif val <= 0.25:
                c = (255, int(127 * ((val - 0.1) / 0.15)), 0)
            elif val <= 0.35:
                c = (255, int(127 + 128 * ((val - 0.25) / 0.1)), 0)
            elif val <= 0.50:
                c = (int(255 * (1 - ((val - 0.35) / 0.15))), 255, 0)
            elif val <= 0.65:
                c = (0, 255, int(255 * ((val - 0.50) / 0.15)))
            elif val <= 0.75:
                c = (0, int(255 * (1 - ((val - 0.65) / 0.1))), 255)
            elif val <= 0.85:
                c = (int(127 * ((val - 0.75) / 0.1)), 0, 255)
            else:
                c = (int(127 * (1 - ((val - 0.85) / 0.15))), 0, int(255 * (1 - ((val - 0.85) / 0.15))))
            colors_bgr.append((c[2], c[1], c[0]))

        lx, ly = self.W - 90, 10
        bw, bh = 25, 20
        cv2.rectangle(mk_cpu, (lx - 5, ly - 5), (lx + 175, ly + len(dists) * (bh + 5) + 10), 0.7, -1)

        for i, (d, c) in enumerate(zip(dists, colors_bgr)):
            y = ly + i * (bh + 5)
            if y + bh > self.H: break
            cv2.rectangle(ov_cpu, (lx, y), (lx + bw, y + bh), c, -1)
            cv2.rectangle(ov_cpu, (lx, y), (lx + bw, y + bh), (255, 255, 255), 1)
            mk_cpu[y:y + bh, lx:lx + bw] = 1.0
            txt = f"{d * 100:.0f}cm" if d < 1 else f"{d:.1f}m" if d < 1000 else f"{d / 1000:.1f}km"
            cv2.putText(ov_cpu, txt, (lx + 35, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            mk_cpu[y:y + bh, lx + 35:lx + 35 + len(txt) * 8] = 1.0

        self.legend_overlay = torch.from_numpy(ov_cpu).to(self.device)
        self.legend_mask = torch.from_numpy(mk_cpu).unsqueeze(2).to(self.device)

    @torch.no_grad()
    def blend(self, frame):
        if frame.device != self.device: frame = frame.to(self.device)
        return (frame.float() * (1 - self.legend_mask) + self.legend_overlay.float() * self.legend_mask).byte()


# ============================================================================
# SIMPLE CROSS-PROCESS SYNCHRONIZATION
# ============================================================================
class SimpleEvent:
    """File-based event for cross-process signaling (works on Windows/Unix)"""

    def __init__(self, name):
        self.name = name
        self.path = os.path.join(tempfile.gettempdir(), f"depth_event_{name}.flag")
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
class DepthEstimator:
    """
    Client-side API that communicates with the model subprocess.
    ZERO knowledge of ONNX/DML - complete import isolation.
    """

    def __init__(self, H, W, dtype=torch.float16, device='cuda', model_type='small', domain='indoor'):
        self.H = H
        self.W = W
        self.dtype = dtype
        self.device = device
        self.first_frame = True
        self.proc = None  # Initialize to avoid AttributeError in cleanup

        print("[API] Initializing Isolated Subprocess Depth Estimator...")

        # 1. Setup Shared Memory
        # Input: (1, 3, H, W)
        input_bytes = 1 * 3 * H * W * (2 if dtype == torch.float16 else 4)
        self.shm_in = shared_memory.SharedMemory(create=True, size=input_bytes)

        # Output: (H, W) Float32 (Logic returns float32)
        output_bytes = H * W * 4
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
        backend_script = os.path.join(os.path.dirname(__file__), 'depth_estimator_backend.py')
        if not os.path.exists(backend_script):
            backend_script = 'depth_estimator_backend.py'  # Fallback to relative

        # Build command line arguments
        args = [
            # We don't use sys.executable this because anaconda messes with onnx
            "python",  # Current Python interpreter
            backend_script,
            '--shm-in', self.shm_in.name,
            '--shm-out', self.shm_out.name,
            '--height', str(H),
            '--width', str(W),
            '--dtype', dtype_str,
            '--device', str(device),
            '--model-type', model_type,
            '--domain', domain,
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

    def estimate_depth(self, image_tensor):
        """
        Input: Tensor(1, 3, H, W)
        Returns: Tensor(H, W) - Full Resolution
        """
        # Wait for initialization on first use
        if not self._initialized:
            print("[API] Waiting for backend initialization (first use)...")
            if not self.evt_done.wait(timeout=260):
                raise RuntimeError("Backend subprocess failed to initialize")
            self.evt_done.clear()
            self._initialized = True
            print("[API] Backend ready.")

        # 1. Data Transfer (Host/CUDA -> Shared Memory)
        # Ensure contiguous and CPU
        if image_tensor.device.type != 'cpu':
            cpu_tensor = image_tensor.detach().cpu()
        else:
            cpu_tensor = image_tensor

        # Write to buffer
        np_in = np.ndarray((1, 3, self.H, self.W), dtype=np.float16 if self.dtype == torch.float16 else np.float32,
                           buffer=self.shm_in.buf)
        np_in[:] = cpu_tensor.numpy()[:]

        # 2. Signal Process
        self.evt_start.set()

        # 3. Wait for Result
        if not self.evt_done.wait(timeout=205):
            raise RuntimeError("Backend subprocess timed out")
        self.evt_done.clear()

        # 4. Read Result
        np_out = np.ndarray((self.H, self.W), dtype=np.float32, buffer=self.shm_out.buf)

        # Convert to Tensor
        res = torch.from_numpy(np_out.copy())

        # Return on requested device
        if 'cuda' in str(self.device):
            res = res.to(self.device)
        if self.first_frame:
            self.first_frame = False
            return self.estimate_depth(image_tensor)
        return res

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
    WIDTH = 1920//3
    HEIGHT = 1080//3

    # Hardware Selection: 'dml' (iGPU), 'cuda' (Nvidia), 'cpu'
    DEVICE_REQ = 'dml'

    # Precision: torch.float16 or torch.float32
    DTYPE = torch.float32

    # Model Selection (Only Base/Large supported for Metric Depth)
    MODEL_TYPE = 'small'  # 'base', 'large', 'small'
    DOMAIN_TYPE = 'indoor'  # 'indoor', 'outdoor'

    print(f"=== Hybrid Depth (Live) | Mode: {DEVICE_REQ} | Dtype: {DTYPE} ===")

    # 0. Global CUDA Init (Prevent DML collisions in Main process too)
    if torch.cuda.is_available():
        _ = torch.ones(1).cuda()

    # 1. Init Visualizers (Always prefer CUDA for Rendering)
    vis_dev = 'cuda'
    print(f"1. Init Visualizers ({vis_dev})...")

    colorizer = DepthMapColorizer(dtype=torch.float32, device=vis_dev, lut_size=4096)
    renderer = HighPerformanceLegendRenderer(HEIGHT, WIDTH, dtype=torch.float32, device=vis_dev)

    # 2. Init Estimator (This now launches the separate process)
    print(f"2. Init Estimator...")
    try:
        estimator = DepthEstimator(
            HEIGHT, WIDTH,
            dtype=DTYPE,
            device=DEVICE_REQ,
            model_type=MODEL_TYPE,
            domain=DOMAIN_TYPE
        )
    except Exception as e:
        print(f"Failed to init process: {e}")
        return

    cap = cv2.VideoCapture(r"C:\Users\cjlyn\Downloads\Phone Link\20251015_094321.mp4")
    # cap = cv2.VideoCapture("http://10.26.208.31:8080/video")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened(): return

    print("\n>>> RUNNING. Press 'q' to quit.")
    t_last = time.time()
    frames = 0

    try:
        while True:
            ret, frame = cap.read()
            frame = frame[::3, ::3]
            if not ret: break

            # --- STRICT INPUT HANDLING ---
            # Prepare tensor on CPU first for shared memory transfer
            t_in = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
            if DTYPE == torch.float16:
                t_in = t_in.half()
            else:
                t_in = t_in.float()

            rgb_in = t_in.flip(1)

            # --- INFERENCE ---
            t0 = time.perf_counter()
            # depth_map returned here is now full resolution (H x W)
            depth_map = estimator.estimate_depth(rgb_in)
            dt = time.perf_counter() - t0

            # --- POST-PROCESS & VISUALIZE ---
            # Ensure depth_vis is on visualization device
            depth_vis = depth_map.float().to(vis_dev, non_blocking=True)

            heatmap = colorizer.colorize(depth_vis)
            final = renderer.blend(heatmap)
            vis_np = final.cpu().numpy().astype(np.uint8)

            frames += 1
            if time.time() - t_last > 1.0:
                print(f"FPS: {frames} | Inference: {dt * 1000:.2f}ms")
                frames = 0
                t_last = time.time()

            cv2.imshow("Hybrid Depth", vis_np)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        estimator.cleanup()
        cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Support Windows Multiprocessing
    multiprocessing.freeze_support()

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    WIDTH = 1920
    HEIGHT = 1080

    # Hardware Selection: 'dml' (iGPU), 'cuda' (Nvidia), 'cpu'
    DEVICE_REQ = 'dml'

    # Precision: torch.float16 or torch.float32
    DTYPE = torch.float32

    # Model Selection (Only Base/Large supported for Metric Depth)
    MODEL_TYPE = 'large'  # 'base', 'large', 'small'
    DOMAIN_TYPE = 'indoor'  # 'indoor', 'outdoor'

    # --- VIDEO SAVING CONFIGURATION ---
    OUTPUT_FILENAME = "depth_visualization_output.mp4"
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    FPS = 30.0  # Target FPS for the output video
    # ----------------------------------

    print(f"=== Hybrid Depth (Live) | Mode: {DEVICE_REQ} | Dtype: {DTYPE} ===")

    # 0. Global CUDA Init (Prevent DML collisions in Main process too)
    if torch.cuda.is_available():
        _ = torch.ones(1).cuda()

    # 1. Init Visualizers (Always prefer CUDA for Rendering)
    vis_dev = 'cuda'
    print(f"1. Init Visualizers ({vis_dev})...")

    colorizer = DepthMapColorizer(dtype=torch.float32, device=vis_dev, lut_size=4096)
    renderer = HighPerformanceLegendRenderer(HEIGHT, WIDTH, dtype=torch.float32, device=vis_dev)

    # 2. Init Estimator (This now launches the separate process)
    print(f"2. Init Estimator...")
    try:
        estimator = DepthEstimator(
            HEIGHT, WIDTH,
            dtype=DTYPE,
            device=DEVICE_REQ,
            model_type=MODEL_TYPE,
            domain=DOMAIN_TYPE
        )
    except Exception as e:
        print(f"Failed to init process: {e}")
        return

    cap = cv2.VideoCapture(r"C:\Users\cjlyn\Downloads\Phone Link\20251015_094321.mp4")
    # cap = cv2.VideoCapture("http://10.26.208.31:8080/video")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened(): return

    # 3. Init Video Writer
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, FOURCC, FPS, (WIDTH, HEIGHT))
    if not video_writer.isOpened():
        print(f"!!! WARNING: Could not open VideoWriter for {OUTPUT_FILENAME}")
        return

    print("\n>>> RUNNING. Press 'q' to quit.")
    t_last = time.time()
    frames = 0

    try:
        while True:
            ret, frame = cap.read()
            frame = frame#[::3, ::3]
            if not ret: break

            # --- STRICT INPUT HANDLING ---
            # Prepare tensor on CPU first for shared memory transfer
            t_in = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
            if DTYPE == torch.float16:
                t_in = t_in.half()
            else:
                t_in = t_in.float()

            rgb_in = t_in.flip(1)

            # --- INFERENCE ---
            t0 = time.perf_counter()
            # depth_map returned here is now full resolution (H x W)
            depth_map = estimator.estimate_depth(rgb_in)
            dt = time.perf_counter() - t0

            # --- POST-PROCESS & VISUALIZE ---
            # Ensure depth_vis is on visualization device
            depth_vis = depth_map.float().to(vis_dev, non_blocking=True)

            heatmap = colorizer.colorize(depth_vis)
            final = renderer.blend(heatmap)
            vis_np = final.cpu().numpy().astype(np.uint8)

            # --- VIDEO WRITING ---
            video_writer.write(vis_np)  # Write the frame to the video file

            frames += 1
            if time.time() - t_last > 1.0:
                print(f"FPS: {frames} | Inference: {dt * 1000:.2f}ms")
                frames = 0
                t_last = time.time()

            cv2.imshow("Hybrid Depth", vis_np)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        print("\nStopping and cleaning up...")
        estimator.cleanup()
        cap.release()
        video_writer.release()  # Release the video writer
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()