#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAFT Tracker Backend - Isolated Subprocess
This script runs in a completely separate Python process.
ONNX imports here will NEVER contaminate the calling code.
"""

# Force UTF-8 output encoding on Windows
import sys

if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# CRITICAL: onnxruntime must be imported BEFORE torch to avoid DLL conflicts on Windows
try:
    import onnxruntime as ort
except ImportError:
    ort = None
    print("Warning: onnxruntime not found. DML (iGPU) mode will not work.")

import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torchvision.models.optical_flow as of
from multiprocessing import shared_memory
import argparse

# ============================================================================
# HYBRID RAFT TRACKER
# ============================================================================
class RaftTracker:
    """
    Hybrid RAFT Tracker with Integrated Sampling. Handles CUDA and DML backends.
    """
    MODEL_DIR = 'models'

    def __init__(self, H, W, num_landmarks, dtype=torch.float16, device='cuda', model_type='small'):
        self.device = device
        self.device_str = str(device).lower()
        self.dtype = dtype
        self.H = H
        self.W = W

        # 1. Logic & Config
        if model_type.lower() == 'small':
            self.weights = of.Raft_Small_Weights.DEFAULT
            self.builder = of.raft_small
            self.cache_base = "raft_small"
        else:
            self.weights = of.Raft_Large_Weights.DEFAULT
            self.builder = of.raft_large
            self.cache_base = "raft_large"

        # 2. Execution Device Setup
        if self.device_str in ['igpu', 'dml', 'directml'] and ort is not None:
            self.mode = 'dml'
            self.exec_device = torch.device('cpu')
            print(f"[RAFT] Mode: DML (iGPU/ONNX). Logic execution on CPU.")
        elif torch.cuda.is_available() and 'cuda' in self.device_str:
            self.mode = 'cuda'
            self.exec_device = torch.device('cuda', torch.cuda.current_device())
            print(f"[RAFT] Mode: Standard CUDA ({self.exec_device}).")
        else:
            self.mode = 'cpu'
            self.exec_device = torch.device('cpu')
            print(f"[RAFT] Mode: Standard CPU.")

        # 3. Caching Config
        self.cache_base = f"{self.cache_base}_{H}x{W}_{str(self.dtype).split('.')[-1]}_w"
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # 4. Normalization Constants
        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=self.dtype, device=self.exec_device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=self.dtype, device=self.exec_device).view(1, 3, 1, 1)

        # 5. Buffer Allocations
        self.img1 = torch.empty((1, 3, H, W), dtype=self.dtype, device=self.exec_device)
        self.img2 = torch.empty((1, 3, H, W), dtype=self.dtype, device=self.exec_device)

        # Bounds logic
        self.bound_w = torch.tensor(W - W // 10, dtype=torch.float32, device=self.exec_device)
        self.bound_h = torch.tensor(H - H // 10, dtype=torch.float32, device=self.exec_device)
        self.min_pad_w = torch.tensor(W // 10, dtype=torch.float32, device=self.exec_device)
        self.min_pad_h = torch.tensor(H // 10, dtype=torch.float32, device=self.exec_device)

        # Pre-allocate masks
        self.valid_mask = torch.empty(num_landmarks, dtype=torch.bool, device=self.exec_device)
        self.temp_bool = torch.empty(num_landmarks, dtype=torch.bool, device=self.exec_device)
        self.zero_points = torch.empty((0, 2), dtype=torch.float32, device=self.exec_device)
        self.zero_mask = torch.empty(0, dtype=torch.bool, device=self.exec_device)

        # 6. Init Backend
        if self.mode == 'dml':
            self._init_dml()
        else:
            self._init_cuda()

        print(f"OK RAFT Tracker Ready: {H}x{W}")

    def _init_dml(self):
        onnx_file = f"{self.cache_base}_sampled_opset16.onnx"
        onnx_path = os.path.join(self.MODEL_DIR, onnx_file)

        if not os.path.exists(onnx_path):
            print(f"[RAFT] Tracing & Exporting ONNX...")
            self._export_onnx_with_sampling(onnx_path)

        print(f"[RAFT] Loading DirectML Session...")
        opts = ort.SessionOptions()

        # 1. ENABLE OPTIMIZATIONS
        opts.enable_mem_pattern = True
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 2. Set provider options
        providers = [('DmlExecutionProvider', {
            'device_id': 0,
            # 'skip_onnx_opt': False # Ensure DML specific opts are on
        })]

        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)

        # 3. Setup IO Binding (Pre-calculating shapes helps)
        self.binding = self.session.io_binding()
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.output_names = [node.name for node in self.session.get_outputs()]

        print(f"[RAFT] DirectML Session Ready")

    def _export_onnx_with_sampling(self, path):
        class RaftWithSampling(torch.nn.Module):
            def __init__(self, builder, weights):
                super().__init__()
                self.model = builder(weights=weights)

            def forward(self, img1, img2, norm_grid):
                list_of_flows = self.model(img1, img2)
                flow = list_of_flows[-1]
                sampled_delta = F.grid_sample(
                    flow.to(dtype=torch.float32), norm_grid, mode='bilinear', padding_mode='border', align_corners=True
                )
                return sampled_delta.squeeze(2).permute(0, 2, 1)

        model = RaftWithSampling(self.builder, self.weights).eval()
        model = model.to('cuda')
        if self.dtype == torch.float16:
            model = model.half()

        H, W = self.H, self.W
        d_img = torch.randn(1, 3, H, W, dtype=self.dtype, device='cuda')
        d_grid = torch.randn(1, 1, 128, 2, dtype=torch.float32, device='cuda')

        print(f"[RAFT] Exporting...")
        torch.onnx.export(
            model, (d_img, d_img, d_grid), path,
            input_names=["img1", "img2", "norm_grid"],
            output_names=["sampled_delta"],
            opset_version=16,
            do_constant_folding=True,
            dynamic_axes={'norm_grid': {2: 'num_points'}, 'sampled_delta': {1: 'num_points'}}
        )
        print(f"[RAFT] Export complete.")

    def _init_cuda(self):
        jit_file = f"{self.cache_base}_sampled_jit.pt"
        jit_path = os.path.join(self.MODEL_DIR, jit_file)

        if os.path.exists(jit_path):
            print(f"[RAFT] Loading JIT from cache...")
            self.traced_model = torch.jit.load(jit_path, map_location=self.exec_device).eval()
            self.traced_model = torch.jit.freeze(self.traced_model)
        else:
            print(f"[RAFT] Tracing CUDA JIT with Sampling...")

            class RaftWithSampling(torch.nn.Module):
                def __init__(self, builder, weights):
                    super().__init__()
                    self.model = builder(weights=weights)

                def forward(self, img1, img2, norm_grid):
                    flow = self.model(img1, img2)[-1]
                    sampled = F.grid_sample(flow.to(dtype=torch.float32), norm_grid, mode='bilinear',
                                            padding_mode='border', align_corners=True)
                    return sampled.squeeze(2).permute(0, 2, 1)

            model = RaftWithSampling(self.builder, self.weights).to(self.exec_device).eval()
            if self.dtype == torch.float16:
                model = model.half()

            H, W = self.H, self.W
            d_img = torch.randn(1, 3, H, W, device=self.exec_device, dtype=self.dtype)
            d_grid = torch.randn(1, 1, 128, 2, device=self.exec_device, dtype=torch.float32)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=self.dtype):
                self.traced_model = torch.jit.trace(model, (d_img, d_img, d_grid), check_trace=False)
                self.traced_model.eval()
                self.traced_model = torch.jit.freeze(self.traced_model)

            torch.jit.save(self.traced_model, jit_path)
            print(f"[RAFT] Saved JIT to {jit_path}")

    @torch.inference_mode()
    def track_points(self, img1, img2, points):
        """
        img1, img2: (3, H, W) RGB/BGR tensor
        points: (N, 2) coordinates [x, y]
        """
        # 1. Coordinate Normalization
        norm_grid = points.unsqueeze(0).unsqueeze(1).clone()
        if points.shape[0] == 0:
            return self.zero_points, self.zero_mask, None, None

        bw = self.bound_w.to(self.dtype)
        bh = self.bound_h.to(self.dtype)

        norm_grid[..., 0].mul_(2.0).div_(bw - 1).sub_(1.0)
        norm_grid[..., 1].mul_(2.0).div_(bh - 1).sub_(1.0)

        # 2. Image Prep
        i1_norm = self.img1.copy_(img1[None]).div_(255.0).sub_(self.mean).div_(self.std)
        i2_norm = self.img2.copy_(img2[None]).div_(255.0).sub_(self.mean).div_(self.std)

        # 3. Inference
        if self.mode == 'dml':
            inputs = {
                self.input_names[0]: i1_norm.cpu().numpy(),
                self.input_names[1]: i2_norm.cpu().numpy(),
                self.input_names[2]: norm_grid.cpu().numpy()
            }
            res = self.session.run(None, inputs)
            delta = torch.from_numpy(res[0]).to(self.exec_device)
        else:
            ctx = torch.amp.autocast('cuda', dtype=self.dtype) if self.dtype == torch.float16 else torch.no_grad()
            with ctx:
                delta = self.traced_model(i1_norm, i2_norm, norm_grid)

        # 4. Update
        new_points = points + delta[0]

        # 5. Masking
        x = new_points[:, 0]
        y = new_points[:, 1]
        N = x.shape[0]
        mask = self.valid_mask[:N]

        torch.ge(x, self.min_pad_w, out=mask)
        torch.lt(x, self.bound_w, out=self.temp_bool[:N])
        mask.bitwise_and_(self.temp_bool[:N])

        torch.ge(y, self.min_pad_h, out=self.temp_bool[:N])
        mask.bitwise_and_(self.temp_bool[:N])

        torch.lt(y, self.bound_h, out=self.temp_bool[:N])
        mask.bitwise_and_(self.temp_bool[:N])

        return new_points, mask, None, None


# ============================================================================
# SUBPROCESS MAIN LOOP
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='RAFT Tracker Backend Subprocess')
    parser.add_argument('--shm-in', required=True, help='Input shared memory name')
    parser.add_argument('--shm-out', required=True, help='Output shared memory name')
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--num-points', type=int, required=True)
    parser.add_argument('--dtype', required=True, choices=['float16', 'float32'])
    parser.add_argument('--device', required=True)
    parser.add_argument('--model-type', required=True, choices=['small', 'large'])

    # Event names (cross-platform file-based)
    parser.add_argument('--evt-start', required=True)
    parser.add_argument('--evt-done', required=True)
    parser.add_argument('--evt-stop', required=True)

    args = parser.parse_args()

    # Reconstruct dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    d_size = 2 if dtype == torch.float16 else 4

    H, W, N = args.height, args.width, args.num_points

    # Shared memory layout
    img_size = 3 * H * W * d_size
    pts_size = N * 4 * d_size
    # Attach to shared memory
    shm_in = shared_memory.SharedMemory(name=args.shm_in)
    shm_out = shared_memory.SharedMemory(name=args.shm_out)

    # Reconstruct file-based events
    class SimpleEvent:
        """File-based event for cross-process signaling"""

        def __init__(self, name):
            self.name = name
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.',
                                     f"raft_event_{name}.flag")
            # Use temp directory on Windows
            if sys.platform == 'win32':
                import tempfile
                self.path = os.path.join(tempfile.gettempdir(), f"raft_event_{name}.flag")

        def set(self):
            with open(self.path, 'wb') as f:
                f.write(b'\x01')

        def clear(self):
            try:
                with open(self.path, 'wb') as f:
                    f.write(b'\x00')
            except:
                pass

        def wait(self, timeout=None):
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
                time.sleep(0.001)

        def is_set(self):
            try:
                with open(self.path, 'rb') as f:
                    return f.read(1) == b'\x01'
            except:
                return False

    evt_start = SimpleEvent(args.evt_start)
    evt_done = SimpleEvent(args.evt_done)
    evt_stop = SimpleEvent(args.evt_stop)

    # Initialize tracker
    print("[Backend] Initializing tracker...")
    sys.stdout.flush()  # Force flush before slow operation
    tracker = RaftTracker(H, W, N, dtype=dtype, device=args.device, model_type=args.model_type)

    print("[Backend] Initialization complete. Ready for frames.")
    sys.stdout.flush()  # Force flush
    evt_done.set()  # Signal init complete

    try:
        while True:
            # Wait for start signal
            evt_start.wait()
            if evt_stop.is_set():
                break
            evt_start.clear()

            # Read inputs from shared memory
            np_img1 = np.ndarray((3, H, W), dtype=np.float16 if dtype == torch.float16 else np.float32,
                                 buffer=shm_in.buf, offset=0)
            t_img1 = torch.from_numpy(np_img1).to(tracker.exec_device)

            np_img2 = np.ndarray((3, H, W), dtype=np.float16 if dtype == torch.float16 else np.float32,
                                 buffer=shm_in.buf, offset=img_size)
            t_img2 = torch.from_numpy(np_img2).to(tracker.exec_device)

            np_pts = np.ndarray((N, 2), dtype=np.float32,
                                buffer=shm_in.buf, offset=img_size + img_size)
            t_pts = torch.from_numpy(np_pts).to(tracker.exec_device)

            # Run tracking
            new_pts, mask, _, _ = tracker.track_points(t_img1, t_img2, t_pts)

            # Write outputs
            res_pts_np = new_pts.cpu().numpy()
            np_out_pts = np.ndarray((N, 2), dtype=np.float32, buffer=shm_out.buf, offset=0)
            np_out_pts[:] = res_pts_np[:]

            res_mask_np = mask.cpu().numpy().astype(np.bool_)
            np_out_mask = np.ndarray((N,), dtype=np.bool_, buffer=shm_out.buf, offset=pts_size)
            np_out_mask[:] = res_mask_np[:]

            # Signal done
            evt_done.set()

    except KeyboardInterrupt:
        print("[Backend] Interrupted")
    except Exception as e:
        print(f"[Backend] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Backend] Shutting down.")
        shm_in.close()
        shm_out.close()


if __name__ == "__main__":
    main()