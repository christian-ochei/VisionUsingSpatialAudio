#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Estimator Backend - Isolated Subprocess
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

import torch
import torch.nn.functional as F
import numpy as np
import os
from multiprocessing import shared_memory
from transformers import AutoModelForDepthEstimation
import argparse


# ============================================================================
# HYBRID DEPTH ESTIMATOR (LOGIC SIDE)
# ============================================================================
class DepthEstimatorLogic:
    """
    The actual inference logic running inside the separate process.
    """
    MODEL_DIR = 'models'

    def __init__(self, H, W, dtype=torch.float16, device='cuda', model_type='base', domain='indoor'):
        self.device_str = str(device).lower()
        self.dtype = dtype

        # Dimensions
        self.H = H
        self.W = W

        # 1. Logic & Config
        m_type = model_type.title()
        dom = domain.title()

        if m_type not in ['Base', 'Large', 'Small']:
            raise Exception(f"[Process] Error: '{m_type}' not explicit. Defaulting to Base.")

        self.model_id = f"depth-anything/Depth-Anything-V2-Metric-{dom}-{m_type}-hf"
        self.cache_base = f"depth_v2_metric_{dom.lower()}_{m_type.lower()}"

        # 2. Execution Device Setup
        if self.device_str in ['igpu', 'dml', 'directml']:
            self.mode = 'dml'
            self.exec_device = torch.device('cpu')
            if ort is None: raise ImportError("onnxruntime-directml required for iGPU/DML mode")
            print(f"[Process] Mode: DML (iGPU). Logic execution on CPU.")
        elif self.device_str == 'cpu':
            self.mode = 'cpu'
            self.exec_device = torch.device('cpu')
            print(f"[Process] Mode: Standard CPU.")
        else:
            self.mode = 'cuda'
            if torch.cuda.is_available():
                self.exec_device = torch.device('cuda', torch.cuda.current_device())
            else:
                self.exec_device = torch.device('cuda')
            print(f"[Process] Mode: Standard CUDA ({self.exec_device}).")

        # 3. Resolution & Caching
        self.target_size = 364
        patch_size = 14
        scale = self.target_size / W if W > H else self.target_size / H
        self.new_h = int(round((H * scale) / patch_size) * patch_size)
        self.new_w = int(round((W * scale) / patch_size) * patch_size)
        self.new_h = max(self.new_h, patch_size)
        self.new_w = max(self.new_w, patch_size)

        dtype_str = str(self.dtype).split('.')[-1]
        self.cache_base = f"{self.cache_base}_{self.new_h}x{self.new_w}_{dtype_str}_raw_out"
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # 4. Buffer Allocation
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=self.dtype, device=self.exec_device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=self.dtype, device=self.exec_device).view(1, 3, 1, 1)

        # 5. Init Backend
        if self.mode == 'dml':
            self._init_dml()
        else:
            self._init_cuda()
        print(f"OK [Process] Estimator Ready: {self.new_h}x{self.new_w} -> {self.H}x{self.W}")

    def _init_dml(self):
        onnx_file = f"{self.cache_base}_opset16.onnx"
        onnx_path = os.path.join(self.MODEL_DIR, onnx_file)

        if not os.path.exists(onnx_path):
            print(f"[Process] Tracing & Exporting ONNX...")
            self._export_onnx(onnx_path)

        print(f"[Process] Loading DirectML Session...")
        sys.stdout.flush()
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        providers = [('DmlExecutionProvider', {'device_id': 0})]
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[Process] DirectML Session Ready")
        sys.stdout.flush()

    def _export_onnx(self, path):
        hf_model = AutoModelForDepthEstimation.from_pretrained(self.model_id).eval()
        if self.dtype == torch.float16:
            hf_model = hf_model.half()

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x, return_dict=False)[0]

        wrapper = Wrapper(hf_model)
        wrapper = wrapper.to('cuda')
        dummy = torch.randn(1, 3, self.new_h, self.new_w, dtype=self.dtype, device='cuda')
        torch.onnx.export(wrapper, dummy, path, input_names=["pixel_values"], output_names=["depth_map"],
                          opset_version=16, do_constant_folding=True)

    def _init_cuda(self):
        jit_file = f"{self.cache_base}_jit.pt"
        jit_path = os.path.join(self.MODEL_DIR, jit_file)

        if os.path.exists(jit_path):
            print(f"[Process] Loading JIT from cache...")
            self.traced_model = torch.jit.load(jit_path, map_location=self.exec_device)
            self.traced_model.eval()
            self.traced_model = torch.jit.freeze(self.traced_model)
        else:
            print(f"[Process] Tracing CUDA JIT...")
            hf_model = AutoModelForDepthEstimation.from_pretrained(self.model_id, torch_dtype=self.dtype).to(
                self.exec_device).eval()

            class Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(pixel_values=x, return_dict=False)[0]

            wrapper = Wrapper(hf_model)
            dummy = torch.randn(1, 3, self.new_h, self.new_w, device=self.exec_device, dtype=self.dtype)
            with torch.no_grad():
                self.traced_model = torch.jit.trace(wrapper, dummy, check_trace=False)
                self.traced_model.eval()
                self.traced_model = torch.jit.freeze(self.traced_model)
            torch.jit.save(self.traced_model, jit_path)

    @torch.inference_mode()
    def process_frame(self, image_tensor):
        """
        Input: Tensor(1, 3, H, W)
        Returns: Tensor(H, W) - Full Resolution
        """
        # Ensure input is on correct device for inference
        if image_tensor.device != self.exec_device:
            image_tensor = image_tensor.to(self.exec_device)

        # 1. Resize Input
        x = F.interpolate(image_tensor, size=(self.new_h, self.new_w), mode='bicubic', align_corners=False)
        x.div_(255.0).sub_(self.mean).div_(self.std)

        # 2. Inference
        if self.mode == 'dml':
            inputs = {self.input_name: x.cpu().numpy()}
            res_raw = self.session.run(None, inputs)[0]
            res = torch.from_numpy(res_raw)
        else:
            res = self.traced_model(x)

        # 3. UP-SCALING
        if res.dim() == 2:
            res = res.unsqueeze(0).unsqueeze(0)
        elif res.dim() == 3:
            res = res.unsqueeze(1)

        # Full resolution upscale
        full_res = F.interpolate(res.float(), size=(self.H, self.W), mode="bicubic", align_corners=False)
        return full_res.squeeze()


# ============================================================================
# SUBPROCESS MAIN LOOP
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Depth Estimator Backend Subprocess')
    parser.add_argument('--shm-in', required=True, help='Input shared memory name')
    parser.add_argument('--shm-out', required=True, help='Output shared memory name')
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--dtype', required=True, choices=['float16', 'float32'])
    parser.add_argument('--device', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--domain', required=True)

    # Event names (cross-platform file-based)
    parser.add_argument('--evt-start', required=True)
    parser.add_argument('--evt-done', required=True)
    parser.add_argument('--evt-stop', required=True)

    args = parser.parse_args()

    # Reconstruct dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32

    H, W = args.height, args.width

    # Shared memory layout
    # Attach to shared memory
    shm_in = shared_memory.SharedMemory(name=args.shm_in)
    shm_out = shared_memory.SharedMemory(name=args.shm_out)

    # Reconstruct file-based events
    class SimpleEvent:
        """File-based event for cross-process signaling"""

        def __init__(self, name):
            self.name = name
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.',
                                     f"depth_event_{name}.flag")
            # Use temp directory on Windows
            if sys.platform == 'win32':
                import tempfile
                self.path = os.path.join(tempfile.gettempdir(), f"depth_event_{name}.flag")

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

    # Initialize estimator
    print("[Backend] Initializing estimator...")
    sys.stdout.flush()
    estimator = DepthEstimatorLogic(H, W, dtype=dtype, device=args.device,
                                    model_type=args.model_type, domain=args.domain)

    print("[Backend] Initialization complete. Ready for frames.")
    sys.stdout.flush()
    evt_done.set()  # Signal init complete

    try:
        while True:
            # Wait for start signal
            evt_start.wait()
            if evt_stop.is_set():
                break
            evt_start.clear()

            # Read from Shared Memory
            np_in = np.ndarray((1, 3, H, W), dtype=np.float16 if dtype == torch.float16 else np.float32,
                               buffer=shm_in.buf)

            # Convert to Tensor
            t_in = torch.from_numpy(np_in)

            # RUN PIPELINE
            result_tensor = estimator.process_frame(t_in)

            # Write Result to Shared Memory
            result_np = result_tensor.cpu().numpy().astype(np.float32)

            # Create numpy view of output buffer
            np_out = np.ndarray((H, W), dtype=np.float32, buffer=shm_out.buf)
            np_out[:] = result_np[:]

            # Signal Done
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