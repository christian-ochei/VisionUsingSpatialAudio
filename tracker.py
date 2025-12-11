
from raft_check import RaftTracker
from depth_client import MultiThreadedZSampler, DepthMapColorizer
# These 2 imports must stay at the very top
import random
import string
import threading
from typing import Tuple, Union
import cv2
from collections import deque
from filters import LowPassFilter
from process_depthmap import DepthFocus, HighPerformanceLegendRenderer, stable_checker, CameraProjection
from imu_sense import IMUSensing
import torch
from realtime_frame_reader import SingleShotCamera
from vsa_visualizer import VSAVisualizer
import torch.nn.functional as functional
import numpy as np
import time
from contextlib import nullcontext
from line_profiler_pycharm import profile

GREEN = '\033[92m'
ENDC = '\033[0m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BLUE = '\033[94m'

def makeid(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


# Placeholder for torch.jit.load, which can be an async call in some frameworks
def _jit_load_sync(path, map_location=None):
    """Synchronous JIT load wrapper."""
    return torch.jit.load(path, map_location=map_location)

class BKTrackingPipeline:
    """Keyframe-based point tracking with proper fresh point handling."""

    def __init__(
        self, H, W, num_landmarks=2000, dtype=torch.float32, device='cuda', model_execution_lock=None,
        is_portrait_mode=False, verbose=True
    ):
        self.H, self.W = H, W
        self.verbose = verbose
        self.num_landmarks = num_landmarks
        self.dtype = dtype
        self.device = device
        if model_execution_lock is None:
            model_execution_lock = nullcontext()
        self.model_execution_lock = model_execution_lock
        self.EDGE_MARGIN = 40
        self.KEYFRAME_INTERVAL = 10
        self.is_portrait_mode = is_portrait_mode
        if is_portrait_mode:
            h, w = W, H
        else:
            h, w = H, W
        self.raft_tracker = RaftTracker(
            # We use float32 Because model for some reason, flops on float32
            device='dml', dtype=torch.float32, H=h, W=w, num_landmarks=num_landmarks
        )
        self.current_flows = self.generate_grid_points()
        self.current_masks = torch.ones(num_landmarks, dtype=torch.bool, device=device)
        self.keyframe_tensor = None
        self.keyframe_tensor_cp = torch.empty((H, W, 3), dtype=self.raft_tracker.dtype, device=self.raft_tracker.exec_device)
        self.frame_tensor_cp = torch.empty((H, W, 3), dtype=self.raft_tracker.dtype, device=self.raft_tracker.exec_device)
        self.keyframe_flows = self.current_flows.clone()
        self.fresh_integration_mask_cp = torch.zeros(num_landmarks, dtype=torch.bool, device=device)
        self.fresh_integration_mask = torch.zeros(num_landmarks, dtype=torch.bool, device=device)
        self.arange_buf = torch.arange(num_landmarks + num_landmarks*5 + 10, device=self.device)
        self.pending_integration_command = None
        self.candidate_lock = threading.Lock()
        self.frame_count = 0
        print(f"🎯 Tracker: {num_landmarks} points @ {W}x{H} (Keyframe-based, Fixed)\n")

        # Debug tools ===========================================================================
        self.alive_mask_ones_debug = torch.ones(num_landmarks, dtype=torch.bool, device=device)
        self.alive_mask_zeros_debug = torch.zeros(num_landmarks, dtype=torch.bool, device=device)
        self.landmarks_debug = torch.empty((num_landmarks, 2), dtype=dtype, device=device)
        self.landmarks_debug[..., 0] = W / 2
        self.landmarks_debug[..., 1] = H / 2

    def generate_grid_points(self):
        """Generate grid in [X, Y] format."""
        aspect = self.W / self.H
        if self.num_landmarks == 2:
            return torch.tensor([[self.W / 4, self.H / 2],
                                 [3 * self.W / 4, self.H / 2]], dtype=self.dtype, device=self.device)
        elif self.num_landmarks == 1:
            return torch.tensor([[self.W / 2, self.H / 2]], dtype=self.dtype, device=self.device)
        elif self.num_landmarks == 0:
            return torch.empty((0, 2), dtype=self.dtype, device=self.device)
        ny = int(np.sqrt(self.num_landmarks / aspect))
        nx = int(np.ceil(self.num_landmarks / ny))
        y = np.linspace(30, self.H - 30, ny)
        x = np.linspace(30, self.W - 30, nx)
        xv, yv = np.meshgrid(x, y)
        points = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)[:self.num_landmarks]
        if len(points) < self.num_landmarks:
            points = np.vstack([points, np.tile(points[-1:], (self.num_landmarks - len(points), 1))])
        return torch.from_numpy(points).to(self.device)

    def get_and_clear_fresh_mask(self):
        """Get and clear fresh mask atomically."""
        fresh = self.fresh_integration_mask_cp.copy_(self.fresh_integration_mask)
        self.fresh_integration_mask.zero_()
        return fresh

    def set_candidate_points(self, points, z_values, amp_values_norm, frame_tensor):
        """Store candidates WITH their pre-sampled z-values."""
        with self.candidate_lock:
            self.pending_integration_command = {
                "candidates_S": points,
                "z_values_S": z_values,
                "amp_values_norm_S": amp_values_norm,
                "keyframe_S": self.frame_tensor_cp.copy_(frame_tensor),
            }

    def _execute_integration_command(self, current_frame_tensor):
        """Execute integration and return z-values for fresh points."""
        command = None
        with self.candidate_lock:
            if self.pending_integration_command is not None:
                command = self.pending_integration_command
                self.pending_integration_command = None

        if command is None:
            return None

        candidates_S = command["candidates_S"]
        z_values_S = command["z_values_S"]
        amp_values_norm_S = command["amp_values_norm_S"]
        keyframe_S = command["keyframe_S"]
        is_dead_mask = ~self.current_masks
        # num_dead = is_dead_mask.sum()
        num_candidates = len(candidates_S)
        # Non-blocking replacement
        # rand_indices = torch.randperm(num_candidates, device=self.device)
        # Create a boolean mask of size 'num_candidates' where the first 'num_dead' are True
        # We use arange to compare against num_dead purely on GPU
        # mask_indices = self.arange_buf[:num_candidates] < num_dead
        # Now filter. Note: This creates a dynamic shape tensor, see point 2.
        # indices = rand_indices[mask_indices]
        indices = self.arange_buf[:num_candidates]
        selected_S = candidates_S[indices]
        selected_zs_S = z_values_S[indices]
        selected_amps_norm_S = amp_values_norm_S[indices]
        self.verb_print(f"   ... running S→T mini-track for points")

        # For some reason, portrait fails so we simply permute to landscape temporarily
        if self.is_portrait_mode:
            selected_S = selected_S[:, [1, 0]]
            keyframe_S = keyframe_S.permute(0, 2, 1)
            current_frame_tensor = current_frame_tensor.permute(0, 2, 1)
        selected_T, valid_T, _, _ = self.raft_tracker.track_points(
            keyframe_S,
            current_frame_tensor,
            selected_S.to(dtype=torch.float32)
        )
        if self.is_portrait_mode:
            selected_T = selected_T[:, [1, 0]]

        dead_indices = torch.where(is_dead_mask)[0]
        valid_indices_into_dead = torch.where(valid_T)[0]
        valid_new_indices = dead_indices[valid_indices_into_dead]
        valid_new_points_T = selected_T[valid_T]
        valid_zs_S = selected_zs_S[valid_T]
        valid_amps_norm_S = selected_amps_norm_S[valid_T]
        final_integration_mask = valid_new_indices
        # final_integration_mask = torch.zeros(self.num_landmarks, dtype=torch.bool, device=self.device)
        # final_integration_mask[valid_new_indices] = True
        self.current_flows[final_integration_mask] = valid_new_points_T
        self.current_masks[final_integration_mask] = True
        self.fresh_integration_mask[final_integration_mask] = True
        self.verb_print(f"   ✓ Synced new points to present (T)")
        return {
            'integrated': True,
            'fresh_indices': valid_new_indices,
            'fresh_z_values': valid_zs_S,
            'normalized_amp_values': valid_amps_norm_S
        }

    def update(self, frame_tensor):
        """Main tracking update with z-value passthrough."""
        # 1. HANDLING INITIALIZATION OR REBASE
        # We rebase if it's the first frame, OR if we hit the interval
        # You might also want to OR this with: `or self.fresh_integration_occurred`
        if self.keyframe_tensor is None or (self.frame_count % self.KEYFRAME_INTERVAL == 0):
            if self.keyframe_tensor is None:
                self.keyframe_tensor = self.keyframe_tensor_cp.copy_(frame_tensor)
            else:
                self.keyframe_tensor.copy_(frame_tensor)
            self.keyframe_flows[:] = self.current_flows

        # 2. TRACKING (Keyframe -> Current)
        if self.is_portrait_mode:
            keyframe_tensor_ = self.keyframe_tensor.permute(1, 0, 2)
            frame_tensor_ = frame_tensor.permute(1, 0, 2)
            keyframe_flows_ = self.keyframe_flows[:, [1, 0]]
        else:
            keyframe_tensor_ = self.keyframe_tensor
            frame_tensor_ = frame_tensor
            keyframe_flows_ = self.keyframe_flows

        new_points, valid, flow_field, sampled_flow = self.raft_tracker.track_points(
            # keyframe_flows must be float32 for accurate accumulative precision
            keyframe_tensor_[None].permute(0, 3, 1, 2).clone(), frame_tensor_[None].permute(0, 3, 1, 2).clone(), keyframe_flows_
        )
        if self.is_portrait_mode:
            new_points = new_points[:, [1, 0]]
        self.current_masks[~valid] = False
        self.current_flows[self.current_masks] = new_points[self.current_masks]


        # 4. INTEGRATION
        fresh_mask = self.get_and_clear_fresh_mask()
        integration_result = self._execute_integration_command(frame_tensor)

        # OPTIONAL PRO TIP:
        # If integration happened, the new points exist in 'frame_tensor' but NOT in 'self.keyframe_tensor'.
        # You must force a rebase on the NEXT frame, or immediately update the keyframe now.
        # Otherwise, the next loop will try to track these new points from the OLD keyframe.
        if integration_result and integration_result['integrated']:
            # Force keyframe update immediately so new points stick
            self.keyframe_tensor.copy_(frame_tensor)
            self.keyframe_flows[:] = self.current_flows
            self.verb_print("   🔄 Forced Rebase due to Integration")

        self.frame_count += 1
        return self.current_flows, self.current_masks, fresh_mask, integration_result

    def verb_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

# ==================================================================================
# FIXED TRACKING PIPELINE (INTEGRATED)
# ==================================================================================
class TrackingPipeline:
    """
    Keyframe-based point tracking (Frame-to-Frame).
    Fully integrated with Z-Sampler candidate logic.
    """

    def __init__(
            self, H, W, num_landmarks=256, dtype=torch.float32, device='cpu', model_execution_lock=None,
             is_portrait_mode=False, verbose=True):
        self.H, self.W = H, W
        self.verbose = verbose
        self.num_landmarks = num_landmarks
        self.dtype = dtype
        self.device = device
        self.is_portrait_mode = is_portrait_mode
        if model_execution_lock is None:
            model_execution_lock = nullcontext()
        self.model_execution_lock = model_execution_lock

        print(f"   🔧 Initializing RAFT on {device}...")
        self.raft_tracker = RaftTracker(
            device='dml', dtype=torch.float32, H=H, W=W, num_landmarks=num_landmarks
        )

        # State
        self.current_flows = self.generate_grid_points()
        self.current_masks = torch.ones(num_landmarks, dtype=torch.bool, device=device)
        self.keyframe_tensor = None

        # Pre-allocate keyframe flows
        self.keyframe_flows = self.current_flows.clone()
        self.frame_count = 0

        # --- INTEGRATION BUFFERS (From original tracker.py) ---
        self.fresh_integration_mask = torch.zeros(num_landmarks, dtype=torch.bool, device=device)
        self.fresh_integration_mask_cp = torch.zeros(num_landmarks, dtype=torch.bool, device=device)
        # Allocate enough buffer space for operations
        self.arange_buf = torch.arange(num_landmarks + num_landmarks * 5 + 10, device=self.device)
        self.pending_integration_command = None
        self.candidate_lock = threading.Lock()

        print(f"   ✅ Tracker Initialized: {num_landmarks} points")

    def generate_grid_points(self):
        """Generate grid in [X, Y] format."""
        aspect = self.W / self.H
        if self.num_landmarks == 2:
            return torch.tensor([[self.W / 4, self.H / 2],
                                 [3 * self.W / 4, self.H / 2]], dtype=self.dtype, device=self.device)
        elif self.num_landmarks == 1:
            return torch.tensor([[self.W / 2, self.H / 2]], dtype=self.dtype, device=self.device)
        elif self.num_landmarks == 0:
            return torch.empty((0, 2), dtype=self.dtype, device=self.device)
        ny = int(np.sqrt(self.num_landmarks / aspect))
        nx = int(np.ceil(self.num_landmarks / ny))
        y = np.linspace(30, self.H - 30, ny)
        x = np.linspace(30, self.W - 30, nx)
        xv, yv = np.meshgrid(x, y)
        points = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)[:self.num_landmarks]
        if len(points) < self.num_landmarks:
            points = np.vstack([points, np.tile(points[-1:], (self.num_landmarks - len(points), 1))])
        return torch.from_numpy(points).to(self.device)

    def get_and_clear_fresh_mask(self):
        """Get and clear fresh mask atomically."""
        fresh = self.fresh_integration_mask_cp.copy_(self.fresh_integration_mask)
        self.fresh_integration_mask.zero_()
        return fresh

    def set_candidate_points(self, points, z_values, amp_values_norm, frame_tensor):
        """Store candidates WITH their pre-sampled z-values."""
        with self.candidate_lock:
            # We clone frame_tensor to ensure it persists safely until integration runs
            self.pending_integration_command = {
                "candidates_S": points,
                "z_values_S": z_values,
                "amp_values_norm_S": amp_values_norm,
                "keyframe_S": frame_tensor.detach().clone().contiguous(),
            }

    def _execute_integration_command(self, current_frame_tensor):
        """Execute integration and return z-values for fresh points."""
        command = None
        with self.candidate_lock:
            if self.pending_integration_command is not None:
                command = self.pending_integration_command
                self.pending_integration_command = None

        if command is None:
            return None

        candidates_S = command["candidates_S"]
        z_values_S = command["z_values_S"]
        amp_values_norm_S = command["amp_values_norm_S"]
        keyframe_S = command["keyframe_S"]

        is_dead_mask = ~self.current_masks
        num_dead = is_dead_mask.sum()
        num_candidates = len(candidates_S)

        # Non-blocking replacement logic
        rand_indices = torch.randperm(num_candidates, device=self.device)
        mask_indices = self.arange_buf[:num_candidates] < num_dead
        indices = rand_indices[mask_indices]

        selected_S = candidates_S[indices]
        selected_zs_S = z_values_S[indices]
        selected_amps_norm_S = amp_values_norm_S[indices]
        self.verb_print(f"   ... running S→T mini-track for {len(selected_S)} points")

        # --- PREPARE INPUTS FOR RAFT (APPLYING THE FIX) ---
        # Ensure dimensions match what RAFT expects: (1, C, H, W)
        # keyframe_S is (H, W, C) -> unsqueeze -> (1, H, W, C) -> permute -> (1, C, H, W)
        if keyframe_S.ndim == 3:
            keyframe_S_in = keyframe_S.unsqueeze(0).permute(0, 3, 1, 2)
            current_frame_in = current_frame_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        else:
            keyframe_S_in = keyframe_S.permute(0, 3, 1, 2)
            current_frame_in = current_frame_tensor.permute(0, 3, 1, 2)

        if self.is_portrait_mode:
            selected_S = selected_S[:, [1, 0]]
            # Note: Permuting the image itself for portrait mode if needed is handled by logic
            # outside RAFT usually, but if your model is landscape-only, the above permute
            # handles the channel ordering. If you need 90deg rotation, that's separate.
            # Assuming standard behavior here as per main update loop.

        # Track Candidates
        selected_T, valid_T, _, _ = self.raft_tracker.track_points(
            keyframe_S_in,
            current_frame_in,
            selected_S.to(dtype=torch.float32)
        )

        if self.is_portrait_mode:
            selected_T = selected_T[:, [1, 0]]

        # Flatten outputs if batch dim exists
        if selected_T.ndim == 3: selected_T = selected_T[0]
        if valid_T.ndim == 2: valid_T = valid_T[0]

        dead_indices = torch.where(is_dead_mask)[0]
        valid_indices_into_dead = torch.where(valid_T)[0]
        valid_new_indices = dead_indices[valid_indices_into_dead]

        valid_new_points_T = selected_T[valid_T]
        valid_zs_S = selected_zs_S[valid_T]
        valid_amps_norm_S = selected_amps_norm_S[valid_T]

        final_integration_mask = valid_new_indices

        self.current_flows[final_integration_mask] = valid_new_points_T
        self.current_masks[final_integration_mask] = True
        self.fresh_integration_mask[final_integration_mask] = True
        # self.verb_print(f"   ✓ Synced {len(valid_new_indices)} new points")

        return {
            'integrated': True,
            'fresh_indices': valid_new_indices,
            'fresh_z_values': valid_zs_S,
            'normalized_amp_values': valid_amps_norm_S
        }

    def update(self, frame_tensor):
        """
        Frame-to-Frame Tracking with Correct Permutation.
        """

        # 1. SANITIZE INPUT
        current_frame_clean = frame_tensor.detach().clone().contiguous()

        # 2. INITIALIZATION
        if self.keyframe_tensor is None:
            self.keyframe_tensor = torch.empty_like(current_frame_clean)
            self.keyframe_tensor.copy_(current_frame_clean)
            self.keyframe_flows[:] = self.current_flows
            self.frame_count += 1
            return self.current_flows, self.current_masks, None, None

        # 3. PREPARE INPUTS (Permute Here)
        if self.keyframe_tensor.ndim == 3:
            keyframe_input = self.keyframe_tensor.unsqueeze(0).permute(0, 3, 1, 2)
            current_input = current_frame_clean.unsqueeze(0).permute(0, 3, 1, 2)
        else:
            keyframe_input = self.keyframe_tensor.permute(0, 3, 1, 2)
            current_input = current_frame_clean.permute(0, 3, 1, 2)

        points_input = self.keyframe_flows.clone()

        # 4. TRACK (T-1 -> T)
        new_points, valid, _, _ = self.raft_tracker.track_points(
            keyframe_input,
            current_input,
            points_input
        )

        if new_points.ndim == 3: new_points = new_points[0]
        if valid.ndim == 2: valid = valid[0]

        # Filter
        self.current_masks[~valid] = False
        self.current_flows[self.current_masks] = new_points[self.current_masks]

        # 5. INTEGRATION (Added Back)
        fresh_mask = self.get_and_clear_fresh_mask()
        integration_result = self._execute_integration_command(current_frame_clean)

        # 6. UPDATE KEYFRAME (T becomes T-1)
        self.keyframe_tensor.copy_(current_frame_clean)
        self.keyframe_flows[:] = self.current_flows

        self.frame_count += 1
        return self.current_flows, self.current_masks, fresh_mask, integration_result

    def verb_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class XYZPointTracker:
    """Main 3D tracking pipeline."""
    def __init__(
        self, W, H, K, num_landmarks=2000, dtype=torch.float32, device='cuda', dsp=None, is_portrait_mode=False,
        verbose=True
    ):
        self.W, self.H = W, H
        self.verbose = verbose
        self.K = K
        self.device = device
        self.num_landmarks = num_landmarks
        self.dtype = dtype
        self.tracker_lock = threading.Lock()
        self.model_execution_lock = threading.Lock()
        # Grids sample does not support float16 on CPU, neither is it better
        self.tracker = TrackingPipeline(
            H, W, num_landmarks=num_landmarks, dtype=dtype, device=device,
            model_execution_lock=self.model_execution_lock, is_portrait_mode=is_portrait_mode, verbose=verbose
        )
        self.accum_z_amp = torch.ones((num_landmarks, 2), dtype=dtype, device=device)
        self.accum_z_amp[:, 1] = 0.3
        self.accum_valid = torch.zeros(num_landmarks, device=device, dtype=torch.bool)
        self.depth_sampler = MultiThreadedZSampler(
            H=H, W=W, K=K, num_landmarks=num_landmarks, tracker=self.tracker, parent=self, dsp=dsp,
            model_execution_lock=self.model_execution_lock, verbose=verbose
        )
        self.color_heuristics_echo_db = -1.0
        self.alive_mask = torch.ones(num_landmarks, dtype=torch.bool, device=device)
        self.frame_count = 0
        self._absorbance = torch.tensor(-6, dtype=dtype, device=device)
        self.fresh_mask = torch.ones(num_landmarks, dtype=torch.bool, device=device)
        self.xyz_amp = torch.empty((self.num_landmarks, 4),
                                   device=self.device,
                                   dtype=self.dtype)

        # Buffers
        N = self.num_landmarks

        # buffer for indices of not_fresh elements (worst case: all landmarks)
        self.not_fresh_idx = torch.empty(N, dtype=torch.int64, device=self.device)
        self.min_distance = torch.empty(N, dtype=dtype, device=device)

        # Debug tools ===========================================================================
        self.alive_mask_ones_debug = torch.ones(num_landmarks, dtype=torch.bool, device=device)
        self.alive_mask_zeros_debug = torch.zeros(num_landmarks, dtype=torch.bool, device=device)

        self.landmarks_debug = torch.empty((num_landmarks, 3), dtype=dtype, device=device)
        self.landmarks_debug[..., 0] = W/2
        self.landmarks_debug[..., 1] = H/2
        self.landmarks_debug[..., 2] = 1.0

        self.landmarks4_debug = torch.empty((num_landmarks, 4), dtype=dtype, device=device)
        self.landmarks4_debug[..., 0] = W / 2
        self.landmarks4_debug[..., 1] = H / 2
        self.landmarks4_debug[..., 2] = 1.0
        self.landmarks4_debug[..., 3] = 1.0

    def verb_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def close(self):
        self.depth_sampler.close()

    def absorbance(self):
        return self._absorbance

    def room_head_geometry(self):
        return self.depth_sampler.room_head_geometry()

    @stable_checker
    def track(
        self, frame, frame_timestamp_sec, should_queue_regeneration=False, rotation_matrix=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Main tracking with IMMEDIATE z-value initialization for fresh points."""
        # Ignore absorbance estimation completely
        # if self.frame_count % 4 == 0:
        #     self.color_heuristics_echo_db = self.depth_sampler.absorption_estimator.estimate_echo_db(frame) + self.color_heuristics_echo_db * 0.8
        #     self._absorbance = self.color_heuristics_echo_db + self.depth_sampler.depth_heuristics_echo_db
        self.frame_count += 1

        # Next is from depth map
        with self.tracker_lock:
            # This is Not a bottleneck
            landmarks, alive_mask, _, integration_result = self.tracker.update(frame)
            self.alive_mask = alive_mask
            fresh_mask = self.fresh_mask
            fresh_mask.zero_()
            just_initialized = False
            if integration_result is not None and integration_result.get('integrated'):
                just_initialized = True
                fresh_indices = integration_result['fresh_indices']
                fresh_zs = integration_result['fresh_z_values']
                normalized_amp_values = integration_result['normalized_amp_values']
                if fresh_zs.dtype != self.dtype:
                    fresh_zs = fresh_zs.to(self.dtype)
                self.accum_z_amp[fresh_indices, 0] = fresh_zs
                self.accum_z_amp[fresh_indices, 1] = normalized_amp_values
                self.accum_valid[fresh_indices] = True
                self.depth_sampler.integration_period = True
                fresh_mask[fresh_indices] = True
                self.verb_print(f"   🎉 IMMEDIATE HARD RESET: fresh points initialized!")

            self.depth_sampler.queue_work(
                frame, self.tracker.current_flows, self.tracker.current_masks, frame_timestamp_sec,
                should_queue_regeneration=should_queue_regeneration, rotation_matrix=rotation_matrix
            )
            z_data = self.depth_sampler.get_depth_data()
            # self.xyz_amp[:, :2] = landmarks[:, :2]
            # self.xyz_amp[:, 2:] = self.accum_z_amp
            # return self.xyz_amp, alive_mask, fresh_mask, just_initialized
            if z_data is not None:
                # not_fresh = ~fresh_mask
                # if (
                #     integration_result is None or not integration_result.get('integrated')
                # ):
                # ) and not self.depth_sampler.integration_period: #
                    # TODO: For now
                self.accum_z_amp[:, 0] = z_data['zs']
                # self.accum_z_amp[not_fresh, 0] = z_data['zs'][not_fresh]
                # new_zs = z_data['zs'][not_fresh]
                # old_zs = self.accum_z_amp[not_fresh, 0]
                # # Calculate alpha | len is fine here because we use CPU in this pipeline
                # # min_distance = torch.min(new_zs, old_zs, out=self.min_distance[:len(not_fresh)])
                # # alpha = (min_distance.neg_()).mul_(0.1).exp_()
                # alpha = 0.8 # Since computation is really slow and depth does not flicker, We do this. Low pass filter handles softness
                # new_zs.mul_(alpha)
                # if isinstance(alpha, torch.Tensor):
                #     alpha.neg_().add_(1.0)
                # else:
                #     alpha = 1.0 - alpha
                # old_zs.mul_(alpha)
                # self.accum_z_amp[not_fresh, 0] = old_zs.add_(new_zs)
                # Amps will forever stay consistent
            self.xyz_amp[:, :2] = landmarks[:, :2]
            self.xyz_amp[:, 2:] = self.accum_z_amp
            return self.xyz_amp, alive_mask, fresh_mask, just_initialized

class SensorFusedXYZPointTracker(XYZPointTracker):
    def __init__(
        self, W, H, K, view_sensor, head_mounted_sensor, num_landmarks,
        dtype=torch.float32, device='cuda', cutoff_freq=4.3, dsp=None, is_portrait_mode=False, verbose=True
    ):
        # Make the entire tracking Pipeline with all depth estimation code on CPU and IGPU
        # This guarantees DSP runs at maximum CUDA performance
        tracker_device = 'cpu'
        super().__init__(
            W, H, K, num_landmarks=num_landmarks, dtype=dtype, device=tracker_device, dsp=dsp,
            is_portrait_mode=is_portrait_mode, verbose=verbose
        )
        self.tracker_device = tracker_device
        self.main_device = device
        self.cutoff_freq = cutoff_freq
        self.lowpass_filter = LowPassFilter(
            cutoff_freq=cutoff_freq, dtype=dtype, device=device, max_nodes=10
        )
        self.camera_transform_tracker = CameraProjection(
            num_landmarks=num_landmarks, H=H, W=W, K=K, dtype=dtype, device=tracker_device
        )
        self.camera_transform_dsp = CameraProjection(
            num_landmarks=num_landmarks, H=H, W=W, K=K, dtype=dtype, device=device
        )
        self.dtype = dtype
        self.view_sensor = view_sensor
        self.head_mounted_sensor = head_mounted_sensor
        self.sensor_id = makeid(10)
        # Cutoff freq and history window must match precisely
        self.view_sensor.make_reading(
            self.sensor_id,
            cutoff_freq=cutoff_freq
        )
        if self.view_sensor is not self.head_mounted_sensor:
            self.head_mounted_sensor.make_reading(
                self.sensor_id,
                cutoff_freq=cutoff_freq
            )

        self.velocities = torch.zeros([num_landmarks, 3], dtype=dtype, device=tracker_device)
        self.position_offsets = torch.zeros([num_landmarks, 3], dtype=dtype, device=tracker_device)
        self.frozen_world_landmarks = torch.zeros([num_landmarks, 4], dtype=dtype, device=tracker_device)
        self.positions_buf = torch.zeros([num_landmarks, 4], dtype=dtype, device=self.main_device)
        self.gpu_reset_values = torch.zeros([num_landmarks, 3], dtype=dtype, device=self.main_device)
        self.gpu_fresh_mask_ = torch.zeros(num_landmarks, dtype=torch.bool, device=self.main_device)
        self.frozen_world_landmarks[:, 0] = W / 2
        self.frozen_world_landmarks[:, 1] = H / 2
        self.frozen_world_landmarks[:, 2] = 1.0
        self.frozen_world_landmarks[:, 3] = 0.4
        self.last_points_in_world_frame = torch.zeros([num_landmarks, 3], dtype=dtype, device=tracker_device)
        self.last_frame_timestamp_sec = None
        self.is_updated_mask = torch.zeros(num_landmarks, dtype=torch.bool, device=device)
        self.is_new_updates = False
        self.is_updated_mask_change_lock = threading.Lock()
        self.rotation_gpu = torch.empty((3, 3), dtype=self.dtype, device=device)
        self.global_point_buf = torch.empty((num_landmarks, 4), dtype=dtype, device=tracker_device)
        self.output_positions = torch.empty((num_landmarks, 4), dtype=dtype, device=tracker_device)
        self.point_in_camera_frame_buf = torch.empty((num_landmarks, 4), dtype=dtype, device=device)
        self.rotation_out = torch.empty(
            (3, 3),
            dtype=self.dtype,
            device=self.device
        )

    def track(self, frame, frame_timestamp_sec, should_queue_regeneration=False, rotation_matrix=None) -> tuple:
        points_, alive_mask, fresh_mask, just_initialized = super().track(
            frame, frame_timestamp_sec,  should_queue_regeneration=should_queue_regeneration,
            rotation_matrix=rotation_matrix
        )

        if points_ is None:
            return None, None

        imu_to_world_rotation = self.view_sensor.rotation_matrix_when(frame_timestamp_sec)
        if imu_to_world_rotation is None:
            return None, None

        point_in_camera_frame = self.camera_transform_tracker.pixels_to_camera_(points_)
        point_in_camera_frame[..., :2] *= -1  # Flip XY
        point_in_imu_frame = point_in_camera_frame[..., :3]
        point_in_world_frame = point_in_imu_frame @ imu_to_world_rotation.T
        # This is not thread safe
        global_point = self.global_point_buf
        global_point[..., :3] = point_in_world_frame
        global_point[..., 3:] = point_in_camera_frame[..., 3:]
        # # We want, velocity on dead nodes and also "NOT" Lowpass new nodes, reset the lowpass for specific indices
        if just_initialized:
            fresh_mask_ = fresh_mask
            reset_values = point_in_world_frame[fresh_mask, :3]
            if reset_values.device != self.main_device and 'cuda' in str(self.main_device):
                # This is already on the CPU
                length = fresh_mask.sum()
                if 'cpu' in str(reset_values.device):
                    reset_values = reset_values.pin_memory()
                reset_values = self.gpu_reset_values[:length].copy_(reset_values, non_blocking=True)

                if 'cpu' in str(fresh_mask_.device):
                    fresh_mask_ = fresh_mask_.pin_memory()
                fresh_mask_ = self.gpu_fresh_mask_.copy_(fresh_mask_, non_blocking=True)

            self.lowpass_filter.values[:, fresh_mask_, :3] = reset_values
            self.lowpass_filter.filtered[:, fresh_mask_, :3] = reset_values

            fresh_amps = point_in_camera_frame[fresh_mask, 3]

            # Ensure device matching (gpu_reset_values logic handles XYZ, we handle Amp here)
            if fresh_amps.device != self.lowpass_filter.values.device:
                fresh_amps = fresh_amps.to(self.lowpass_filter.values.device, non_blocking=True)

            self.lowpass_filter.values[:, fresh_mask_, 3] = fresh_amps
            self.lowpass_filter.filtered[:, fresh_mask_, 3] = fresh_amps

        # Freeze world positions for newly dead points
        self.frozen_world_landmarks[alive_mask] = global_point[alive_mask]
        self.velocities[fresh_mask] = 0.0
        # Reset for fresh points
        if just_initialized:
            with self.is_updated_mask_change_lock:
                self.is_new_updates = True
                self.is_updated_mask[fresh_mask] = True

        # For velocity calculation, use current tracked position
        velocity_updatable_mask = ~fresh_mask & alive_mask
        position_updatable_mask = ~fresh_mask & ~alive_mask
        if self.last_points_in_world_frame is not None and self.last_frame_timestamp_sec is not None:
            delta = (frame_timestamp_sec - self.last_frame_timestamp_sec)
            if delta > 0.00001:
                vel = (point_in_world_frame[velocity_updatable_mask] - self.last_points_in_world_frame[velocity_updatable_mask]) / delta
                self.velocities[velocity_updatable_mask] = vel * 0.01 + self.velocities[velocity_updatable_mask] * 0.99
                self.frozen_world_landmarks[position_updatable_mask, :3] += self.velocities[position_updatable_mask] * delta
        self.last_frame_timestamp_sec = frame_timestamp_sec
        self.last_points_in_world_frame = point_in_world_frame # Only update for alive!
        # Use frozen positions for dead points
        output_positions = self.output_positions.copy_(global_point)
        output_positions[position_updatable_mask] = self.frozen_world_landmarks[position_updatable_mask]
        self.lowpass_filter.add_node(output_positions, frame_timestamp_sec)

    def clear_updated_mask(self):
        if self.is_new_updates:  # Put this check up here,
            # The idea is we must never clear right after thread ands new.
            with self.is_updated_mask_change_lock:
                self.is_updated_mask[:] = False
                self.is_new_updates = False
                return True
        return False

    @stable_checker
    def get_position_global(self, t) -> Union[None, torch.Tensor]:
        # This will be derived in the DSP thread will combine with high passed Accel
        highpass_position = self.head_mounted_sensor.get_highpass_position(self.sensor_id, t)
        positions = self.lowpass_filter.get_value(t)
        if positions is None:
            return None
        positions = self.positions_buf.copy_(positions)
        if highpass_position is not None:
            # Inverse of the camera
            positions[..., :3] -= highpass_position
        return positions

    @stable_checker
    def get_position_screen_space(self, t):
        # This is CUDA
        point_in_world_frame = self.get_position_global(t)
        if point_in_world_frame is None:
            return None, self.alive_mask
        return self.transform_global_point_to_screen_space_(t, point_in_world_frame), self.alive_mask

    def transform_global_point_to_screen_space_(self, frame_timestamp_sec, point_in_world_frame, include_c=True):
        # We use CUDA here
        imu_to_world_rotation = self.head_mounted_sensor.rotation_matrix_when(
            frame_timestamp_sec, device=self.main_device
        )
        if imu_to_world_rotation is None:
            return None
        # Transform: World → IMU → Camera
        point_in_world_frame[..., :3] = point_in_world_frame[..., :3] @ imu_to_world_rotation
        # Apply the flip fix we discovered
        point_in_world_frame[..., :2] *= -1  # Flip XY
        pixel_point = self.camera_transform_dsp.camera_to_pixels_(point_in_world_frame, include_c=include_c)
        return pixel_point

    def get_rotation(self, t):
        if t == 'now':
            return self.head_mounted_sensor.rotation_matrix_now(device=self.main_device)
        # The inverse is the true camera rotation
        src = torch.from_numpy(self.head_mounted_sensor.rotation_matrix_when(t, device=self.main_device))
        self.rotation_gpu.copy_(src, non_blocking=True)
        return self.rotation_gpu


# ============================================================================
# ULTRA-OPTIMIZED RENDERING ENGINE (MAXIMUM PERFORMANCE)
# ============================================================================
class HighPerformanceRenderer(DepthMapColorizer):
    """Maximum performance rendering with BGR-native path and pinned memory."""
    def __init__(self, H, W, num_landmarks, dtype=torch.float32, device='cuda', thickness=2, lut_size=4096):
        super().__init__(dtype=dtype, device=device, lut_size=lut_size)
        self.H = H
        self.W = W
        self.num_landmarks = num_landmarks
        self.dtype = dtype
        self.device = device
        self.thickness = thickness

        # Pre-compute circle mask
        r = int(thickness)
        yy, xx = torch.meshgrid(
            torch.arange(-r, r + 1, device=device),
            torch.arange(-r, r + 1, device=device),
            indexing='ij'
        )
        circle_mask = (xx * xx + yy * yy) <= (r * r)
        self.offsets = torch.stack([xx[circle_mask], yy[circle_mask]], dim=1)
        self.num_offsets = self.offsets.shape[0]
        self.color_lut_bgr = self.color_lut_rgb[:, [2, 1, 0]]
        # Pre-allocate rendering buffers
        max_rendered = num_landmarks * self.num_offsets
        self.expanded_buffer = torch.zeros((max_rendered, 2), dtype=torch.int64, device=device)
        self.color_buffer = torch.zeros((max_rendered, 3), dtype=torch.uint8, device=device)
        print(f"✓ HighPerformanceRenderer: BGR-native, {self.lut_size} LUT entries, pinned buffers")

    @torch.no_grad()
    def get_colors_from_lut(self, z_values):
        """Fast color lookup using pre-computed BGR LUT."""
        z_clamped = torch.clamp(z_values, 0.01, 50000.0)
        log_z = torch.log10(z_clamped)

        t = (log_z - self.log_min) / self.log_range
        indices = (t * (self.lut_size - 1)).long()
        indices = torch.clamp(indices, 0, self.lut_size - 1)
        return self.color_lut_bgr[indices]

    @torch.no_grad()
    def render(self, frame, points, z_values=None, override_color=None):
        # """Ultra-fast rendering with BGR output."""
        if points is None:
            return frame

        pts = points[:, :2].long()
        if pts.shape[0] == 0:
            return frame
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < self.W) & (pts[:, 1] >= 0) & (pts[:, 1] < self.H)
        pts = pts[valid]
        if pts.shape[0] == 0:
            return frame

        # Get BGR colors
        if override_color is not None:
            colors = override_color.expand(pts.shape[0], 3).byte()
        elif z_values is not None:
            z_vals = z_values[valid]
            colors = self.get_colors_from_lut(z_vals)
        else:
            colors = torch.full((pts.shape[0], 3), 255, dtype=frame.dtype, device=frame.device)

        # Vectorized expansion
        expanded = pts.unsqueeze(1) + self.offsets.unsqueeze(0)
        expanded[..., 0].clamp_(0, self.W - 1)
        expanded[..., 1].clamp_(0, self.H - 1)

        flat_pts = expanded.reshape(-1, 2)
        flat_colors = colors.repeat_interleave(self.num_offsets, dim=0)
        frame[flat_pts[:, 1], flat_pts[:, 0]] = flat_colors
        return frame

class PinnedMemoryPool:
    """Pinned memory pool for fast CPU-GPU transfers."""
    def __init__(self, OH, OW, H, W, device='cuda'):
        self.device = device
        # Pre-allocate pinned memory buffers
        # For RGB uint8 input frame
        self.frame_buffer_rgb = torch.zeros((OH, OW, 3), dtype=torch.uint8, device=device)
        # For BGR uint8 output frame
        self.frame_buffer_bgr = torch.zeros((H, W, 3), dtype=torch.uint8).pin_memory()
        self.frame_buffer_bgr_np = self.frame_buffer_bgr.numpy()
        print(f"✓ PinnedMemoryPool: {H}x{W} buffers allocated")

    def upload_frame(self, frame_np):
        """Fast upload from numpy to GPU via pinned memory."""
        # Copy numpy data into pinned buffer (fast mem cpy)
        self.frame_buffer_rgb.copy_(torch.from_numpy(frame_np))
        # Non-blocking transfer to GPU
        return self.frame_buffer_rgb

    def download_frame(self, frame_gpu):
        """Fast download from GPU to numpy via pinned memory."""
        # Non-blocking transfer to CPU pinned memory
        self.frame_buffer_bgr.copy_(frame_gpu, non_blocking=False)
        # Return numpy view (no copy)
        return self.frame_buffer_bgr_np

class FastTextRenderer:
    """Optimized CPU text rendering."""

    @staticmethod
    def render_stats_cpu(frame_np, fps, num_alive, total_points, num_fresh,
                         frame_count, total_fresh, regen_count, frames_until_regen):
        """Batch text rendering."""
        cv2.putText(frame_np, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_np, f"Alive: {num_alive}/{total_points}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_np, f"Fresh: {num_fresh}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if num_fresh > 0 else (128, 128, 128), 2)
        cv2.putText(frame_np, f"Frame (since key): {frame_count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_np, f"Total Fresh: {total_fresh}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_np, f"Regenerations: {regen_count}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(frame_np, f"Next regen in: ~{frames_until_regen}f", (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        return frame_np

class RemoveDistortion:
    """
    Cached CUDA un-distortion - computes mapping once, reuses for all frames.
    Results are identical to OpenCV cv2.undistort.
    """

    def __init__(self, K, dtype, dist_coefficients, image_shape, device='cuda'):
        """
        Args:
            K: (3, 3) camera intrinsic matrix
            dist_coefficients: distortion coefficients [k1, k2, p1, p2, k3, ...]
            image_shape: (H, W) tuple
            device: torch device
        """
        self.device = device
        self.H, self.W = image_shape

        # Convert to tensors
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, dtype=dtype, device=device)
        else:
            K = K.to(dtype=dtype, device=device)

        if not isinstance(dist_coefficients, torch.Tensor):
            dist_coefficients = torch.tensor(dist_coefficients, dtype=dtype, device=device)
        else:
            dist_coefficients = dist_coefficients.to(device)

        # Pad dist_coefficients to at least 5 elements
        if len(dist_coefficients) < 5:
            dist_coefficients = torch.cat([dist_coefficients, torch.zeros(5 - len(dist_coefficients), device=device)])

        # Extract intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create pixel coordinate grid for OUTPUT (undistorted) image
        y, x = torch.meshgrid(
            torch.arange(self.H,  dtype=dtype, device=device),
            torch.arange(self.W,  dtype=dtype, device=device),
            indexing='ij'
        )

        # Normalize to camera coordinates
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy

        # Iteratively find distorted coordinates (OpenCV uses iterative refinement)
        # For speed, we'll use the inverse distortion approximation
        # This matches OpenCV approach
        x_dist_norm = x_norm
        y_dist_norm = y_norm

        k1, k2, p1, p2, k3 = dist_coefficients[0], dist_coefficients[1], dist_coefficients[2], dist_coefficients[3], dist_coefficients[4]
        k4, k5, k6 = (dist_coefficients[5], dist_coefficients[6], dist_coefficients[7]) if len(dist_coefficients) > 7 else (0.0, 0.0, 0.0)

        # Iterative refinement (5 iterations like OpenCV)
        for _ in range(5):
            r2 = x_dist_norm ** 2 + y_dist_norm ** 2
            r4 = r2 * r2
            r6 = r4 * r2

            # Radial distortion
            radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)

            # Tangential distortion
            dx = 2 * p1 * x_dist_norm * y_dist_norm + p2 * (r2 + 2 * x_dist_norm ** 2)
            dy = p1 * (r2 + 2 * y_dist_norm ** 2) + 2 * p2 * x_dist_norm * y_dist_norm

            # Apply distortion
            x_distorted = x_dist_norm * radial + dx
            y_distorted = y_dist_norm * radial + dy

            # Update estimate
            x_dist_norm = x_norm - (x_distorted - x_dist_norm)
            y_dist_norm = y_norm - (y_distorted - y_dist_norm)

        # Final distorted coordinates
        r2 = x_dist_norm ** 2 + y_dist_norm ** 2
        r4 = r2 * r2
        r6 = r4 * r2

        radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        dx = 2 * p1 * x_dist_norm * y_dist_norm + p2 * (r2 + 2 * x_dist_norm ** 2)
        dy = p1 * (r2 + 2 * y_dist_norm ** 2) + 2 * p2 * x_dist_norm * y_dist_norm

        x_distorted = x_dist_norm * radial + dx
        y_distorted = y_dist_norm * radial + dy

        # Convert back to pixel coordinates (these are the SOURCE coordinates in distorted image)
        map_x = x_distorted * fx + cx
        map_y = y_distorted * fy + cy

        # Normalize to [-1, 1] for grid_sample
        grid_x = 2 * map_x / (self.W - 1) - 1
        grid_y = 2 * map_y / (self.H - 1) - 1

        # Cache the grid (1, H, W, 2)
        self.grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    def undistort(self, frame):
        """
        Apply cached un-distortion to a frame.

        Args:
            frame: (H, W, 3) float tensor on same device

        Returns:
            Undistorted (H, W, 3) float tensor
        """
        # Rearrange to (1, 3, H, W) for grid_sample
        frame_permuted = frame.permute(2, 0, 1).unsqueeze(0)
        # Apply cached grid mapping
        undistorted = functional.grid_sample(
            frame_permuted,
            self.grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        # Back to (H, W, 3)
        return undistorted.squeeze(0).permute(1, 2, 0)

    def __call__(self, frame):
        """Convenience method"""
        return self.undistort(frame)

class VisualLandmarksTrackerPipe:
    def __init__(
        # self, view_sensor, head_mounted_sensor, video_ip_and_port="10.26.208.31:8080", K=np.array([
        self, view_sensor, head_mounted_sensor, video_ip_and_port="10.26.212.51:8080", K=np.array([
        # self, view_sensor, head_mounted_sensor, video_ip_and_port="192.168.182.155:8080", K=np.array([
        #     [1346.2626115837832, 0.0, 971.1651220753282],
        #     [0.0, 1345.375887563327, 531.1205273630911],
        #     [0.0, 0.0, 1.0]
                [579.8659319515167, 0.0, 714.5873577485696],
                [0.0, 579.7672780723192, 538.4394297536285],
                [0.0, 0.0, 1.0]
            ]), dist_coefficients=np.array([-0.033823457132834266, 0.0, 0.0, 0.0, 0.0]),
            is_portrait_mode=False,
            W=1440, H=1080, downsample_scale=5, device='cuda',
            # W=1920, H=1080, downsample_scale=5, device='cuda',
            dtype=torch.float32, num_landmarks=1, dsp=None, visualize_projection=False, verbose=True
    ):
        self.verbose = verbose
        self.initial_w = W
        self.initial_h = H
        W = W // downsample_scale
        H = H // downsample_scale
        tracker_device = 'cpu'
        self.K_raw = K.copy()
        self.dtype = dtype
        self.device = device
        self.frame_count = 0
        self.or_w = W
        self.or_h = H
        self.last_time = time.time()
        self.raw_intrinsics = K.copy()
        self.visualizer = None
        self.is_portrait_mode = is_portrait_mode
        self.last_time = time.time()
        if is_portrait_mode:
            K = np.array([
                [K[1, 1], 0.0, K[1, 2]],
                [0.0, K[0, 0], K[0, 2]],
                [0.0, 0.0, 1.0]
            ])
            W, H = H, W
        self.W = W
        self.H = H
        self.pixel_scale = 1.0 / min(W, H)
        self.downsample_scale = downsample_scale
        self.num_landmarks = num_landmarks
        self.video_ip_and_port = video_ip_and_port
        self.view_sensor = view_sensor
        self.head_mounted_sensor: IMUSensing = head_mounted_sensor
        self.regeneration_lock = threading.Lock()
        self.frame_grabber = SingleShotCamera(
            height=self.initial_h, width=self.initial_w,
            ip_and_port=self.video_ip_and_port, downsample_scale=downsample_scale, dtype=torch.float32, device=tracker_device
        )
        self.fps_counter = 0
        self.start_time = time.time()
        self.is_open = True
        K = K.copy()
        self.K = K
        self.dist_coefficients = dist_coefficients
        self.last_ping_t = None
        scale = 1.0 / downsample_scale
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale

        # This is on CPU so beware of performance HITS
        # self.un_distort = RemoveDistortion(
        #     K=K, dist_coefficients=dist_coefficients, image_shape=(H, W),
        #     dtype=self.frame_grabber.dtype, device=self.frame_grabber.device
        # )
        self.spatial_tracker = SensorFusedXYZPointTracker(
            H=H, W=W, K=K, view_sensor=self.view_sensor, head_mounted_sensor=self.head_mounted_sensor,
            num_landmarks=num_landmarks, dtype=dtype, device=device, dsp=dsp,
            is_portrait_mode=is_portrait_mode, verbose=verbose
        )

        # Initialize ultra-optimized components
        self.point_renderer = HighPerformanceRenderer(
            H=H, W=W, num_landmarks=num_landmarks, dtype=self.frame_grabber.dtype, device=self.spatial_tracker.tracker_device, thickness=2,
        )
        self.legend_renderer = HighPerformanceLegendRenderer(
            H=H, W=W, device=self.spatial_tracker.tracker_device, dtype=self.frame_grabber.dtype
        )
        self.memory_pool = PinnedMemoryPool(OH=self.or_h, OW=self.or_w, H=H, W=W, device=tracker_device)
        self.frame_times = deque(maxlen=30)
        self.last_status = time.time()
        self.total_fresh_seen = 0
        self.regeneration_count = 0
        self._reinitialization_latency = 0.15  # Default latency before true latency is computed
        self.last_time = time.time()
        self._vector = torch.empty([num_landmarks, 7], dtype=dtype, device=device)
        self._is_reinitializing_now = False
        self.visualizer_available = True
        self.visualize_projection = visualize_projection
        self.zeros_frame = torch.zeros(
            (self.or_h, self.or_w, 3), dtype=self.dtype, device=self.device
        )
        self.frame_height = None
        self.frame_width = None
        self.frame_buffer_rgb_f16 = None  # Final RGB float32 buffer
        self.frame_buffer_rgb_u8 = None  # Intermediate RGB uint8 buffer
        self.landmarks_buf = torch.empty((self.num_landmarks, 6), dtype=self.dtype, device=self.device)
        self.cpu_landmarks = torch.empty((num_landmarks, 4), dtype=self.dtype, device='cpu', pin_memory=True)
        self.cpu_alive_mask = torch.empty(num_landmarks, dtype=self.dtype,  device='cpu', pin_memory=True)
        self.alive_mask = torch.empty(num_landmarks, dtype=self.dtype,  device=self.device)
        self.tmp_abs = torch.empty((num_landmarks, 2), dtype=dtype, device=device)
        self.tmp_mul = torch.empty((num_landmarks, 2), dtype=dtype, device=device)

    def reinitialization_latency(self):
        return self._reinitialization_latency

    def reinitialize(self):
        with self.regeneration_lock:
            # This is the final step, it is better to always re-initialize
            # self.spatial_tracker.tracker.current_masks[:] = False
            self._is_reinitializing_now = True

    def transform_frame(self, frame_tensor_f16):
        if self.is_portrait_mode:
            # Permute creates a view (no allocation)
            frame_view = frame_tensor_f16.permute(1, 0, 2)
            # Flip + BGR→RGB conversion into pre-allocated buffer
            # Flip means we reverse dimension 1, and we swap channels 0↔2
            frame_view = torch.flip(frame_view, dims=[1])  # R ← B flipped
        else:
            # BGR→RGB conversion into pre-allocated buffer (no flip needed)
            frame_view = frame_tensor_f16
        return frame_view

    @torch.inference_mode()
    def refresh(self, should_queue_regeneration=False):
        if not self.is_open:
            return False

        ret, frame_tensor_f = self.frame_grabber.read()
        if not ret:
            print(f"{RED}⚠️  Frame grabber disconnected. Closing renderer.{RESET}")
            self.close()
        if isinstance(frame_tensor_f, dict):
            frame_timestamp = frame_tensor_f['timestamp_ns']
            frame_tensor_f = frame_tensor_f['frame']
        elif self.head_mounted_sensor is not None:
            frame_timestamp = self.head_mounted_sensor.latest_sensor_timestamp_ns
            if frame_timestamp is None:
                # Do not read frames until we have a valid device time
                return True
        else:
            frame_timestamp = time.time_ns()
        if frame_timestamp is None or frame_tensor_f is None:
            raise RuntimeError(f"Cannot fetch video: frame_timestamp={frame_timestamp} | frame={type(frame_tensor_f)}")
        frame_timestamp_sec = frame_timestamp * 1e-9
        rotation_matrix = self.view_sensor.rotation_matrix_when(
            frame_timestamp_sec, device='cpu'
        )
        frame_tensor_f = self.transform_frame(frame_tensor_f)
        # frame_tensor_f16 = self.un_distort(frame_tensor_f16)
        self.spatial_tracker.track(
            frame_tensor_f, frame_timestamp_sec=frame_timestamp_sec,
            should_queue_regeneration=should_queue_regeneration,
            rotation_matrix=rotation_matrix,
        )
        self.alive_mask.copy_(self.spatial_tracker.alive_mask, non_blocking=True)
        t = time.time()
        if self.frame_count % 30 == 0:
            self.verb_print(f'FPS TRACKER: {1 / (t - self.last_time + 0.0000001)}')
        self.last_time = t
        self.frame_count += 1
        if self.visualize_projection:
            landmarks, alive_mask = self.spatial_tracker.get_position_screen_space(frame_timestamp_sec)
            # These are separate device
            landmarks  = self.cpu_landmarks.copy_(landmarks, non_blocking=True)
            alive_mask = self.cpu_alive_mask.copy_(alive_mask, non_blocking=True)
            num_alive = 0 if alive_mask is None else alive_mask.sum().item()
            num_fresh = 0# if fresh_mask is None else fresh_mask.sum().item()
            if num_fresh > 0:
                self.total_fresh_seen += num_fresh
                self.regeneration_count += 1
                self.verb_print(f"   🎉 FRESH POINTS: {num_fresh} new points at frame {self.spatial_tracker.depth_sampler.last_regeneration_frame}")

            # Convert RGB to BGR on GPU (just channel reordering)
            display_bgr = frame_tensor_f[:, :, [2, 1, 0]].contiguous()
            # Render points in BGR (async on GPU)
            # xyz_amp = landmarks#[alive_mask]
            xyz_amp = landmarks#torch.cat([landmarks, pixel_landmarks])
            if xyz_amp is not None:
                display_bgr = self.point_renderer.render(
                    display_bgr, xyz_amp,
                    z_values=xyz_amp[:, 2]
                )
            # Blend legend (async on GPU)
            display_bgr = self.legend_renderer.blend(display_bgr)
            # Record end of GPU work (non-blocking)
            # Download to CPU (this blocks until GPU work is done)
            display_np = self.memory_pool.download_frame(display_bgr)
            frame_time = time.time()
            self.frame_times.append(frame_time)
            fps = 1.0 / np.mean(self.frame_times)
            # Text rendering (CPU only, after download)
            display_np = FastTextRenderer.render_stats_cpu(
                display_np, 0, num_alive, self.num_landmarks, num_fresh,
                self.spatial_tracker.tracker.frame_count, self.total_fresh_seen,
                self.regeneration_count, 0
            )

            # Display (already in BGR, no conversion needed)
            cv2.imshow("ULTRA-FAST Tracker", display_np)
            if time.time() - self.last_status >= 2.0:
                alive_pct = (num_alive / self.num_landmarks) * 100 if num_alive > 0 else 0
                if hasattr(self, 'spatial_tracker'):
                    self.verb_print(f"\r📊 Frame {frame_timestamp_sec} (Keyframe: {self.spatial_tracker.tracker.frame_count}): "
                          f"{fps:.1f} FPS | "
                          f"{num_alive}/{self.num_landmarks} alive ({alive_pct:.1f}%) | "
                          f"Fresh: {num_fresh} | "
                          f"Regens: {self.regeneration_count}", end='')
                self.last_status = time.time()
            if cv2.waitKey(4) & 0xFF == ord('q'):
                print("\n⏹️  Stopped by user")
                self.close()
                return False

    def close(self):
        self.verb_print(f"{RED}{BOLD}[==CLOSING VisualLandmarksTrackerPipe(...)==]{RESET}")
        self.is_open = False
        if self.visualizer is not None:
            self.visualizer.running = False
            self.visualizer.close()
        self.view_sensor.close()
        if self.view_sensor is not self.head_mounted_sensor:
            self.head_mounted_sensor.close()
        self.frame_grabber.release()
        self.spatial_tracker.close()
        self.visualizer_available = False
        cv2.destroyAllWindows()
        # Camera rotation is fetched in DSP Immediately
        # Position is constantly refined in IMU thread

    def opened(self):
        return self.is_open

    @stable_checker
    @profile
    def landmarks(self):
        # This gets points precisely in SpatialImpulse Coordinate Frame
        # We need time now
        t = time.time_ns() * 1e-9
        # This is CUDA
        position_ = self.spatial_tracker.get_position_global(t)
        if position_ is None:
            return None, None

        vector = self.landmarks_buf
        vector[:, :3] = position_[:, :3]
        vector[:, 5] = position_[:, 3]
        vector[:, 3:5] = DepthFocus.focus_point_(
            # Position is destroyed in place | Because we don't need it anymore
            self.spatial_tracker.transform_global_point_to_screen_space_(
                "now", position_, include_c=False)[:, :2].mul_(self.pixel_scale),
            tmp_abs=self.tmp_abs, tmp_mul=self.tmp_mul
        )
        # x,y,z, X-freq, Y-LFO, Z-AMP
        # X-freq, Y-LFO goes from -1 to 1
        return vector, self.spatial_tracker.is_updated_mask

    def clear_updated_mask(self):
        return self.spatial_tracker.clear_updated_mask()

    @stable_checker
    def absorbance(self):
        return self.spatial_tracker.absorbance()

    @stable_checker
    def room_head_geometry(self):
        return self.spatial_tracker.room_head_geometry()

    @stable_checker
    def head_rotation(self):
        return self.spatial_tracker.get_rotation("now")

    @torch.inference_mode()
    def start_background_update_blocking(self):
        while self.opened():
            # We start timing here
            time.sleep(0.00001)
            depth_thread_id = self.spatial_tracker.depth_sampler.current_thread_id
            with self.regeneration_lock:
                should_queue_regeneration = self._is_reinitializing_now
                self._is_reinitializing_now = False
            self.refresh(should_queue_regeneration=should_queue_regeneration)
            # What happened, is previous thread set this to true, but is now set to false immediately after
            # before a chance could be made
            latest_depth_thread_id = self.spatial_tracker.depth_sampler.current_thread_id
            if latest_depth_thread_id != depth_thread_id:
                time_t = time.time()
                if self.last_ping_t is not None:
                    self._reinitialization_latency = time_t - self.last_ping_t
                self.last_ping_t = time_t

    def start_background_update(self):
        thread = threading.Thread(target=self.start_background_update_blocking, args=(), daemon=True)
        thread.start()

    def run_visualizer_non_blocking(self):
        visualizer_thread = threading.Thread(target=self.run_visualizer_blocking, args=(), daemon=True)
        visualizer_thread.start()

    def run_visualizer_blocking(self):
        self.visualizer = VSAVisualizer(
            intrinsics=self.K_raw, width=1200, height=800, mode='portrait' if self.is_portrait_mode else 'landscape',num_landmarks=self.num_landmarks,
        )
        try:
            while self.visualizer_available and self.visualizer.running:
                time.sleep(0.0333)
                geometry = self.room_head_geometry()
                rotation_matrix = self.head_rotation().cpu().numpy()
                head_position = geometry[3]
                # self.visualizer.set_room_dimensions([geometry[0], geometry[1], geometry[2]])
                # self.visualizer.set_absorption(self.absorbance())
                landmarks = self.spatial_tracker.get_position_global(time.time_ns() * 1e-9)
                if landmarks is not None:
                    self.visualizer.set_landmarks((landmarks[..., :3] + head_position).cpu())
                self.visualizer.render(position=head_position.cpu().numpy(), rotation=rotation_matrix)
                self.view_sensor.update()
                if self.head_mounted_sensor is not self.view_sensor:
                    self.head_mounted_sensor.update()
        finally:
            self.visualizer.close()
            self.close()

    def verb_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


def main():
    """ULTRA-OPTIMIZED production main loop with maximum performance."""
    print("=" * 70)
    print("🚀 ULTRA-FAST TRACKER (MAXIMUM PERFORMANCE)")
    print("=" * 70)
    print("✓ BGR-native rendering (no RGB conversion)")
    print("✓ Pinned memory pool (fast CPU-GPU transfers)")
    print("✓ Async GPU operations (no blocking synchronize)")
    print("✓ Pre-allocated buffers (zero runtime allocation)")
    print("✓ 4096-entry color LUT (instant lookup)")
    print("=" * 70 + "\n")

    if not torch.cuda.is_available():
        print("❌ CUDA required!")
        return

    device = 'cuda'
    dtype = torch.float32
    print(f'device = {device}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    is_portrait_mode = False # Test depth map
    sensor = IMUSensing(mode="portrait" if is_portrait_mode else "landscape", dtype=dtype, device=device)
    visual_landmarks_pipe = VisualLandmarksTrackerPipe(
        view_sensor=sensor, head_mounted_sensor=sensor, is_portrait_mode=is_portrait_mode, dtype=dtype, device=device,
        visualize_projection=True
    )
    # visual_landmarks_pipe.run_visualizer_non_blocking()
    visual_landmarks_pipe.start_background_update_blocking()

if __name__ == "__main__":
    main()
