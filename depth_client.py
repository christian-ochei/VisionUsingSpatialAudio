from depth_check import DepthEstimator
import random
import string
import threading
import time
import cv2
import numpy as np
import torch
from process_depthmap import DepthFocus, LandmarksGenerator, AbsorptionEstimator, HighPerformanceLegendRenderer
from contextlib import nullcontext

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

# ============================================================================
# ULTRA-OPTIMIZED DEPTH MAP COLORIZER (MAXIMUM PERFORMANCE)
# ============================================================================
class DepthMapColorizer:
    """
    High-performance depth map colorization using pre-computed color LUT.

    Maps metric depth values to perceptually-distinct colors using a logarithmic
    scale from white (near) through the rainbow spectrum to black (far).

    Args:
        dtype: Torch dtype for computations (default: torch.float32)
        device: Device for computations (default: 'cuda')
        lut_size: Number of entries in color lookup table (default: 4096)
    """

    def __init__(self, dtype=torch.float32, device='cuda', lut_size=4096):
        self.dtype = dtype
        self.device = device
        self.lut_size = lut_size

        # Logarithmic depth range: 0.01m to ~50,000m
        self.log_min = -2.0
        self.log_max = 4.699
        self.log_range = self.log_max - self.log_min

        # Pre-compute color LUT in RGB order
        lut_distances = torch.logspace(
            self.log_min, self.log_max, self.lut_size,
            dtype=dtype, device=device
        )
        self.color_lut_rgb = self.build_color_lut_rgb(lut_distances)
        print(f"✓ DepthMapColorizer initialized: {self.lut_size} LUT entries, depth range: 0.01m - 50km")

    def build_color_lut_rgb(self, z_values):
        """
        Build color LUT in RGB order using perceptual color progression.

        Color mapping (normalized position t in [0, 1]):
        - t=0.00-0.10: White → Red (nearest)
        - t=0.10-0.25: Red → Orange
        - t=0.25-0.35: Orange → Yellow
        - t=0.35-0.50: Yellow → Green
        - t=0.50-0.65: Green → Cyan
        - t=0.65-0.75: Cyan → Blue
        - t=0.75-0.85: Blue → Violet
        - t=0.85-1.00: Violet → Black (farthest)
        """
        # Normalize depths to [0, 1] range using logarithmic scaling
        t = (torch.log10(torch.clamp(z_values, 0.01, 50000.0)) - self.log_min) / self.log_range
        t = torch.clamp(t, 0.0, 1.0)

        # Initialize RGB colors
        colors_rgb = torch.zeros((len(z_values), 3), dtype=self.dtype, device=self.device)

        # White → Red
        mask = t <= 0.10
        local_t = (t[mask] - 0.0) / 0.10
        colors_rgb[mask, 0] = 255
        colors_rgb[mask, 1] = (255 * (1 - local_t))
        colors_rgb[mask, 2] = (255 * (1 - local_t))

        # Red → Orange
        mask = (t > 0.10) & (t <= 0.25)
        local_t = (t[mask] - 0.10) / 0.15
        colors_rgb[mask, 0] = 255
        colors_rgb[mask, 1] = (127 * local_t)
        colors_rgb[mask, 2] = 0

        # Orange → Yellow
        mask = (t > 0.25) & (t <= 0.35)
        local_t = (t[mask] - 0.25) / 0.10
        colors_rgb[mask, 0] = 255
        colors_rgb[mask, 1] = (127 + 128 * local_t)
        colors_rgb[mask, 2] = 0

        # Yellow → Green
        mask = (t > 0.35) & (t <= 0.50)
        local_t = (t[mask] - 0.35) / 0.15
        colors_rgb[mask, 0] = (255 * (1 - local_t))
        colors_rgb[mask, 1] = 255
        colors_rgb[mask, 2] = 0

        # Green → Cyan
        mask = (t > 0.50) & (t <= 0.65)
        local_t = (t[mask] - 0.50) / 0.15
        colors_rgb[mask, 0] = 0
        colors_rgb[mask, 1] = 255
        colors_rgb[mask, 2] = (255 * local_t)

        # Cyan → Blue
        mask = (t > 0.65) & (t <= 0.75)
        local_t = (t[mask] - 0.65) / 0.10
        colors_rgb[mask, 0] = 0
        colors_rgb[mask, 1] = (255 * (1 - local_t))
        colors_rgb[mask, 2] = 255

        # Blue → Violet
        mask = (t > 0.75) & (t <= 0.85)
        local_t = (t[mask] - 0.75) / 0.10
        colors_rgb[mask, 0] = (127 * local_t)
        colors_rgb[mask, 1] = 0
        colors_rgb[mask, 2] = 255

        # Violet → Black
        mask = t > 0.85
        local_t = (t[mask] - 0.85) / 0.15
        colors_rgb[mask, 0] = (127 * (1 - local_t))
        colors_rgb[mask, 1] = 0
        colors_rgb[mask, 2] = (255 * (1 - local_t))
        return colors_rgb

    @torch.no_grad()
    def colorize(self, depth_map):
        """
        Colorize a depth map using the pre-computed color LUT.

        Args:
            depth_map: Torch tensor of shape (H, W) containing metric depth values

        Returns:
            Torch tensor of shape (H, W, 3) with RGB colors (uint8)
        """
        # Validate input
        if depth_map.dim() != 2:
            raise ValueError(f"Expected 2D depth map, got shape {depth_map.shape}")

        H, W = depth_map.shape

        # Move to correct device if needed
        if depth_map.device != self.device:
            depth_map = depth_map.to(self.device)

        # Flatten for vectorized lookup
        depth_flat = depth_map.reshape(-1)

        # Clamp and compute log-scaled indices
        z_clamped = torch.clamp(depth_flat, 0.01, 50000.0)
        log_z = torch.log10(z_clamped)

        # Map to LUT indices
        t = (log_z - self.log_min) / self.log_range
        indices = (t * (self.lut_size - 1)).long()
        indices = torch.clamp(indices, 0, self.lut_size - 1)

        # Lookup colors and reshape
        colors_flat = self.color_lut_rgb[indices]
        colors_rgb = colors_flat.reshape(H, W, 3)
        # Convert to uint8
        return colors_rgb[..., [2, 1, 0]]

class MultiThreadedZSampler:
    """Background depth estimation with FIXED regeneration scheduling."""

    def __init__(
        self, H, W, K, num_landmarks, tracker, parent, dtype=torch.float32, device='cuda', visualize_depth_map=False,
        visualize_amps=False, dsp=None, should_adjust_room_parameters=False, model_execution_lock=None, verbose=True
    ):
        self.verbose = verbose
        self.last_time = time.time()
        self.tracker = tracker
        self.dsp = dsp
        self.H = H
        self.W = W
        self.K = K
        self.num_landmarks = num_landmarks
        if model_execution_lock is None:
            model_execution_lock = nullcontext()
        self.model_execution_lock = model_execution_lock
        self.should_adjust_room_parameters = should_adjust_room_parameters
        self.parent = parent
        self.dtype = dtype
        self.visualize_depth_map = visualize_depth_map
        self.visualize_amps = visualize_amps
        self.device = device
        self.running = False
        self.depth_estimator = DepthEstimator(
            H=H, W=W, dtype=torch.float32, device='dml'
        )
        self.absorption_estimator = AbsorptionEstimator(H=H, W=W, dtype=torch.float32, device='cpu')
        self.landmarks_gen = LandmarksGenerator(H=H, W=W, dtype=torch.float32, device='cpu')
        self.work_queue = []
        self.work_lock = threading.Lock()
        self.work_available = threading.Event()
        self.K = K
        self.z_map = None
        self.z_lock = threading.Lock()
        self.last_regeneration_frame = -999
        self.regeneration_interval = 5
        self.integration_period = 3
        self.frame_count = 0
        self.depth_heuristics_echo_db = -1.0
        self.current_thread_id = makeid(20)
        # We start with a 10 meter ^ 3 bounding box for room acoustics.
        # When rotation is identity, the vector faces "POSITIVE Z"
        # W, H, D, -W, -H, -D
        self.room_bounding_box = torch.full((6, ), dtype=dtype, device='cpu', fill_value=10.0)
        # Wall normals: -W, -H, -D, +W, +H, +D
        self.wall_normals = torch.tensor([
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=dtype, device='cpu')
        self.z_vector = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device='cpu')
        self.depth_colorize =  DepthMapColorizer(dtype=dtype, device='cpu')
        self.legend_renderer = HighPerformanceLegendRenderer(H=H, W=W, device='cpu')
        self.zeros_depth = torch.ones(
            (self.H, self.W), dtype=torch.float32, device='cpu'
        )
        self.zero_head = torch.zeros([1, 3], dtype=self.dtype, device='cpu')
        self.geom_buffers = {
            "w": torch.zeros((), dtype=dtype, pin_memory=True),
            "h": torch.zeros((), dtype=dtype, pin_memory=True),
            "d": torch.zeros((), dtype=dtype, pin_memory=True),
            "head": torch.zeros(3, dtype=dtype, pin_memory=True),
        }
        self.host_device = 'cuda'
        self._room_head_geometry = self.make_room_head_geometry()
        self.sampler_thread = threading.Thread(target=self.sampler_event_loop, daemon=True)
        self.sampler_thread.start()
        print("🧵 Sampler thread started\n")

    def room_head_geometry(self):
        return self._room_head_geometry

    def make_room_head_geometry(self):
        # room_bounding_box is CPU pinned memory
        bbox = self.room_bounding_box  # [6]

        # Total room dimensions (CPU math is trivial cost)
        w = bbox[0] + bbox[3]
        h = bbox[1] + bbox[4]
        d = bbox[2] + bbox[5]

        # Head offset (CPU)
        head_x = (bbox[0] - bbox[3]) * 0.5
        head_y = (bbox[1] - bbox[4]) * 0.5
        head_z = (bbox[2] - bbox[5]) * 0.5
        head_position = torch.stack([head_x, head_y, head_z])

        # copy CPU→CPU pinned buffers
        self.geom_buffers["w"].copy_(w)
        self.geom_buffers["h"].copy_(h)
        self.geom_buffers["d"].copy_(d)
        self.geom_buffers["head"].copy_(head_position)

        # NON-BLOCKING CPU→GPU TRANSFER
        w_gpu = self.geom_buffers["w"].to(self.host_device, non_blocking=True)
        h_gpu = self.geom_buffers["h"].to(self.host_device, non_blocking=True)
        d_gpu = self.geom_buffers["d"].to(self.host_device, non_blocking=True)
        head_gpu = self.geom_buffers["head"].to(self.host_device, non_blocking=True)
        return w_gpu, h_gpu, d_gpu, head_gpu

    @torch.inference_mode()
    def sampler_event_loop(self):
        """Event-driven loop."""
        self.running = True
        while self.running:
            time.sleep(0.00001)
            self.work_available.wait(timeout=0.1)
            if not self.running:
                break
            with self.work_lock:
                if not self.work_queue:
                    self.work_available.clear()
                    continue
                work_item = self.work_queue.pop(0)
                if not self.work_queue:
                    self.work_available.clear()
            self._process_frame(work_item)
            self.current_thread_id = makeid(20)
        if self.visualize_depth_map or self.visualize_amps:
            cv2.destroyAllWindows()

    # def queue_work(
    #     self, frame, landmarks, alive_mask, frame_timestamp_sec, should_queue_regeneration, rotation_matrix
    # ):
    #     """Queue work with FIXED regeneration logic."""
    #     if should_queue_regeneration:
    #         self.last_regeneration_frame = frame_timestamp_sec
    #         print(f"   ⏰ Regeneration scheduled for frame {frame_timestamp_sec}")
    #
    #     work_item = {
    #         'frame': frame.clone(),
    #         'landmarks': landmarks.clone(),
    #         'alive_mask': alive_mask.clone(),
    #         'frame_timestamp_sec': frame_timestamp_sec,
    #         'should_regenerate': should_queue_regeneration,
    #         'rotation_matrix': rotation_matrix,
    #     }
    #     with self.work_lock:
    #         if should_queue_regeneration:
    #             self.work_queue.append(work_item)
    #         else:
    #             has_regen = any(item['should_regenerate'] for item in self.work_queue)
    #             if has_regen:
    #                 self.work_queue = [item for item in self.work_queue if item['should_regenerate']] + [work_item]
    #             else:
    #                 self.work_queue = [work_item]
    #         self.work_available.set()

    def queue_work(
        self, frame, landmarks, alive_mask, frame_timestamp_sec, should_queue_regeneration, rotation_matrix
    ):
        """
        Queue work with Sticky Regeneration Logic.
        1. Always keeps Queue Size = 1 (Latest Frame).
        2. If a pending frame was meant to regenerate, the NEW frame inherits that duty.
        """
        # Update timestamp for logging/tracking
        if should_queue_regeneration:
            self.last_regeneration_frame = frame_timestamp_sec
            self.verb_print(f"   ⏰ Regeneration requested at {frame_timestamp_sec}")

        with self.work_lock:
            # Check if there is a pending job that we are about to overwrite
            # which had a regeneration flag set.
            if self.work_queue:
                existing_job = self.work_queue[0]
                # OR logic: If the new frame needs regen OR the dropped frame needed regen
                should_queue_regeneration = should_queue_regeneration or existing_job['should_regenerate']

            # Create the work item with the (possibly inherited) flag
            work_item = {
                'frame': frame.clone(),  # Clone is essential for async safety
                'landmarks': landmarks.clone(),
                'alive_mask': alive_mask.clone(),
                'frame_timestamp_sec': frame_timestamp_sec,
                'should_regenerate': should_queue_regeneration,
                'rotation_matrix': rotation_matrix,
            }

            # ALWAYS overwrite. Zero lag.
            self.work_queue = [work_item]
            self.work_available.set()

    def get_depth_data(self):
        """Get depth data."""
        with self.z_lock:
            return self.z_map

    def close(self):
        """Shutdown."""
        self.running = False
        self.work_available.set()
        cv2.destroyAllWindows()

    @torch.inference_mode()
    def _process_frame(self, work_item):
        """Process frame and immediately sample z-values for candidates."""
        frame = work_item['frame']
        landmarks = work_item['landmarks']
        alive_mask = work_item['alive_mask']
        frame_timestamp_sec = work_item['frame_timestamp_sec']
        should_regenerate = work_item['should_regenerate']
        rotation_matrix = work_item['rotation_matrix']
        frame_tensor_chw = frame.permute(2, 0, 1)[None]
        # Inputs are already FP16 via buffer, no need to cast again
        depth_map = self.depth_estimator.estimate_depth(
            frame_tensor_chw
            # This is just faster for float32 on DML
        )
        # Never use None blocking
        if self.visualize_depth_map:
            colored_depth_map = self.depth_colorize.colorize(depth_map).clamp_(0, 255)
            depth_map_np = self.legend_renderer.blend(colored_depth_map).numpy().astype(np.uint8)
            cv2.imshow("depth_map_np", depth_map_np)
            cv2.imshow("frame", frame.numpy().astype(np.uint8))
            cv2.waitKey(4)

        # This tells us the room distance parameters
        if self.should_adjust_room_parameters and self.frame_count % 5 == 0:
            x = depth_map.reshape(-1)[::16]
            far_distance = torch.dot(x, x).div_(x.sum())
            vector = self.z_vector
            if rotation_matrix is not None:
                vector = rotation_matrix @ vector

            # Get raw dot products (projections onto each wall normal)
            dot_products = self.wall_normals @ vector
            weights_raw_ = torch.clamp_(dot_products, min=0)  # Only facing walls

            # Target: the perpendicular distance to each wall
            # Key: far_distance * dot_product gives the correct wall distance
            target_wall_distances = far_distance * weights_raw_ * 2.0

            # Normalized weights for consistent learning rate across all visible walls
            update_weights = weights_raw_.div_(weights_raw_.sum().add_(1e-8))

            # Smooth exponential moving average
            learning_rate = 0.2
            self.room_bounding_box = (
                learning_rate * update_weights * target_wall_distances +
                (1.0 - learning_rate * update_weights) * self.room_bounding_box
            )
            self._room_head_geometry = self.make_room_head_geometry()
        long_landmarks = landmarks.long()
        x_coords = long_landmarks[:, 0].clamp_(0, depth_map.shape[1] - 1)
        y_coords = long_landmarks[:, 1].clamp_(0, depth_map.shape[0] - 1)
        zs = depth_map[y_coords, x_coords]
        if self.visualize_amps:
            amps_vis = DepthFocus.depth_to_amplitude(
                depth_map
            )
            cv2.imshow("amps", ((amps_vis * 255).clamp_(0, 255)).numpy().astype(np.uint8))
            cv2.waitKey(4)

        if should_regenerate:
            self.verb_print(f"   🔍 Regenerating @ frame {frame_timestamp_sec}")
            existing = landmarks[alive_mask] if alive_mask.sum() > 0 else landmarks[:0]
            candidate_amps_dense = DepthFocus.depth_to_amplitude(
                depth_map
            )
            # landmarks_yx = self.landmarks_gen.generate_landmarks(
            #     candidate_amps_dense=candidate_amps_dense, frame=frame,
            #     target_points=self.tracker.num_landmarks,
            #     existing_landmarks=existing
            # )[:self.num_landmarks]

            # landmarks_xy = torch.flip(landmarks_yx, dims=[1])
            # The issue most likely stemed from the fact landmarks is genrated at regions
            # we might move away from and not at regions we move to
            landmarks_xy = self.tracker.generate_grid_points()
            long_landmarks_xy = landmarks_xy.long()
            # Do not clamp. This should never go out of bounds, unless there is a clear bug
            x_coords = long_landmarks_xy[:, 0]
            y_coords = long_landmarks_xy[:, 1]
            candidate_zs = depth_map[y_coords, x_coords]
            candidate_amps = candidate_amps_dense[y_coords, x_coords]

            # Record that this stream is done writing
            self.verb_print(f"   ✓ Generated candidates with pre-sampled z-values")
            with self.parent.tracker_lock:
                self.integration_period = 3
                self.tracker.set_candidate_points(
                    points=landmarks_xy,
                    z_values=candidate_zs,
                    amp_values_norm=candidate_amps, # / amp_values_norm,
                    frame_tensor=frame
                )

        with self.z_lock:
            if should_regenerate:
                self.z_map = None
                self.work_queue = []
                self.work_available.clear()
            else:
                self.z_map = dict(
                    zs=zs,  # .to(dtype=self.dtype),
                    frame_timestamp_sec=frame_timestamp_sec,
                )
            if not should_regenerate:
                self.integration_period -= 1
                if self.integration_period < 0:
                    self.integration_period = 0

        t = time.time()
        if self.frame_count % 20 == 0:
            self.verb_print(f'FPS DEPTH-ESTIMATOR: {1 / (t - self.last_time + 0.0000001)}')
        self.last_time = t
        self.frame_count += 1

    @torch.inference_mode()
    def _process_frame(self, work_item):
        """Process frame and immediately sample z-values for candidates."""
        frame = work_item['frame']
        landmarks = work_item['landmarks']
        alive_mask = work_item['alive_mask']
        frame_timestamp_sec = work_item['frame_timestamp_sec']
        should_regenerate = work_item['should_regenerate']
        rotation_matrix = work_item['rotation_matrix']

        frame_tensor_chw = frame.permute(2, 0, 1)[None]

        # 1. Estimate Depth
        depth_map = self.depth_estimator.estimate_depth(
            frame_tensor_chw
        )

        if self.visualize_depth_map:
            colored_depth_map = self.depth_colorize.colorize(depth_map).clamp_(0, 255)
            depth_map_np = self.legend_renderer.blend(colored_depth_map).numpy().astype(np.uint8)
            cv2.imshow("depth_map_np", depth_map_np)
            cv2.imshow("frame", frame.numpy().astype(np.uint8))
            cv2.waitKey(4)

        # 2. Update Room Geometry (Visual Odometry)
        if self.should_adjust_room_parameters and self.frame_count % 5 == 0:
            x = depth_map.reshape(-1)[::16]
            far_distance = torch.dot(x, x).div_(x.sum())
            vector = self.z_vector
            if rotation_matrix is not None:
                vector = rotation_matrix @ vector

            dot_products = self.wall_normals @ vector
            weights_raw_ = torch.clamp_(dot_products, min=0)
            target_wall_distances = far_distance * weights_raw_ * 2.0
            update_weights = weights_raw_.div_(weights_raw_.sum().add_(1e-8))

            learning_rate = 0.2
            self.room_bounding_box = (
                    learning_rate * update_weights * target_wall_distances +
                    (1.0 - learning_rate * update_weights) * self.room_bounding_box
            )
            self._room_head_geometry = self.make_room_head_geometry()

        # 3. CRITICAL: Calculate Zs for ALL current landmarks (0 to N)
        # We do this EVERY frame, regardless of regeneration status.
        long_landmarks = landmarks.long()
        x_coords = long_landmarks[:, 0].clamp_(0, depth_map.shape[1] - 1)
        y_coords = long_landmarks[:, 1].clamp_(0, depth_map.shape[0] - 1)

        # This gives us a vector of Z values for every landmark index [0..N]
        zs = depth_map[y_coords, x_coords]

        if self.visualize_amps:
            amps_vis = DepthFocus.depth_to_amplitude(depth_map)
            cv2.imshow("amps", ((amps_vis * 255).clamp_(0, 255)).numpy().astype(np.uint8))
            cv2.waitKey(4)

        # 4. Handle Regeneration (New Candidates)
        if should_regenerate:
            self.verb_print(f"   🔍 Regenerating @ frame {frame_timestamp_sec}")
            existing = landmarks[alive_mask] if alive_mask.sum() > 0 else landmarks[:0]
            candidate_amps_dense = DepthFocus.depth_to_amplitude(depth_map)
            # We use uniform landmarks distribution for now
            # landmarks_yx = self.landmarks_gen.generate_landmarks(
            #     candidate_amps_dense=candidate_amps_dense, frame=frame,
            #     target_points=self.tracker.num_landmarks,
            #     existing_landmarks=existing
            # )[:self.num_landmarks]
            #
            # landmarks_xy = torch.flip(landmarks_yx, dims=[1])
            landmarks_xy = self.tracker.generate_grid_points()
            long_landmarks_xy = landmarks_xy.long()
            x_coords_cand = long_landmarks_xy[:, 0]
            y_coords_cand = long_landmarks_xy[:, 1]
            candidate_zs = depth_map[y_coords_cand, x_coords_cand]
            candidate_amps = candidate_amps_dense[y_coords_cand, x_coords_cand]

            self.verb_print(f"   ✓ Generated candidates with pre-sampled z-values")
            with self.parent.tracker_lock:
                self.integration_period = 3
                self.tracker.set_candidate_points(
                    points=landmarks_xy,
                    z_values=candidate_zs,
                    amp_values_norm=candidate_amps,
                    frame_tensor=frame
                )

        # 5. PUBLISH Z DATA (The Fix)
        with self.z_lock:
            # We ALWAYS publish the zs we calculated in Step 3.
            # Even if we regenerate, the tracker needs updated depths for the points
            # that are currently alive while it waits for the new ones to integrate.
            self.z_map = dict(
                zs=zs,
                frame_timestamp_sec=frame_timestamp_sec,
            )

            if should_regenerate:
                # We clear the queue to prevent backing up, but we DO NOT wipe z_map.
                self.work_queue = []
                self.work_available.clear()
            else:
                self.integration_period -= 1
                if self.integration_period < 0:
                    self.integration_period = 0

        t = time.time()
        if self.frame_count % 20 == 0:
            self.verb_print(f'FPS DEPTH-ESTIMATOR: {1 / (t - self.last_time + 0.0000001)}')
        self.last_time = t
        self.frame_count += 1

    def verb_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)