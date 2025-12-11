import cv2
import torch
import torch.nn.functional as functional
import numpy as np

# Define the base directory for model caching
MODEL_DIR = "models"

def rec_check_nan(items):
    if items is None:
        return
    if isinstance(items, (tuple, list)):
        for item in items:
            return rec_check_nan(item)
    elif isinstance(items, (bool, int, str, float)):
        return
    assert not torch.isnan(items).any()

DEBUG_NANS = False
def stable_checker(func):
    if not DEBUG_NANS:
        return func
    def inner_func(*args, **kwargs):
        out = func(*args, **kwargs)
        rec_check_nan(out)
        return out
    return inner_func

class CameraProjection:
    """
    Camera projection utilities with in-place operations.

    Args:
        num_landmarks: Number of landmarks/points to process
        H: Image height
        W: Image width
        K: Camera intrinsic matrix (3x3)
        dtype: Data type for buffers (default: torch.float32)
        device: Device for buffers (default: 'cuda')
    """

    def __init__(self, num_landmarks, H, W, K, dtype=torch.float32, device='cuda'):
        self.num_landmarks = num_landmarks
        self.H = H
        self.W = W
        self.dtype = dtype
        self.device = device

        # Extract and store intrinsic parameters
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

        # Pre-allocate intermediate computation buffers
        self.temp_buffer = torch.zeros(
            (num_landmarks,),
            dtype=dtype,
            device=device
        )

    def pixels_to_camera_(self, points):
        """
        Convert pixel coordinates to camera frame (in-place version).
        Modifies the input tensor directly.

        Args:
            points: Input tensor of shape [num_landmarks, 4] with (x, y, Z, amp)
                   Modified in-place to contain (X, Y, Z, amp)

        Returns:
            points (modified in-place)
        """
        # Read input columns (these reads happen before writes)
        x = points[:, 0]
        y = points[:, 1]
        Z = points[:, 2]

        # Compute X = (cx - x) * Z / fx (in-place into column 0)
        torch.sub(self.cx, x, out=self.temp_buffer)
        torch.mul(self.temp_buffer, Z, out=self.temp_buffer)
        torch.div(self.temp_buffer, self.fx, out=points[:, 0])

        # Compute Y = (y - cy) * Z / fy (in-place into column 1)
        torch.sub(y, self.cy, out=self.temp_buffer)
        torch.mul(self.temp_buffer, Z, out=self.temp_buffer)
        torch.div(self.temp_buffer, self.fy, out=points[:, 1])

        # Columns 2 (Z) and 3 (amp) remain unchanged

        return points

    def camera_to_pixels_(self, point_in_camera_frame, include_c=True):
        """
        Convert camera frame coordinates to pixel coordinates (in-place version).
        Modifies the input tensor directly.

        Args:
            point_in_camera_frame: Input tensor of shape [num_landmarks, 4] with (x, y, z, amp)
                                  Modified in-place to contain (x_pix, y_pix, z, amp)
            include_c: Whether to include principal point offset

        Returns:
            point_in_camera_frame (modified in-place)
        """
        # Read input columns (these reads happen before writes)
        x = point_in_camera_frame[:, 0]
        y = point_in_camera_frame[:, 1]
        z = point_in_camera_frame[:, 2]

        if include_c:
            # Compute x_pix = cx - (x * fx / z) (in-place into column 0)
            torch.mul(x, self.fx, out=self.temp_buffer)
            torch.div(self.temp_buffer, z, out=self.temp_buffer)
            torch.sub(self.cx, self.temp_buffer, out=point_in_camera_frame[:, 0])

            # Compute y_pix = (y * fy / z) + cy (in-place into column 1)
            torch.mul(y, self.fy, out=self.temp_buffer)
            torch.div(self.temp_buffer, z, out=self.temp_buffer)
            torch.add(self.temp_buffer, self.cy, out=point_in_camera_frame[:, 1])
        else:
            # Compute x_pix = -(x * fx / z) (in-place into column 0)
            torch.mul(x, self.fx, out=self.temp_buffer)
            torch.div(self.temp_buffer, z, out=self.temp_buffer)
            torch.neg(self.temp_buffer, out=point_in_camera_frame[:, 0])

            # Compute y_pix = (y * fy / z) (in-place into column 1)
            torch.mul(y, self.fy, out=self.temp_buffer)
            torch.div(self.temp_buffer, z, out=point_in_camera_frame[:, 1])

        # Columns 2 (z) and 3 (amp) remain unchanged
        return point_in_camera_frame

@stable_checker
def calculate_peak_gain_estimate(
    head_position: torch.Tensor,
    speaker_position: torch.Tensor,
    absorption_db: float,
    # Engine parameters for an accurate "soft" calculation:
    min_distance: float = 0.13,  # 0.05 mic + 0.08 speaker
    late_reverb_mix: float = 0.9
) -> torch.Tensor:
    distance = torch.linalg.norm(speaker_position - head_position, dim=-1)
    distance[distance < min_distance] = min_distance
    direct_gain = 1.0 / distance
    liveness_factor = 10.0 ** (absorption_db / 20.0)
    total_reverb_multiplier = liveness_factor * (1.0 + late_reverb_mix)
    total_peak_gain = direct_gain * (1.0 + total_reverb_multiplier)
    return total_peak_gain

class AbsorptionEstimator:
    def __init__(self, H, W, dtype=torch.float32, device='cuda'):
        self.kernel = torch.ones(1, 1, 3, 3, device=device, dtype=dtype) / 9.0
        self.device = device
        self.dtype = dtype

        # Static Scalar Constants (on GPU to avoid CPU transfers during math)
        self.eps = torch.tensor(1e-6, dtype=dtype, device=device)
        self.neg_three = torch.tensor(-3.0, dtype=dtype, device=device)
        self.zero = torch.tensor(0.0, dtype=dtype, device=device)
        self.one = torch.tensor(1.0, dtype=dtype, device=device)
        self.nan = torch.tensor(float('nan'), dtype=dtype, device=device)

        # Config constants
        self.pt_15 = torch.tensor(0.15, dtype=dtype, device=device)
        self.pt_5 = torch.tensor(0.5, dtype=dtype, device=device)
        self.pt_3 = torch.tensor(0.3, dtype=dtype, device=device)
        self.pt_2 = torch.tensor(0.2, dtype=dtype, device=device)
        self.subbed_buf = torch.empty((H//2, W//2, 3), dtype=dtype, device=device)

        # Memory Arena
        self.h_w = (0, 0)
        self.buf_main = None  # Main depth buffer
        self.buf_scratch_1 = None  # Floating point scratch (diff(s), vars)
        self.buf_scratch_2 = None  # Floating point scratch (min(s), sq_sums)
        self.buf_mask_1 = None  # Boolean Mask
        self.buf_mask_2 = None  # Boolean Mask
        self.blur_kernels = {}

    def _ensure_arena(self, h, w):
        """Allocates static buffers only when resolution changes."""
        if self.h_w == (h, w):
            return

        # Allocate full size buffers
        self.buf_main = torch.empty((h, w), dtype=self.dtype, device=self.device)
        self.buf_scratch_1 = torch.empty((h, w), dtype=self.dtype, device=self.device)
        self.buf_scratch_2 = torch.empty((h, w), dtype=self.dtype, device=self.device)

        # Byte buffers for masking
        self.buf_mask_1 = torch.empty((h, w), dtype=torch.bool, device=self.device)
        self.buf_mask_2 = torch.empty((h, w), dtype=torch.bool, device=self.device)
        self.h_w = (h, w)

    def _get_blur_kernel(self, k, sigma):
        key = (k, sigma)
        if key in self.blur_kernels:
            return self.blur_kernels[key]

        coords = torch.arange(k, device=self.device, dtype=self.dtype) - (k - 1) / 2.0
        g = torch.exp(-0.5 * (coords / (sigma + self.eps)) ** 2)
        g = g / (g.sum() + self.eps)

        k_h = g.view(1, 1, 1, k)
        k_v = g.view(1, 1, k, 1)
        self.blur_kernels[key] = (k_h, k_v)
        return k_h, k_v

    @stable_checker
    @torch.inference_mode()
    def estimate_db_from_noisy_depth(
        self, depth: torch.Tensor,
        *,
        max_depth_ignore: float = 1000.0,
        median_kernel: int = 3,
        apply_median: bool = True,
        gaussian_kernel: int = 5,
        gaussian_sigma: float = 1.0,
        mad_clip_k: float = 3.0,
        discontinuity_threshold: float = 0.15,
        min_discontinuity_density: float = 0.05,
        use_depth_variance: bool = True,
        depth_var_scale: float = 0.25
    ):
        H, W = depth.shape
        self._ensure_arena(H, W)

        # 1. Load Input
        self.buf_main.copy_(depth)

        # 2. Median Filter
        # We use in-place copy back to buf_main to keep memory static
        if apply_median and median_kernel >= 3:
            pad = median_kernel // 2
            padded = functional.pad(self.buf_main.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
            patches = padded.unfold(2, median_kernel, 1).unfold(3, median_kernel, 1)
            # median() allocates a small reduction tensor, unavoidable but fast
            med_res = patches.contiguous().view(1, 1, H, W, -1).median(dim=-1).values.squeeze()
            self.buf_main.copy_(med_res)

        # 3. Gaussian Blur
        if gaussian_kernel > 1:
            k_h, k_v = self._get_blur_kernel(gaussian_kernel, gaussian_sigma)
            # Conv-2d allocates result, we copy immediately to buffer and drop reference
            temp = functional.pad(self.buf_main.view(1, 1, H, W), (gaussian_kernel // 2, gaussian_kernel // 2, 0, 0),
                         mode='reflect')
            temp = functional.conv2d(temp, k_h)
            temp = functional.pad(temp, (0, 0, gaussian_kernel // 2, gaussian_kernel // 2), mode='reflect')
            temp = functional.conv2d(temp, k_v)
            self.buf_main.copy_(temp.squeeze())

        # 4. MAD Outlier Removal (No Stalls, No Allocations)
        # --------------------------------------------------
        # Generate Valid Mask: (val > 0) & (val <= max)
        torch.gt(self.buf_main, 0.0, out=self.buf_mask_1)
        torch.le(self.buf_main, max_depth_ignore, out=self.buf_mask_2)
        self.buf_mask_1.logical_and_(self.buf_mask_2)

        # REMOVED: if not self.buf_mask_1.any(): return ... (STALL)
        # Instead, we compute blindly. If mask is empty, nan-median returns NaN.
        # We handle NaNs at the very end.

        # Prepare Statistics Buffer (Valid -> Value, Invalid -> NaN)
        self.buf_scratch_1.copy_(self.buf_main)
        torch.logical_not(self.buf_mask_1, out=self.buf_mask_2)
        self.buf_scratch_1.masked_fill_(self.buf_mask_2, self.nan)

        # Compute Median & MAD
        med = torch.nanmedian(self.buf_scratch_1)  # Scalar tensor

        # Reuse Scratch for |Val - Med|
        # sub_ and abs_ are strictly in-place
        self.buf_scratch_1.sub_(med).abs_()

        mad = torch.nanmedian(self.buf_scratch_1)  # Scalar tensor
        mad = mad.maximum(self.eps)  # Safety for div/0 later

        # Compute Clamp Bounds (Scalars)
        lower = med - (mad * mad_clip_k)
        upper = med + (mad * mad_clip_k)

        # Apply Clamp to Main Buffer
        self.buf_main.clamp_(min=lower, max=upper)

        # Re-verify Mask (Clamp might have pushed NaNs or weirdness, though unlikely)
        # We ensure positive depth
        self.buf_main = self.buf_main.maximum(self.eps)
        torch.gt(self.buf_main, self.eps, out=self.buf_mask_1)
        torch.le(self.buf_main, max_depth_ignore, out=self.buf_mask_2)
        self.buf_mask_1.logical_and_(self.buf_mask_2)

        # 5. Scale-Invariant Metrics (Accumulation)
        # -----------------------------------------
        # We accumulate into 0-dim tensors to avoid creating large edge lists
        total_valid_edges = self.zero.clone()
        total_variation_sum = self.zero.clone()
        total_discontinuities = self.zero.clone()

        # === Vertical Pass ===
        curr_v = self.buf_main[:-1, :]
        next_v = self.buf_main[1:, :]
        mask_curr_v = self.buf_mask_1[:-1, :]
        mask_next_v = self.buf_mask_1[1:, :]

        # Reuse Scratch Buffers (Views)
        s_diff_v = self.buf_scratch_1[:-1, :]  # Will hold rel_change
        s_min_v = self.buf_scratch_2[:-1, :]  # Will hold denominators
        s_mask_v = self.buf_mask_2[:-1, :]  # Will hold valid pair mask

        # Valid Pairs Logic
        torch.logical_and(mask_curr_v, mask_next_v, out=s_mask_v)

        # Math: |next - curr| / (min + eps)
        torch.minimum(curr_v, next_v, out=s_min_v)
        s_min_v.add_(self.eps)

        s_diff_v.copy_(next_v)
        s_diff_v.sub_(curr_v).abs_()
        s_diff_v.div_(s_min_v)

        # Filter Invalid (set to 0.0 so sum isn't affected)
        # logical_not is in-place on byte buffer if we use the right output
        # But masked_fill requires a boolean mask.
        s_diff_v.masked_fill_(~s_mask_v, 0.0)

        # Accumulate
        total_valid_edges.add_(s_mask_v.count_nonzero())
        total_variation_sum.add_(s_diff_v.sum())

        # Discontinuities (Reuse s_mask_v for threshold check)
        # We strictly check > threshold. Zeros (masked out) won't trigger.
        torch.gt(s_diff_v, discontinuity_threshold, out=s_mask_v)
        total_discontinuities.add_(s_mask_v.count_nonzero())

        # === Horizontal Pass ===
        curr_h = self.buf_main[:, :-1]
        next_h = self.buf_main[:, 1:]
        mask_curr_h = self.buf_mask_1[:, :-1]
        mask_next_h = self.buf_mask_1[:, 1:]

        s_diff_h = self.buf_scratch_1[:, :-1]
        s_min_h = self.buf_scratch_2[:, :-1]
        s_mask_h = self.buf_mask_2[:, :-1]

        torch.logical_and(mask_curr_h, mask_next_h, out=s_mask_h)
        torch.minimum(curr_h, next_h, out=s_min_h)
        s_min_h.add_(self.eps)

        s_diff_h.copy_(next_h)
        s_diff_h.sub_(curr_h).abs_()
        s_diff_h.div_(s_min_h)

        s_diff_h.masked_fill_(~s_mask_h, 0.0)

        total_valid_edges.add_(s_mask_h.count_nonzero())
        total_variation_sum.add_(s_diff_h.sum())

        torch.gt(s_diff_h, discontinuity_threshold, out=s_mask_h)
        total_discontinuities.add_(s_mask_h.count_nonzero())

        # 6. Global Variance (Zero Allocation)
        # ------------------------------------
        if use_depth_variance:
            # We need variance of valid pixels in buf_main.
            # Standard std() allocates. We use manual sum/sq_sum using scratch buffers.

            # 1. Copy valid data to scratch_1, invalid to 0 (for sum)
            self.buf_scratch_1.copy_(self.buf_main)
            self.buf_scratch_1.masked_fill_(~self.buf_mask_1, 0.0)

            # Count N
            N = self.buf_mask_1.count_nonzero()
            N_safe = N + self.eps

            # Sum X
            sum_x = self.buf_scratch_1.sum()
            mean_x = sum_x / N_safe

            # Sum (X - Mean)^2
            # We reuse scratch_1. logic: (valid_val - mean)^2.
            # We must be careful: (0 - mean)^2 would add error for invalid pixels.
            # Approach: Fill scratch_1 with valid data, others with mean_x (so diff is 0)
            self.buf_scratch_1.copy_(self.buf_main)
            self.buf_scratch_1.masked_fill_(~self.buf_mask_1, mean_x)

            # In-place ops
            self.buf_scratch_1.sub_(mean_x).pow_(2)

            sum_sq_diff = self.buf_scratch_1.sum()
            var_x = sum_sq_diff / N_safe
            std_x = var_x.sqrt()

            # Normalized Std: std / median
            # We need a clean median of the clamped data.
            # Fill scratch_2 with NaNs for nan-median
            self.buf_scratch_2.copy_(self.buf_main)
            self.buf_scratch_2.masked_fill_(~self.buf_mask_1, self.nan)
            curr_med = torch.nanmedian(self.buf_scratch_2)
            norm_std = std_x / (curr_med + self.eps)
            variance_score = torch.minimum(norm_std / depth_var_scale, self.one)
            variance_db = self.neg_three * variance_score
        else:
            variance_db = self.zero
        # 7. Final Combination & Output (Sanitized)
        # -----------------------------------------
        safe_edges = total_valid_edges + self.eps
        density = total_discontinuities / safe_edges
        variation = total_variation_sum / safe_edges
        # Discontinuity dB
        disc_ratio = density / min_discontinuity_density
        disc_ratio = disc_ratio.minimum(self.one)
        disc_db = self.neg_three * disc_ratio
        # Variation dB
        var_ratio = variation / self.pt_15
        var_ratio = var_ratio.minimum(self.one)
        var_db = self.neg_three * var_ratio
        # Weighted Sum
        final_db = (disc_db * self.pt_5) + (var_db * self.pt_3) + (variance_db * self.pt_2)
        # Clamp range
        final_db.clamp_(min=-3.0, max=0.0)
        # CRITICAL: Sanitize NaNs
        # If the scene was empty (all pixels invalid), division by eps or nan-median
        # might have propagated NaNs. nan_to_num_ fixes this in-place on the GPU
        # without a CPU branch.
        final_db.nan_to_num_(0.0)
        return final_db

    @stable_checker
    def estimate_echo_db(self, image: torch.Tensor):
        """
        Estimate room echo in dB based on visual heuristics.

        0 dB = maximum echo (empty, grayscale room with hard surfaces)
        -3 dB = minimum echo (cluttered, colorful room with soft materials)

        Heuristics:
        - Grayscale (low saturation) → more echo → closer to 0 dB
        - Colorful (high saturation) → less echo → closer to -3 dB
        - Smooth (low texture) → more echo → closer to 0 dB
        - Detailed/cluttered (high texture) → less echo → closer to -3 dB
        """

        # 1️⃣ Downscale for faster computation
        img_small = image[::2, ::2, :]  # (H//2, W//2, C)

        # 2️⃣ Normalize to [0, 1] per channel
        min_val_ = img_small.min(dim=0, keepdim=True)[0].min(dim=0, keepdim=True)[0]
        max_val_ = img_small.max(dim=0, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        img_norm = torch.sub(img_small, min_val_, out=self.subbed_buf).div_(max_val_.sub_(min_val_).add_(1e-6))

        # 3️⃣ High-order texture score (visual complexity indicating clutter)
        # Convert to grayscale for texture analysis
        gray = img_norm.mean(dim=2, keepdim=True)  # (H//2, W//2, 1)

        # Compute local variance as measure of high-order texture
        # Using a simple 3x3 neighborhood approximation
        gray_sq = gray ** 2

        # Mean and mean of squares in local neighborhoods
        gray_for_conv = gray.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        gray_sq_for_conv = gray_sq.permute(2, 0, 1).unsqueeze(0)
        local_mean = functional.conv2d(gray_for_conv, self.kernel, padding=1)
        local_mean_sq = functional.conv2d(gray_sq_for_conv, self.kernel, padding=1)
        local_variance = local_mean_sq - local_mean ** 2

        # Texture score: high variance = high detail = less echo
        texture_score = local_variance.mean()

        # Also include edge density as additional texture measure
        grad_x = torch.abs(gray[1:, :, :] - gray[:-1, :, :])
        grad_y = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
        edge_density = (grad_x.mean() + grad_y.mean()) / 2

        # Combine variance and edge density for robust texture metric
        combined_texture = texture_score * 0.6 + edge_density * 0.4

        # Scale to dB: high texture → -3 dB, low texture → 0 dB
        # Assuming texture values typically in [0, 0.1] range, adjust as needed
        texture_db = -3.0 * torch.clamp(combined_texture / 0.08, 0, 1)

        # 4️⃣ Saturation score (colorfulness indicating fabrics/decoration)
        max_rgb, _ = img_norm.max(dim=2)  # (H//2, W//2)
        min_rgb, _ = img_norm.min(dim=2)
        saturation = (max_rgb - min_rgb).mean()

        # Scale to dB: high saturation → -3 dB, low saturation → 0 dB
        # Saturation typically in [0, 1] range
        sat_db = -3.0 * torch.clamp(saturation / 0.6, 0, 1)

        # 5️⃣ Combine heuristics
        # Texture is more important for echo (physical stuff absorbs sound)
        # Weight texture higher than saturation
        echo_db = texture_db * 0.75 + sat_db * 0.25

        # Clamp to valid range
        return echo_db.clamp_(-3.0, 0.0)

class HighPerformanceLegendRenderer:
    """Pre-rendered BGR legend with GPU blending."""

    def __init__(self, H, W, dtype=torch.float32, device='cuda'):
        self.H = H
        self.W = W
        self.device = device
        self.dtype = dtype

        self.legend_overlay = torch.zeros((H, W, 3), dtype=torch.uint8, device=device)
        self.legend_mask = torch.zeros((H, W, 1), dtype=dtype, device=device)
        self._create_legend_bgr()
        print("✓ HighPerformanceLegendRenderer: BGR-native GPU blending")

    def _create_legend_bgr(self):
        """Create BGR legend directly."""
        legend_distances = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 50000.0]

        overlay_cpu = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        mask_cpu = np.zeros((self.H, self.W), dtype=np.float32)

        legend_x = self.W - 70
        legend_y = 10
        box_width = 25
        box_height = 20
        text_offset_x = 35

        z_tensor = torch.tensor(legend_distances, dtype=self.dtype, device=self.device)
        log_z = torch.log10(torch.clamp(z_tensor, 0.01, 50000.0))
        log_min = -2.0
        log_max = 4.699
        t = (log_z - log_min) / (log_max - log_min)
        t: torch.Tensor = torch.clamp(t, 0.0, 1.0)

        colors_rgb = torch.zeros((len(legend_distances), 3), dtype=torch.uint8, device=self.device)

        for i, ti in enumerate(t):
            ti: torch.Tensor
            ti_val = ti.item()
            if ti_val <= 0.10:
                local_t = (ti_val - 0.0) / 0.10
                colors_rgb[i] = torch.tensor([255, int(255 * (1 - local_t)), int(255 * (1 - local_t))],
                                             device=self.device)
            elif ti_val <= 0.25:
                local_t = (ti_val - 0.10) / 0.15
                colors_rgb[i] = torch.tensor([255, int(127 * local_t), 0], device=self.device)
            elif ti_val <= 0.35:
                local_t = (ti_val - 0.25) / 0.10
                colors_rgb[i] = torch.tensor([255, int(127 + 128 * local_t), 0], device=self.device)
            elif ti_val <= 0.50:
                local_t = (ti_val - 0.35) / 0.15
                colors_rgb[i] = torch.tensor([int(255 * (1 - local_t)), 255, 0], device=self.device)
            elif ti_val <= 0.65:
                local_t = (ti_val - 0.50) / 0.15
                colors_rgb[i] = torch.tensor([0, 255, int(255 * local_t)], device=self.device)
            elif ti_val <= 0.75:
                local_t = (ti_val - 0.65) / 0.10
                colors_rgb[i] = torch.tensor([0, int(255 * (1 - local_t)), 255], device=self.device)
            elif ti_val <= 0.85:
                local_t = (ti_val - 0.75) / 0.10
                colors_rgb[i] = torch.tensor([int(127 * local_t), 0, 255], device=self.device)
            else:
                local_t = (ti_val - 0.85) / 0.15
                colors_rgb[i] = torch.tensor(
                    [int(127 * (1 - local_t)), 0, int(255 * (1 - local_t))], device=self.device
                )

        colors_bgr = colors_rgb[:, [2, 1, 0]].to('cpu', non_blocking=True).numpy()
        panel_height = len(legend_distances) * (box_height + 5) + 10
        panel_alpha = 0.7
        cv2.rectangle(mask_cpu, (legend_x - 5, legend_y - 5),
                      (legend_x + 175, legend_y + panel_height),
                      panel_alpha, -1)

        for i, (dist, color_bgr) in enumerate(zip(legend_distances, colors_bgr)):
            y_pos = legend_y + i * (box_height + 5)
            if y_pos + box_height > self.H:
                break

            color_tuple = tuple(int(c) for c in color_bgr)

            cv2.rectangle(overlay_cpu, (legend_x, y_pos),
                          (legend_x + box_width, y_pos + box_height),
                          color_tuple, -1)
            cv2.rectangle(overlay_cpu, (legend_x, y_pos),
                          (legend_x + box_width, y_pos + box_height),
                          (255, 255, 255), 1)

            mask_cpu[y_pos:y_pos + box_height, legend_x:legend_x + box_width] = 1.0

            if dist < 1.0:
                dist_text = f"{dist * 100:.0f}cm"
            elif dist < 1000:
                dist_text = f"{dist:.1f}m"
            else:
                dist_text = f"{dist / 1000:.1f}km"

            cv2.putText(overlay_cpu, dist_text,
                        (legend_x + text_offset_x, y_pos + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            text_width = len(dist_text) * 8
            mask_cpu[y_pos:y_pos + box_height,
            legend_x + text_offset_x:legend_x + text_offset_x + text_width] = 1.0

        self.legend_overlay = torch.from_numpy(overlay_cpu).to(self.device)
        self.legend_mask = torch.from_numpy(mask_cpu).unsqueeze(2).to(self.device)

    @torch.no_grad()
    def blend(self, frame):
        """Ultra-fast GPU blending."""
        frame_f32 = frame.float()
        legend_f32 = self.legend_overlay.float()
        blended = frame_f32 * (1 - self.legend_mask) + legend_f32 * self.legend_mask
        return blended.byte()

class DepthFocus:

    @staticmethod
    @stable_checker
    def depth_to_amplitude(x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the point-wise function based on:
        f(x) = 1 / log2(1.14*x + 1.3)
        Result = exp(f(x)) - exp(f(1)) + 1
        Uses in-place operations and clips the final result between 0.1 and 1.0.
        """
        # Constant calculation:
        # Subtraction Term = exp(f(1)) - 1 = 2.17509542752 - 1 = 1.17509542752
        return x.mul(0.03).neg_().add_(1.0).pow_(1.7).clamp_(min=0.001, max=1.0)
        # return x.mul(1.14).add_(3).log2_().reciprocal_().exp_().sub_(1.17509542752).clamp_(min=0.001, max=1.0)

    @staticmethod
    def focus_point_(point, tmp_abs, tmp_mul):
        # tmp_abs = |p|
        torch.abs(point, out=tmp_abs)

        # tmp_mul = p * |p|
        torch.mul(point, tmp_abs, out=tmp_mul)

        # point = 2p
        point.mul_(2)

        # point = 2p - p|p|
        point.sub_(tmp_mul)
        return point

class LandmarksGenerator:
    def __init__(self, H, W, max_points=10000, dtype=torch.float32, device='cuda'):
        self.dtype = dtype
        self.device = device
        self.max_points = max_points

        # 1. Stride Setup (4x Downsample via View)
        self.stride = 4
        self.dH = H // self.stride
        self.dW = W // self.stride
        self.num_pixels = self.dH * self.dW

        # 2. Pre-allocate Coordinate Grid
        # Flatten immediately. shape: [num_pixels, 2]
        grid_y = torch.arange(0, H, step=self.stride, device=device, dtype=dtype).unsqueeze(1).expand(self.dH, self.dW)
        grid_x = torch.arange(0, W, step=self.stride, device=device, dtype=dtype).unsqueeze(0).expand(self.dH, self.dW)

        # Contiguous flattened grid for masking
        self.flat_grid = torch.stack((grid_y, grid_x), dim=2).view(-1, 2).contiguous()

        # 3. Pre-allocate Scratch Buffers (Fixed VRAM Footprint)
        self.buf_frame = torch.zeros((self.dH, self.dW), dtype=dtype, device=device)
        self.buf_gy = torch.zeros((self.dH, self.dW), dtype=dtype, device=device)
        self.buf_gx = torch.zeros((self.dH, self.dW), dtype=dtype, device=device)
        self.buf_weights = torch.zeros(self.num_pixels, dtype=dtype, device=device)
        self.buf_rand = torch.zeros(self.num_pixels, dtype=dtype, device=device)
        self.buf_mask = torch.zeros(self.num_pixels, dtype=torch.bool, device=device)

        # SINGLE OUTPUT BUFFER
        # Allocated to MAX capacity (all pixels).
        self.landmarks_buffer = torch.zeros(self.num_pixels * 2, dtype=dtype, device=device)

        # 4. Pre-calc Static Heuristics
        cy, cx = self.dH / 2.0, self.dW / 2.0
        y_dist = (torch.arange(self.dH, device=device) - cy) / cy
        x_dist = (torch.arange(self.dW, device=device) - cx) / cx
        dist_sq = y_dist.pow(2).unsqueeze(1) + x_dist.pow(2).unsqueeze(0)
        mask_spatial = torch.sqrt(dist_sq)
        mask_spatial = (mask_spatial - mask_spatial.min()) / (mask_spatial.max() - mask_spatial.min())

        center_bias = 1.0 - mask_spatial * 0.7
        center_bias.clamp_(0, 1).pow_(1.5)

        edge = 30 // self.stride
        center_bias[:edge, :] = 0
        center_bias[-edge:, :] = 0
        center_bias[:, :edge] = 0
        center_bias[:, -edge:] = 0
        self.flat_center_bias = center_bias.view(-1)

    @stable_checker
    @torch.inference_mode()
    def generate_landmarks(self, candidate_amps_dense, frame, target_points=5000, existing_landmarks=None):
        # 1. Determine Need (Math only, no branches)
        # If existing_landmarks is None, num_existing becomes 0 via logical OR
        num_existing = (existing_landmarks is not None and existing_landmarks.shape[0]) or 0
        needed = target_points - num_existing

        # 2. Zero-Alloc Downsample
        # frame is [H, W, 3]. We take channel 0. View stride avoids copy.
        self.buf_frame.copy_(frame[::self.stride, ::self.stride, 0])

        # 3. In-Place Gradient Calculation
        torch.sub(self.buf_frame[1:, :], self.buf_frame[:-1, :], out=self.buf_gy[:-1, :])
        torch.sub(self.buf_frame[:, 1:], self.buf_frame[:, :-1], out=self.buf_gx[:, :-1])
        self.buf_gy.pow_(2)
        self.buf_gx.pow_(2)
        self.buf_gy.add_(self.buf_gx).sqrt_()
        self.buf_gy.mul_(0.00392)  # 1/255

        # 4. Prepare Weights
        flat_grads = self.buf_gy.view(-1)
        # View amps with stride, flatten
        flat_amps = candidate_amps_dense[::self.stride, ::self.stride].reshape(-1)

        torch.mul(flat_grads, 0.4, out=self.buf_weights)
        self.buf_weights.add_(0.6)
        self.buf_weights.mul_(flat_amps)
        self.buf_weights.mul_(self.flat_center_bias)
        self.buf_weights.add_(1e-6)

        # 5. Calculate Threshold (Scaler)
        # If needed <= 0, scaler becomes <= 0.
        total_weight = self.buf_weights.sum()
        scaler = needed / total_weight

        # 6. Generate Mask
        # If scaler <= 0, (weights * scaler) <= 0.
        # Since rand is [0, 1), (rand < neg) is False.
        # Mask becomes all False naturally. No 'if' needed.
        self.buf_rand.uniform_(0, 1)
        self.buf_weights.mul_(scaler)
        torch.lt(self.buf_rand, self.buf_weights, out=self.buf_mask)

        # 7. Select into Buffer
        # Resize to 0 allows PyTorch to reuse storage without reallocation complaints
        self.landmarks_buffer.resize_(0)

        # Expand mask to (y,x) view and select
        mask_2d = self.buf_mask.unsqueeze(1).expand(-1, 2)
        torch.masked_select(self.flat_grid, mask_2d, out=self.landmarks_buffer)

        # 8. View and Cap
        # View entire buffer as pairs. If buffer is empty, view is [0, 2]
        result_view = self.landmarks_buffer.view(-1, 2)
        return result_view



def main():
    import cv2
    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    frame_path = "img_1.png"
    depth_path = "enhanced_depth_hdr.png"

    # 2. Load IO
    frame_cv = cv2.imread(frame_path)
    H, W = frame_cv.shape[:2]
    depth_cv = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    # 3. Prepare Tensors (Non-Blocking)
    # Upload uint8 -> GPU -> Float.
    frame_tensor = torch.from_numpy(frame_cv).to(device, non_blocking=True).float()
    amps_tensor = torch.from_numpy(depth_cv).to(device, non_blocking=True)
    amps_tensor = DepthFocus.depth_to_amplitude(amps_tensor.to(dtype=torch.float32))

    # 4. Initialize
    generator = LandmarksGenerator(H, W, max_points=10000, device=device)

    # 5. Execution (Timed)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # Warmup
    generator.generate_landmarks(amps_tensor, frame_tensor, target_points=100)

    start_evt.record()
    landmarks = generator.generate_landmarks(amps_tensor, frame_tensor, target_points=2048)
    end_evt.record()
    torch.cuda.synchronize()

    print(f"Time: {start_evt.elapsed_time(end_evt):.3f} ms | Points: {landmarks.shape[0]}")

    # 6. Visualization (Zero-Branching)
    # NumPy handles empty arrays gracefully.
    pts = landmarks.cpu().numpy()
    pts_int = np.round(pts).astype(np.int32)

    # Clip ensures indices are valid even if array is empty
    ys = np.clip(pts_int[:, 0], 0, H - 2)
    xs = np.clip(pts_int[:, 1], 0, W - 2)

    # Assign Color
    # If ys/xs are empty, this operation does nothing (no-op), no error.
    frame_cv[ys, xs] = [0, 255, 0]
    frame_cv[ys, xs + 1] = [0, 255, 0]
    frame_cv[ys + 1, xs] = [0, 255, 0]
    frame_cv[ys + 1, xs + 1] = [0, 255, 0]

    cv2.imshow("Surgical Landmarks", frame_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()