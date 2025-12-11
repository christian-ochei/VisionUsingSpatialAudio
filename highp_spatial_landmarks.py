import threading
import typing
import torch.fft
import torch.utils.dlpack
from typing import List, Dict
import torch.fft
import cupy
import numpy as np
import os
import itertools
from scipy.io import loadmat
from scipy.spatial import Delaunay
from scipy.fft import next_fast_len
from typing import Tuple
from scipy.signal import lfilter
from typing import Union
import torch
import math
from line_profiler_pycharm import profile

from utils import check_nans

_indexing_type = typing.Union[slice, int, torch.Tensor]


class Head:
    """
    Optimized Head class for loading and caching 3D audio data.

    This class loads CIPIC HRIR data at its native 44.1kHz sample rate,
    pre-calculates a 360-degree HRIR grid, and saves it to a highly
    optimized cache.

    It no longer performs any resampling; it assumes the audio engine
    will handle the "resampling" in the frequency domain during convolution.
    """

    # --- CIPIC Database Definitions ---
    AZIMUTH_ANGLES = [
        -90, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15,
        20, 25, 30, 35, 40, 45, 55, 65, 90,
    ]
    ELEVATION_ANGLES = -45 + 5.625 * np.arange(0, 50)
    POINTS_NP = np.array(list(itertools.product(AZIMUTH_ANGLES, ELEVATION_ANGLES)))
    AZ_DICT = dict(zip(AZIMUTH_ANGLES, np.arange(len(AZIMUTH_ANGLES))))
    EL_DICT = dict(zip(ELEVATION_ANGLES, np.arange(len(ELEVATION_ANGLES))))

    # --- Anthropometry Data Loading ---
    try:
        anthro_data = loadmat('CIPIC_hrtf_database/anthropometry/anthro.mat')
        PARAMETERS = {
            'info': ['Age:', 'Sex:', 'Weight:'],
            'X': ['Head width:', 'Head height:', 'Head depth:', 'Pinna offset down:',
                  'Pinna offset back:', 'Neck width:', 'Neck height:', 'Neck depth:',
                  'Torso top width:', 'Torso top height:', 'Torso top depth:',
                  'Shoulder width:', 'Head offset forward:', 'Height:',
                  'Seated height:', 'Head circumference:', 'Shoulder circumference:'],
            'D': ['Cavum concha height:', 'Cymba concha height:', 'Cavum concha width:',
                  'Fossa height:', 'Pinna height:', 'Pinna width:',
                  'Intertragal incisure width:', 'Cavum concha depth:'],
            'theta': ['Pinna rotation angle:', 'Pinna flare angle:']
        }

        anthro_ids_raw = np.squeeze(anthro_data['id'])
        ANTHRO_ID = [str(int(id_)) for id_ in anthro_ids_raw]
        ID_TO_IDX = dict(zip([int(id_) for id_ in ANTHRO_ID], range(len(ANTHRO_ID))))

        for key, value in anthro_data.items():
            if key not in ['__header__', '__version__', '__globals__', 'id', 'sex']:
                if key == 'age':
                    anthro_data[key][np.isnan(anthro_data[key])] = 0
                    anthro_data[key] = np.squeeze(value.astype('int')).astype('str')
                    anthro_data[key][anthro_data[key] == '0'] = '-'
                else:
                    anthro_data[key] = np.around(np.squeeze(value), 1).astype('str')
                    anthro_data[key][anthro_data[key] == 'nan'] = '-'

    except Exception as e:
        print(f"CRITICAL WARNING: Could not parse 'id' from anthro.mat: {e}")
        ID_TO_IDX = {}


    def __init__(self, num_speakers, block_size: int, subject='018', device='cuda', dtype=torch.float32,
                 grid_size: Tuple[int, int] = (256, 256)):
        """
        Initialize Head.

        Args:
            block_size: Audio processing block size in samples (e.g., 512, 1024).
                        This is the *only* parameter needed from the audio engine.
            subject: CIPIC subject ID.
            device: Torch device ('cuda' or 'cpu').
            dtype: Torch dtype.
            grid_size: HRIR cache grid resolution (azimuth, elevation).
        """
        if not Head.ID_TO_IDX:
            raise RuntimeError("Head class failed to initialize. Could not load subject IDs.")

        idx = Head.ID_TO_IDX[int(subject)]
        self.properties_cm = {}
        for param, values in Head.PARAMETERS.items():
            if param != 'info':
                LR_const = 2 if param in ('D', 'theta') else 1
                for j in range(LR_const * len(values)):
                    if Head.anthro_data[param][idx][j] != '-':
                        key = values[j % len(values)].strip().lower().replace(':', '').replace(' ', '_')
                        self.properties_cm[key] = float(Head.anthro_data[param][idx][j])

        self.device = device
        self.dtype = dtype
        self.subject = subject
        self.block_size = block_size
        self.native_sample_rate = 44100  # Hard-coded, we never resample

        if not (isinstance(grid_size, tuple) and len(grid_size) == 2):
            raise ValueError("grid_size must be a tuple of two integers (e.g., (256, 256))")
        self.grid_size = grid_size

        # --- Cache Management ---
        cache_dir = "hrir_cache"
        # FFT cache is specific to subject, block size, and grid.
        cache_file = f"subject_{self.subject}_b{self.block_size}_grid{self.grid_size[0]}x{self.grid_size[1]}_NATIVE_FFT.pt"
        cache_path = os.path.join(cache_dir, cache_file)

        N_AZ, N_EL = self.grid_size
        print(f"\rFFT HRIR cache initialized ({N_AZ}x{N_EL})        ")
        self.cache_az_step = 360.0 / (N_AZ - 1)
        self.cache_el_step = 135.0 / (N_EL - 1)

        # --- Head State (Global) ---
        self.head_position = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.head_rotation = torch.eye(3, device=device, dtype=dtype)
        self.head_rotation_dirty = True

        # Add to __init__ or cache initialization:
        self.cache_az_offset = 180.0
        self.cache_el_offset = 45.0
        self.cache_az_step_inv = 1.0 / self.cache_az_step
        self.cache_el_step_inv = 1.0 / self.cache_el_step
        self.hypot_buf = torch.empty(num_speakers*5*8, device=device, dtype=dtype)
        self.arctan2_buf = torch.empty(num_speakers*5*8, device=device, dtype=dtype)
        self.arctan2_buf2 = torch.empty(num_speakers*5*8, device=device, dtype=dtype)
        self.az_idx = torch.empty(num_speakers*5*8*8, device=device, dtype=torch.long)
        self.el_idx = torch.empty(num_speakers*5*8*8, device=device, dtype=torch.long)
        self.azimuth_add_buf = torch.empty(num_speakers*8, dtype=dtype, device=device)
        self.elevation_add_buf = torch.empty(num_speakers*8, dtype=dtype, device=device)
        if os.path.exists(cache_path):
            try:
                print(f"Loading cached FFT Head data from {cache_path}...")
                cached_data = torch.load(cache_path, map_location=self.device, weights_only=True)

                # Validate cache
                if (tuple(cached_data.get('grid_size', (0, 0))) != self.grid_size or
                        cached_data.get('block_size', 0) != self.block_size):
                    print("Cache parameter mismatch! Re-initializing...")
                    raise ValueError("Cache parameter mismatch")

                self.hrir_length = cached_data['hrir_length']
                self.fft_size = cached_data['fft_size']
                self.overlap_size = cached_data['overlap_size']
                self.hrir_fft_cache = cached_data['hrir_fft_cache']
                self.T_inv = cached_data['T_inv']
                self.cache_az_step = cached_data['cache_az_step']
                self.cache_el_step = cached_data['cache_el_step']

                self.hrtf_points = torch.from_numpy(self.POINTS_NP).to(device=self.device, dtype=self.dtype)
                self.triangulation = Delaunay(self.POINTS_NP)

                print(f"Cache loaded. FFT size: {self.fft_size}, Block: {self.block_size}")

            except Exception as e:
                print(f"Failed to load cache: {e}. Re-initializing...")
                self._full_initialization(subject)
        else:
            print(f"Cache not found. Initializing FFT HRIR cache...")
            self._full_initialization(subject)

    @staticmethod

    def _muffle(hrir_np: np.ndarray, a: float = 0.6) -> np.ndarray:
        """
        Applies a simple one-pole low-pass filter to simulate rear-hemisphere muffling.
        Vectorized using scipy.signal.lfilter (direct form II transposed).
        """
        # Numerator (b) coefficients: applied to input x[n]
        b = [1.0 - a]
        # Denominator (a) coefficients: applied to output y[n]
        # The '1.0' corresponds to y[n], the '-a' corresponds to y[n-1]
        # (scipy expects coefficients on the left side of the difference equation)
        a_coeffs = [1.0, -a]
        return lfilter(b, a_coeffs, hrir_np)


    def _full_initialization(self, subject):
        """Full initialization with NATIVE 44.1kHz HRIRs."""
        hrir = loadmat(f'CIPIC_hrtf_database/standard_hrir_database/subject_{subject}/hrir_final.mat')

        hrir_l = np.array(hrir['hrir_l'])
        hrir_r = np.array(hrir['hrir_r'])
        hrir_combined = np.concatenate([hrir_l[..., None], hrir_r[..., None]], -1)
        # Transpose to [Az, El, Channel, Samples]
        hrir_combined = hrir_combined.transpose(0, 1, 3, 2)

        self.hrir_data_native = torch.from_numpy(hrir_combined).to(device=self.device, dtype=self.dtype)
        self.hrir_length = self.hrir_data_native.shape[-1]  # Should be 200

        self.hrtf_points = torch.from_numpy(self.POINTS_NP).to(device=self.device, dtype=self.dtype)

        # --- FFT Size Calculation ---
        self.fft_size = next_fast_len(self.hrir_length + self.block_size - 1)
        self.overlap_size = self.hrir_length - 1

        print(f"Native HRIR length: {self.hrir_length} samples.")
        print(f"FFT size: {self.fft_size}, Block: {self.block_size}")

        # Initialize FFT cache
        self.N_AZ, self.N_EL = self.grid_size
        self.hrir_fft_cache = torch.zeros((self.N_AZ, self.N_EL, 2, self.fft_size),
                                          device=self.device, dtype=torch.complex64)

        self._initialize_triangulation()
        self._initialize_fft_cache_360()

        # Save cache atomically
        self._save_cache()


    def _save_cache(self):
        """Save FFT cache atomically to prevent corruption."""
        cache_dir = "hrir_cache"
        cache_file = f"subject_{self.subject}_b{self.block_size}_grid{self.grid_size[0]}x{self.grid_size[1]}_NATIVE_FFT.pt"
        cache_path = os.path.join(cache_dir, cache_file)
        temp_cache_path = cache_path + ".tmp"

        print(f"Saving initialized FFT cache to {cache_path}...")

        try:
            os.makedirs(cache_dir, exist_ok=True)
            data_to_save = {
                'hrir_length': self.hrir_length,
                'fft_size': self.fft_size,
                'overlap_size': self.overlap_size,
                'hrir_fft_cache': self.hrir_fft_cache.cpu(),
                'T_inv': self.T_inv.cpu(),
                'cache_az_step': self.cache_az_step,
                'cache_el_step': self.cache_el_step,
                'grid_size': self.grid_size,
                'block_size': self.block_size,
            }

            torch.save(data_to_save, temp_cache_path, _use_new_zipfile_serialization=True)
            os.replace(temp_cache_path, cache_path)
            print(f"FFT cache saved successfully.")

        except Exception as e:
            print(f"WARNING: Failed to save RFFT cache: {e}")
            if os.path.exists(temp_cache_path):
                try:
                    os.remove(temp_cache_path)
                except Exception as e:
                    print(f"WARNING: Failed to remove failed RFFT cache: {e}")
                    pass


    def _initialize_fft_cache_360(self):
        """Initialize 360-degree FFT cache."""

        self.cache_az_range = np.linspace(-180, 180, self.N_AZ)
        self.cache_el_range = np.linspace(-45, 90, self.N_EL)

        print(f"Caching 360° FFT HRIRs ({self.N_AZ}x{self.N_EL})...", end="", flush=True)

        # Pre-allocate padding buffers
        padded_left = torch.zeros(self.fft_size, device=self.device, dtype=self.dtype)
        padded_right = torch.zeros(self.fft_size, device=self.device, dtype=self.dtype)

        for i, azimuth in enumerate(self.cache_az_range):
            for j, elevation in enumerate(self.cache_el_range):
                interp_az = azimuth
                is_rear = False

                if azimuth > 90.0:
                    interp_az = 180.0 - azimuth
                    is_rear = True
                elif azimuth < -90.0:
                    interp_az = -180.0 - azimuth
                    is_rear = True

                hrir_left_np, hrir_right_np = self._interpolate_hrir_triangulation(interp_az, elevation)

                if is_rear:
                    hrir_left_np = self._muffle(hrir_left_np)
                    hrir_right_np = self._muffle(hrir_right_np)

                padded_left.zero_()
                padded_right.zero_()
                padded_left[:self.hrir_length] = torch.from_numpy(hrir_left_np).to(self.device)
                padded_right[:self.hrir_length] = torch.from_numpy(hrir_right_np).to(self.device)

                # --- Perform FFT ---
                self.hrir_fft_cache[i, j, 0] = torch.fft.fft(padded_left, n=self.fft_size)
                self.hrir_fft_cache[i, j, 1] = torch.fft.fft(padded_right, n=self.fft_size)

            if i % (max(1, self.N_AZ // 20)) == 0 or i == self.N_AZ - 1:
                progress = int((i + 1) / self.N_AZ * 100)
                print(f"\rCaching 360° FFT HRIRs... {progress}%", end="", flush=True)


    def _initialize_triangulation(self):
        """Initialize Delaunay triangulation."""
        self.triangulation = Delaunay(self.POINTS_NP)
        T_inv_np = self._calculate_T_inv_np()
        self.T_inv = torch.from_numpy(T_inv_np).to(device=self.device, dtype=self.dtype)


    def _calculate_T_inv_np(self):
        """Calculate inverse transformation matrices for triangulation."""
        A = self.POINTS_NP[self.triangulation.simplices][:, 0, :]
        B = self.POINTS_NP[self.triangulation.simplices][:, 1, :]
        C = self.POINTS_NP[self.triangulation.simplices][:, 2, :]

        T = np.empty((2 * A.shape[0], A.shape[1]))
        T[::2, :] = A - C
        T[1::2, :] = B - C
        T = T.reshape(-1, 2, 2)
        return np.linalg.inv(T)


    def _interpolate_hrir_triangulation(self, azimuth: float, elevation: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate HRIR using triangulation (used during caching)."""
        azimuth = np.clip(azimuth, -90, 90)
        elevation = np.clip(elevation, -45, 231.875)

        position = np.array([azimuth, elevation])
        triangle_idx = self.triangulation.find_simplex(position)

        if triangle_idx == -1:
            # Fallback: nearest neighbor
            distances = np.sum((self.POINTS_NP - position) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            az_idx = self.AZ_DICT[self.AZIMUTH_ANGLES[nearest_idx % len(self.AZIMUTH_ANGLES)]]
            el_idx = self.EL_DICT[self.ELEVATION_ANGLES[nearest_idx // len(self.AZIMUTH_ANGLES)]]
            return (self.hrir_data_native[az_idx, el_idx, 0].cpu().numpy(),
                    self.hrir_data_native[az_idx, el_idx, 1].cpu().numpy())

        vertices = self.POINTS_NP[self.triangulation.simplices[triangle_idx]]
        T_inv_np = self.T_inv[triangle_idx].cpu().numpy()
        g = np.dot(position - vertices[2], T_inv_np)
        g_1, g_2 = g[0], g[1]
        g_3 = 1 - g_1 - g_2

        if g_1 >= 0 and g_2 >= 0 and g_3 >= 0:
            # Barycentric interpolation
            v_indices = [
                (self.AZ_DICT[v[0]], self.EL_DICT[v[1]]) for v in vertices
            ]
            hrir_l = (g_1 * self.hrir_data_native[v_indices[0][0], v_indices[0][1], 0] +
                      g_2 * self.hrir_data_native[v_indices[1][0], v_indices[1][1], 0] +
                      g_3 * self.hrir_data_native[v_indices[2][0], v_indices[2][1], 0])
            hrir_r = (g_1 * self.hrir_data_native[v_indices[0][0], v_indices[0][1], 1] +
                      g_2 * self.hrir_data_native[v_indices[1][0], v_indices[1][1], 1] +
                      g_3 * self.hrir_data_native[v_indices[2][0], v_indices[2][1], 1])
            return hrir_l.cpu().numpy(), hrir_r.cpu().numpy()
        else:
            # Fallback: nearest vertex in triangle
            distances = [np.linalg.norm(position - v) for v in vertices]
            nearest_vertex = vertices[np.argmin(distances)]
            az_idx, el_idx = self.AZ_DICT[nearest_vertex[0]], self.EL_DICT[nearest_vertex[1]]
            return (self.hrir_data_native[az_idx, el_idx, 0].cpu().numpy(),
                    self.hrir_data_native[az_idx, el_idx, 1].cpu().numpy())

    def get_cached_fft_hrir(self, azimuth, elevation) -> torch.Tensor:
        """
        Get interpolated FFT HRIR from 360° cache.
        Supports batched inputs [...].

        Returns: [..., 2, fft_size] complex tensor
        """
        # Track if input was scalar-like (scalar or single-element 1D)
        is_scalar_like = azimuth.dim() == 0 or (azimuth.dim() == 1 and azimuth.shape[0] == 1)
        size = azimuth.flatten().shape[0]
        # Convert scalar to 1D for uniform processing
        if azimuth.dim() == 0:
            azimuth = azimuth[None]
            elevation = elevation[None]
        # Map to cache indices - no flattening needed!
        az_idx = make_shape(azimuth, self.az_idx).copy_(torch.add(azimuth, self.cache_az_offset, out=make_shape(azimuth, self.azimuth_add_buf)).mul_(self.cache_az_step_inv))
        el_idx = make_shape(azimuth, self.el_idx).copy_(torch.add(elevation, self.cache_el_offset, out=make_shape(elevation, self.elevation_add_buf)).mul_(self.cache_el_step_inv))

        # Advanced indexing preserves the shape of index tensors
        fft_stereo = self.hrir_fft_cache[az_idx, el_idx]  # [..., 2, fft_size]

        # Return scalar result if input was scalar-like
        return fft_stereo[0] if is_scalar_like else fft_stereo

    # --- Coordinate Transforms & Setters ---
    def world_to_head_coordinates(self, world_pos: torch.Tensor) -> torch.Tensor:
        """Transform world to head-relative coordinates. Supports batched [..., 3]."""
        # This is the exact transformation used by well tested transform_global_point_to_screen_space()
        relative_pos = world_pos - self.head_position
        return relative_pos @ self.head_rotation

    def head_coords_to_cipic_angles(self, head_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert head coords to CIPIC angles. Supports batched [..., 3]."""
        x = head_coords[..., 0]
        y = head_coords[..., 1]
        z = head_coords[..., 2]
        azimuth_deg = torch.rad2deg_(torch.arctan2(x, z, out=make_shape(z, self.arctan2_buf)))
        elevation_deg = torch.rad2deg_(torch.arctan2(
            y, torch.hypot(x, z, out=make_shape(x, self.hypot_buf)), out=make_shape(x, self.arctan2_buf2))
        )
        return azimuth_deg, elevation_deg

    def set_head_rotation(self, rotation_matrix: torch.Tensor):
        """Set head rotation matrix."""
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        # This unfortunately stalls the GPU to a level where it's better to just always make head_rotation_dirty
        # Interleave all position and HRIR assignments to prevent stalls
        # if not torch.allclose(self.head_rotation, rotation_matrix):
        self.head_rotation = rotation_matrix
        self.head_rotation_dirty = True

class LocalizationBlueprint:
    def __init__(self, device, dtype, num_speakers=1, speed_of_sound=343.0):
        self.memory_lock = threading.RLock()
        self.device = device
        self.dtype = dtype
        self.num_speakers = num_speakers
        self.speed_of_sound = speed_of_sound
        self.late_spatial_regions = 4  # Reduced from 8
        self.late_reverb_mix = 0.9


        # --- BATCHIFIED ---
        self.speaker_positions = torch.zeros((self.num_speakers, 3), device=device, dtype=dtype)
        self.fine_speaker_positions = torch.zeros((self.num_speakers, 3), device=device, dtype=dtype)
        self.course_speaker_positions = torch.zeros((self.num_speakers, 3), device=device, dtype=dtype)
        self.fine_speaker_positions[:, 1] = 0
        self.course_speaker_positions[:, 1] = 0
        # Set a default position
        self.speaker_positions[:, 2] = 15.0

        # --- GLOBAL ---
        self.room_dimensions = torch.tensor([40.0, 20.0, 60.0], device=device, dtype=dtype)
        self.half_dims = self.room_dimensions / 2.0
        self.absorption_db = torch.tensor(-4.8, device=device, dtype=dtype)
        self.params_dirty = True
        self.small_val_tensor = torch.tensor(1e9, device=device, dtype=dtype)
        self.rot60 = self.compute_rt60()

        self.speaker_indices = torch.arange(self.num_speakers, device=self.device, dtype=torch.long)

        self._position_change_intermediate_buffer = [torch.empty(
            [self.num_speakers, 3],
            dtype=self.dtype,
            device=self.device
        ) for _ in range(4)]

        self._position_sub_intermediate_buffer = [torch.empty(
            [self.num_speakers, 3],
            dtype=self.dtype,
            device=self.device
        ) for _ in range(4)]

        self._position_minimum_intermediate_buffer = [torch.empty(
            (self.num_speakers, 3),
            dtype=self.dtype,
            device=self.device
        ) for _ in range(4)]

        self._position_greater_intermediate_buffer = [torch.empty(
            [self.num_speakers, 3],
            dtype=torch.bool,
            device=self.device
        ) for _ in range(4)]

        self._position_sum_intermediate_buffer = [torch.empty(
            self.num_speakers,
            dtype=self.dtype,
            device=self.device
        ) for _ in range(4)]

        self._position_thresh_intermediate_buffer = [torch.empty(
            self.num_speakers,
            dtype=self.dtype,
            device=self.device
        ) for _ in range(4)]

        self._position_greater_output_intermediate_buffer = [torch.empty(
            self.num_speakers,
            dtype=torch.bool,
            device=self.device
        ) for _ in range(4)]

        # --- GLOBAL (Image source indices are constant) ---
        self.er_indices_np = self._get_early_reflection_indices()
        self.er_indices = torch.from_numpy(self.er_indices_np).to(device=self.device, dtype=self.dtype)
        self.num_er_sources = len(self.er_indices) + 1  # +1 for direct path
        self.dist_to_walls = torch.zeros([self.num_speakers, 6], dtype=self.dtype, device=device)

        # --- GLOBAL (Optimized reflection index calculations) ---
        W, H, D = self.room_dimensions
        self.scaled_er_indices_w = self.er_indices[:, 0] * W
        self.scaled_er_indices_h = self.er_indices[:, 1] * H
        self.scaled_er_indices_d = self.er_indices[:, 2] * D
        self.centered_er_indices_w = (self.er_indices[:, 0] % 2 * 2 - 1)
        self.centered_er_indices_h = (self.er_indices[:, 1] % 2 * 2 - 1)
        self.centered_er_indices_d = (self.er_indices[:, 2] % 2 - 1)

        self.mirrors = torch.tensor([
            [-1, 1, 1], [1, -1, 1], [1, 1, -1],
            [-1, 1, 1], [1, -1, 1], [1, 1, -1]
        ], device=device, dtype=dtype)

        self.offsets = torch.tensor([
            [-W, 0, 0], [0, -H, 0], [0, 0, -D],
            [W, 0, 0], [0, H, 0], [0, 0, D]
        ], device=device, dtype=dtype)

        # --- GLOBAL (Cluster definitions are global) ---
        self.enhanced_clusters = torch.tensor(
            LocalizationBlueprint._define_optimized_clusters(), device=self.device, dtype=self.dtype
        )
        self.num_enhanced_clusters = len(self.enhanced_clusters)

        # --- GLOBAL Late Reverb Setup ---
        self.comb_delays_s = torch.tensor([0.0297, 0.0371, 0.0411, 0.0437], device=self.device, dtype=self.dtype)
        self.allpass_delays_s = torch.tensor([0.0051, 0.0037], device=self.device, dtype=self.dtype)
        self.late_weights = torch.tensor([0.3, 0.25, 0.25, 0.2], device=self.device, dtype=self.dtype)  # Added

        self.max_speakers = self.num_speakers + 50 + self.num_speakers // 2
        self.buf_delta = torch.zeros((self.max_speakers, 3), device=device, dtype=dtype)
        self.buf_dist_direct = torch.zeros(self.max_speakers, device=device, dtype=dtype)
        self.buf_g_direct = torch.zeros(self.max_speakers, device=device, dtype=dtype)
        self.buf_amp_direct = torch.zeros(self.max_speakers, device=device, dtype=dtype)  # Output Accumulator

        # 2. Proximity Effect Buffers
        self.buf_walls_neg = torch.zeros((self.max_speakers, 3), device=device, dtype=dtype)
        self.buf_walls_pos = torch.zeros((self.max_speakers, 3), device=device, dtype=dtype)
        self.buf_min_wall = torch.zeros(self.max_speakers, device=device, dtype=dtype)
        self.buf_indices = torch.zeros(self.max_speakers, device=device, dtype=torch.long)  # for torch.min
        self.buf_prox_gain = torch.zeros(self.max_speakers, device=device, dtype=dtype)

        # 3. Reflection & Image Source Buffers
        self.buf_img_src = torch.zeros((self.max_speakers, 6, 3), device=device, dtype=dtype)
        self.buf_d_refs = torch.zeros((self.max_speakers, 6), device=device, dtype=dtype)
        self.buf_amp_refs = torch.zeros((self.max_speakers, 6), device=device, dtype=dtype)
        self.buf_rms_refs = torch.zeros(self.max_speakers, device=device, dtype=dtype)

        # 4. Cluster & Scratch Buffers
        self.buf_dist_sq = torch.zeros(self.max_speakers, device=device, dtype=dtype)
        self.buf_reverb_scratch = torch.zeros(4, device=device, dtype=dtype)  # Tiny buffer for reverb

        self.k_decay = 0.67
        self.asymptote = 0.985
        self.dynamic_range = 1.8 - 0.985

        self.proximity_dirty = torch.ones(self.num_speakers, dtype=torch.bool, device=self.device)
        self.reflections_dirty = torch.ones(self.num_speakers, dtype=torch.bool, device=self.device)
        self.hrirs_dirty = torch.ones(self.num_speakers, dtype=torch.bool, device=self.device)
        self.absorption_gains_dirty = True
        self.any_reflections_dirty = True

    @staticmethod
    def _define_optimized_clusters() -> List[Tuple[float, float]]:
        """Define optimized angular clusters (Global)."""
        return [
            (0, 0),  # Front center
            (-45, 0),  # Front left
            (45, 0),  # Front right
            (-90, 0),  # Left
            (90, 0),  # Right
            (0, 45),  # Above
            (0, -30),  # Below
            (-90, 30),  # Left up (represents back/up region)
        ]

    @staticmethod
    def _get_early_reflection_indices():
        """Get image source indices for early reflections - (NumPy Init, Global)."""
        return np.array([
            [0, 1, 0],  # top
            [0, -1, 0],  # bottom
            [-1, 0, 0],  # left
            [1, 0, 0],  # right
        ], dtype=np.float32)
    
    def compute_rt60(self):
        """Compute RT60 (PyTorch) (Global)."""
        # Remove this, that if check is a GPU Stall
        # if self.absorption_db >= -0.0001:
        #     return self.small_val_tensor
        volume = torch.prod(self.room_dimensions)
        surface_area = 2 * torch.sum(self.room_dimensions * torch.roll(self.room_dimensions, 1))
        alpha = 1.0 - 10 ** (self.absorption_db / 10.0)
        rt60 = 0.161 * volume / (surface_area * alpha)
        return torch.clamp_(rt60, min=0.1, max=1000.0)


# ============================================================================
# 1. MODIFIED CUDA KERNEL FOR 1ST-ORDER FILTERS (Prox Filter)
# ============================================================================
# This kernel is ALREADY BATCHED (B = blockIdx.x). We will use B = num_speakers.
first_order_iir_batched_N_filters_kernel = cupy.RawKernel(r'''
extern "C" __global__
void first_order_iir_batched_N_filters_kernel(
    const float* x_in,     // Input tensor [B, N]
    float* y_out,          // Output tensor [B, N]
    const float* zi_in,    // Input state [B]
    float* zf_out,         // Output state [B]
    const float* b0_in,    // Batched b0 [B]
    const float* b1_in,    // Batched b1 [B]
    const float* a1_in,    // Batched a1 [B]
    int N_samples
) {
    int batch_id = blockIdx.x; // This will be speaker_id (0 to num_speakers-1)

    // Get filter coefficients for this batch item
    float b0 = b0_in[batch_id];
    float b1 = b1_in[batch_id];
    float a1 = a1_in[batch_id];

    const float* x = x_in + batch_id * N_samples;
    float* y = y_out + batch_id * N_samples;
    float z = zi_in[batch_id];

    for (int i = 0; i < N_samples; i++) {
        float xi = x[i];
        float yi = b0 * xi + z;
        y[i] = yi;
        z = b1 * xi - a1 * yi;
    }
    zf_out[batch_id] = z;
}
''', 'first_order_iir_batched_N_filters_kernel')

# ============================================================================
# 3. MODIFIED CUDA KERNEL FOR PARALLEL MULTI-FILTER COMB
# ============================================================================
# This kernel is ALREADY BATCHED (speaker_id = blockIdx.x).
multi_comb_filter_N_speakers_kernel = cupy.RawKernel(r'''
extern "C" __global__
void multi_comb_filter_N_speakers_kernel(
    const float* x_in,           // Input signals [num_speakers, N_samples]
    float* y_out,                // Output signals [num_speakers, N_filters, N_samples]
    float* packed_buffer,        // Packed state buffer [num_speakers, sum(N_delays)]
    const int* buffer_offsets,   // Shared Offsets for each filter [N_filters]
    int* buffer_idx_inout,       // State indices [num_speakers, N_filters]
    const float* feedback_in,    // Feedback values [num_speakers, N_filters]
    const int* N_delays_in,      // Shared Delay lengths [N_filters]
    int N_samples,
    int N_filters,               // e.g., 4
    int total_N_delays           // sum(N_delays)
) {
    int speaker_id = blockIdx.x; // num_speakers
    int filter_id = blockIdx.y;  // N_filters (e.g., 4)

    // Get parameters for this specific filter
    int N_delays = N_delays_in[filter_id];

    // Get pointers for this filter
    const float* x = x_in + speaker_id * N_samples; // Shared input for this speaker

    // Output: y_out[speaker_id, filter_id, sample]
    float* y = y_out + (speaker_id * N_filters + filter_id) * N_samples;

    // Buffer: packed_buffer[speaker_id, offset + sample]
    float* buf_batch = packed_buffer + speaker_id * total_N_delays + buffer_offsets[filter_id];

    // Index: buffer_idx_inout[speaker_id, filter_id]
    int idx_pos = speaker_id * N_filters + filter_id;
    int idx = buffer_idx_inout[idx_pos];

    // Feedback: feedback_in[speaker_id, filter_id]
    float feedback = feedback_in[idx_pos];

    // The core logic is identical
    for (int i = 0; i < N_samples; i++) {
        float delayed = buf_batch[idx];
        float yi = x[i] + delayed;
        y[i] = yi;
        buf_batch[idx] = feedback * yi;

        idx++;
        if (idx == N_delays) {
            idx = 0;
        }
    }

    // Write back the updated index
    buffer_idx_inout[idx_pos] = idx;
}
''', 'multi_comb_filter_N_speakers_kernel')

# ============================================================================
# 5. NEW CUDA KERNEL FOR ALL-PASS FILTERS
# ============================================================================
# This kernel is ALREADY BATCHED (speaker_id = blockIdx.x).
allpass_filter_N_speakers_kernel = cupy.RawKernel(r'''
extern "C" __global__
void allpass_filter_N_speakers_kernel(
    const float* x_in,           // Input signals [num_speakers, N_filters, N_samples]
    float* y_out,                // Output signals [num_speakers, N_filters, N_samples]
    float* buffer,               // State buffer [num_speakers, N_filters, N_delays]
    int* buffer_idx_inout,       // State indices [num_speakers, N_filters]
    const float* gain_in,        // Gain values (g) [num_speakers, N_filters]
    int N_delays,                // Delay length for this filter
    int N_samples,
    int N_filters                // e.g., 4
) {
    int speaker_id = blockIdx.x; // num_speakers
    int filter_id = blockIdx.y;  // N_filters (e.g., 4)

    // Get pointers for this filter
    const float* x = x_in + (speaker_id * N_filters + filter_id) * N_samples;
    float* y = y_out + (speaker_id * N_filters + filter_id) * N_samples;

    // Buffer: buffer[speaker_id, filter_id, sample]
    float* buf = buffer + (speaker_id * N_filters + filter_id) * N_delays;

    // Index: buffer_idx_inout[speaker_id, filter_id]
    int idx_pos = speaker_id * N_filters + filter_id;
    int idx = buffer_idx_inout[idx_pos];

    // Gain: gain_in[speaker_id, filter_id]
    float g = gain_in[idx_pos];

    // Loop for all samples
    for (int i = 0; i < N_samples; i++) {
        float delayed_v = buf[idx]; // v[n-M]
        float xi = x[i];            // x[n]

        // v[n] = x[n] - g*v[n-M]
        float vi = xi - g * delayed_v;

        // y[n] = g*v[n] + v[n-M]
        float yi = g * vi + delayed_v;

        y[i] = yi;
        buf[idx] = vi; // Store v[n]

        idx++;
        if (idx == N_delays) {
            idx = 0;
        }
    }

    // Write back the updated index
    buffer_idx_inout[idx_pos] = idx;
}
''', 'allpass_filter_N_speakers_kernel')


# ============================================================================
# 4. MODIFIED CUDA KERNEL WRAPPERS (Now accept pre-converted CuPy arrays)
# ============================================================================
def first_order_iir_cupy_batched(
        b0_cupy: cupy.ndarray, b1_cupy: cupy.ndarray, a1_cupy: cupy.ndarray,
        x_cp: cupy.ndarray,
        z_cupy: cupy.ndarray,
        y_out_cupy: cupy.ndarray,
        stream: cupy.cuda.ExternalStream
):
    """
    Batched First-order IIR using CuPy kernel.
    Accepts pre-converted CuPy arrays for persistent state (z), output (y),
    and coefficients (b, a).
    Accepts Torch tensor for the DYNAMIC input signal (x).
    """
    B, N_samples = x_cp.shape

    # --- Convert DYNAMIC input (signal) ---
    first_order_iir_batched_N_filters_kernel(
        (B,), (1,),  # Grid = (num_speakers,), Block = (1,)
        (x_cp, y_out_cupy, z_cupy, z_cupy,
         b0_cupy, b1_cupy, a1_cupy, N_samples),
        stream=stream
    )
    # No return needed; y_out_cupy and z_cupy are modified in-place (and share
    # memory with their corresponding torch tensors).


def multi_comb_filter_N_speakers_cupy(
        x_cp: cupy.ndarray,  # [num_speakers, N_samples] (Dynamic)
        y_cupy: cupy.ndarray,  # [num_speakers, N_filters, N_samples] (Persistent)
        packed_buffer_cupy: cupy.ndarray,  # [num_speakers, sum(N_delays)] (Persistent)
        buffer_offsets_cupy: cupy.ndarray,  # [N_filters] (Persistent)
        buffer_idx_cupy: cupy.ndarray,  # [num_speakers, N_filters] (Persistent)
        feedback_cupy: cupy.ndarray,  # [num_speakers, N_filters] (Persistent, smoothed)
        N_delays_cupy: cupy.ndarray,  # [N_filters] (Persistent)
        total_N_delays: int,  # sum(N_delays)
        stream: cupy.cuda.ExternalStream
) -> None:
    """
    Batched Multi-Comb Filter for N speakers.
    Accepts pre-converted CuPy arrays for persistent state/buffers.
    Accepts Torch tensor for DYNAMIC input (x).
    """
    num_speakers, N_samples = x_cp.shape
    N_filters = N_delays_cupy.shape[0]

    # --- Convert DYNAMIC input (signal) ---
    # Grid: (num_speakers, N_filters), Block: (1,)
    multi_comb_filter_N_speakers_kernel(
        (num_speakers, N_filters), (1,),
        (x_cp, y_cupy,
         packed_buffer_cupy, buffer_offsets_cupy, buffer_idx_cupy,
         feedback_cupy, N_delays_cupy,
         N_samples, N_filters, total_N_delays),
        stream=stream
    )
    # No return needed, y_cupy is modified in-place


def allpass_filter_N_speakers_cupy(
        x_cupy: cupy.ndarray,  # [num_speakers, N_filters, N_samples] (Persistent)
        y_cupy: cupy.ndarray,  # [num_speakers, N_filters, N_samples] (Persistent)
        buffer_cupy: cupy.ndarray,  # [num_speakers, N_filters, N_delays] (Persistent)
        buffer_idx_cupy: cupy.ndarray,  # [num_speakers, N_filters] (Persistent)
        gain_cupy: cupy.ndarray,  # [num_speakers, N_filters] (Persistent)
        N_delays: int,
        stream: cupy.cuda.ExternalStream
) -> None:
    """
    Batched All-Pass Filter for N speakers.
    Accepts pre-converted CuPy arrays for all inputs/outputs.
    NO DLPack calls needed here.
    """
    num_speakers, N_filters, N_samples = x_cupy.shape

    # Grid: (num_speakers, N_filters), Block: (1,)
    allpass_filter_N_speakers_kernel(
        (num_speakers, N_filters), (1,),
        (x_cupy, y_cupy,
         buffer_cupy, buffer_idx_cupy,
         gain_cupy, N_delays,
         N_samples, N_filters),
        stream=stream
    )
    # No return needed, y_cupy is modified in-place


# def calculate_peak_gain_estimate(
#         head_position: torch.Tensor,
#         speaker_position: torch.Tensor,
#         absorption_db: float,
#         # Engine parameters for accurate estimation:
#         min_distance: float = 0.13,  # 0.05 mic + 0.08 speaker
#         late_reverb_mix: float = 0.9,
#         num_early_reflections: int = 4,  # 4 wall reflections
#         avg_reflection_distance_ratio: float = 1.5,  # Images are ~1.5x farther
# ) -> torch.Tensor:
#     """
#     Estimates peak gain of SpaceSimulationEngine output.
#
#     Models three components:
#     1. Direct path: 1/distance with proximity boost
#     2. Early reflections: Attenuated by absorption and distance
#     3. Late reverb: Diffuse field energy, independent of source distance
#
#     Args:
#         head_position: [N, 3] or [3] position of listener
#         speaker_position: [N, 3] or [3] position of source(s)
#         absorption_db: Wall absorption coefficient in dB (negative = more absorption)
#         min_distance: Minimum distance for gain clamping
#         late_reverb_mix: Late reverb mixing coefficient (0-1)
#         num_early_reflections: Number of first-order wall reflections
#         avg_reflection_distance_ratio: Average distance multiplier for image sources
#
#     Returns:
#         Estimated peak amplitude gain [N] or scalar
#     """
#     # Direct path gain (1/r law with proximity effect)
#     distance = torch.linalg.norm(speaker_position - head_position, dim=-1)
#     distance_clamped = torch.clamp(distance, min=min_distance)
#
#     # Direct gain with approximate proximity boost
#     # Engine applies high-shelf filter near walls; approximate as ~1.2x boost at min distance
#     proximity_boost = 1.0 + 0.2 * torch.exp(-distance_clamped / 0.3)
#     direct_gain = proximity_boost / distance_clamped
#
#     # Reflection/absorption coefficient
#     reflection_coef = 10.0 ** (absorption_db / 20.0)  # Linear amplitude coefficient
#
#     # Early reflections contribution
#     # Each reflection travels farther and is attenuated by one wall bounce
#     avg_image_distance = distance_clamped * avg_reflection_distance_ratio
#     early_reflection_gain = (reflection_coef / avg_image_distance) * num_early_reflections
#
#     # Late reverb contribution (diffuse field)
#     # Models the steady-state diffuse field energy in the room
#     # Key insight: Late reverb is ~independent of source distance,
#     # depends mainly on room size and absorption
#
#     # Approximate RT60 (Sabine equation, simplified)
#     # More absorption (lower reflection_coef) → shorter RT60 → less reverb energy
#     # This is a heuristic approximation; actual engine calculates RT60 from room volume
#     rt60_estimate = 0.161 * (6.0 / max(0.01, 1.0 - reflection_coef ** 2))  # Seconds, order of magnitude
#
#     # Diffuse field energy is proportional to RT60 and (1 - absorption)
#     # The factor of ~0.3 is empirically tuned to match engine output levels
#     diffuse_field_gain = 0.3 * rt60_estimate * reflection_coef * late_reverb_mix
#
#     # Total peak gain is the ADDITIVE combination (not multiplicative)
#     # All three components sum in the time domain
#     total_peak_gain = direct_gain + early_reflection_gain + diffuse_field_gain
#
#     # Apply calibration factor to match engine
#     # This accounts for HRIR convolution gain, filter coefficients, etc.
#     calibration = 0.435  # Empirically tuned to match ~2.3x observed offset
#
#     return total_peak_gain * calibration
#

def make_shape(x, out_buffer):
    return out_buffer[:x.flatten().shape[0]].reshape(x.shape)


class SpaceSimulationEngine(LocalizationBlueprint):
    
    def __init__(
            self, num_speakers=1, sample_rate=44100, block_size=256, subject='018', device='cuda', dtype=torch.float32,
            stream: torch.cuda.Stream = None, present='fast', speed_of_sound=343.0
    ):
        super().__init__(device=device, dtype=dtype, num_speakers=num_speakers, speed_of_sound=speed_of_sound)
        self.sample_rate = sample_rate
        self.present = present
        self.block_size = block_size
        self.user_block_size = block_size
        self.device = device
        self.dtype = dtype
        self.frame_counter = 0
        self.crossfade_len = block_size
        self.crossfade_linspace = torch.linspace(0, math.pi, self.crossfade_len, device=self.device, dtype=self.dtype)
        self.reset_mask = torch.zeros(self.num_speakers, dtype=torch.bool, device=self.device)

        # --- 3. Early Reflections (Energy Sum) ---
        # --- GLOBAL (1 Head) ---
        self.head = Head(
            num_speakers=num_speakers, block_size=block_size, subject=subject, device=self.device, dtype=self.dtype
        )

        # We do this for efficient, head delta computation for sparse, speaker parameter updates
        self.last_head_positions = torch.zeros((self.num_speakers, 3), device=device, dtype=dtype)

        # Use self.num_speakers from base class
        print(f"Initializing SpaceSimulationEngine for {self.num_speakers} speakers.")

        # --- Physical Parameters (Global) ---
        self.mic_radius = 0.05
        self.speaker_radius = 0.08
        self.min_distance = torch.tensor(self.mic_radius + self.speaker_radius, device=self.device, dtype=self.dtype)
        self.num_precise_reflections = 4

        # --- BATCHIFIED Early Reflections Setup ---
        self.er_delay_line_size = 8192
        is_power_of_two = (self.er_delay_line_size & (self.er_delay_line_size - 1) == 0) and self.er_delay_line_size > 0
        assert is_power_of_two, "er_delay_line_size MUST be a power of 2 for fast bitwise masking."
        self.er_size_mask = self.er_delay_line_size - 1
        self.er_delay_line = torch.zeros((self.num_speakers, self.er_delay_line_size), device=self.device,
                                         dtype=self.dtype)
        self.er_write_head = 0  # Global write head, as all speakers are processed in sync

        # --- BATCHIFIED Per-reflection parameters ---
        self.reflection_delays = torch.zeros((self.num_speakers, self.num_er_sources), device=self.device,
                                             dtype=torch.long)
        self.reflection_gains = torch.zeros((self.num_speakers, self.num_er_sources), device=self.device,
                                            dtype=self.dtype)
        self.reflection_positions = torch.zeros((self.num_speakers, self.num_er_sources, 3), device=self.device,
                                                dtype=self.dtype)

        # --- NEW: State for parameter interpolation ---
        self.prev_reflection_delays = torch.zeros_like(self.reflection_delays)
        self.prev_reflection_gains = torch.zeros_like(self.reflection_gains)
        # --- BATCHIFIED Proximity Filter (CUDA) ---
        self.prox_filter_zi_torch = torch.zeros(self.num_speakers, device=self.device, dtype=self.dtype)
        self.prox_filter_b0 = torch.ones(self.num_speakers, device=self.device, dtype=self.dtype)
        self.prox_filter_b1 = torch.zeros(self.num_speakers, device=self.device, dtype=self.dtype)
        self.prox_filter_a1 = torch.zeros(self.num_speakers, device=self.device, dtype=self.dtype)
        self.prox_filter_output_buffer = torch.zeros((self.num_speakers, self.block_size), device=self.device,
                                                     dtype=self.dtype)

        # Calculate delayed signal with CURRENT parameters
        self.delayed_signal_out = {
            0: torch.empty([self.num_speakers, self.block_size, self.num_precise_reflections], dtype=torch.float32,
                           device=self.device),
            1: torch.empty([self.num_speakers, self.block_size, self.num_precise_reflections], dtype=torch.float32,
                           device=self.device),
            2: torch.empty([self.num_speakers, self.block_size, self.num_er_sources - self.num_precise_reflections],
                           dtype=torch.float32, device=self.device),
            3: torch.empty([self.num_speakers, self.block_size, self.num_er_sources - self.num_precise_reflections],
                           dtype=torch.float32, device=self.device),
        }

        self.read_indices_intermediate = {
            0: torch.empty([self.num_speakers, self.block_size, self.num_precise_reflections], dtype=torch.long,
                           device=self.device),
            1: torch.empty([self.num_speakers, self.block_size, self.num_precise_reflections], dtype=torch.long,
                           device=self.device),
            2: torch.empty([self.num_speakers, self.block_size, self.num_er_sources - self.num_precise_reflections],
                           dtype=torch.long,
                           device=self.device),
            3: torch.empty([self.num_speakers, self.block_size, self.num_er_sources - self.num_precise_reflections],
                           dtype=torch.long,
                           device=self.device),
        }

        # --- BATCHIFIED Late Reverb (CUDA) ---
        self.comb_delays_samples = (self.comb_delays_s * self.sample_rate).to(dtype=torch.int32)
        self.total_comb_delays = int(self.comb_delays_samples.sum().item())
        self.comb_buffer_offsets = torch.zeros(self.late_spatial_regions, device=self.device, dtype=torch.int32)
        self.comb_buffer_offsets[1:] = torch.cumsum(self.comb_delays_samples[:-1], dim=0)

        self.comb_packed_buffer = torch.zeros((self.num_speakers, self.total_comb_delays), device=self.device,
                                              dtype=self.dtype)
        self.comb_buffer_idx = torch.zeros((self.num_speakers, self.late_spatial_regions), device=self.device,
                                           dtype=torch.int32)

        self.comb_feedback = torch.zeros((self.num_speakers, self.late_spatial_regions), device=self.device,
                                         dtype=self.dtype)

        # NEW: This stores where we WANT to go (Target State)
        self.target_comb_feedback = torch.zeros((self.num_speakers, self.late_spatial_regions), device=self.device,
                                                dtype=self.dtype)

        # NEW: Parameter smoothing coefficient (0.0 to 1.0).
        # 0.05 roughly equals 100ms smoothing time at 256 block size, eliminating pops.
        self.param_smoothing_factor = 0.05

        self.comb_output_buffer = torch.zeros((self.num_speakers, self.late_spatial_regions, self.block_size),
                                              device=self.device, dtype=self.dtype)
        self.direct_path_signal = torch.zeros((self.num_speakers, self.block_size), device=self.device,
                                              dtype=self.dtype)

        # --- GLOBAL (HRIRs for late reverb regions) ---
        self.late_fft_hrirs = torch.zeros((self.late_spatial_regions, 2, self.head.fft_size),
                                          device=self.device, dtype=torch.complex64)

        # --- BATCHIFIED (Overlap state is per-speaker, per-region) ---
        self.late_spatial_overlap = torch.zeros(
            (self.num_speakers, self.late_spatial_regions, 2, self.head.overlap_size),
            device=self.device,
            dtype=self.dtype)

        # --- BATCHIFIED (Precise reflection HRIRs are per-speaker, per-reflection) ---
        self.az_diff_buf = torch.empty(self.num_speakers*60, dtype=dtype, device=device)
        self.el_diff_buf = torch.empty(self.num_speakers*60, dtype=dtype, device=device)
        self.az_diff_min_buf = torch.empty(self.num_speakers*60, dtype=dtype, device=device)
        self.cluster_az_buf = torch.empty(self.num_speakers*60, dtype=dtype, device=device)
        self.cluster_el_buf = torch.empty(self.num_speakers*60, dtype=dtype, device=device)
        self.precise_fft_hrirs = torch.zeros((self.num_speakers, self.num_precise_reflections, 2, self.head.fft_size),
                                             device=self.device,
                                             dtype=torch.complex64)
        self.precise_overlap = torch.zeros((self.num_speakers, self.num_precise_reflections, 2, self.head.overlap_size),
                                           device=self.device,
                                           dtype=self.dtype)

        # --- BATCHIFIED (Cluster assignments and buffers are per-speaker) ---
        self.reflection_clusters = torch.zeros((self.num_speakers, self.num_er_sources), device=self.device,
                                               dtype=torch.int64)
        self.cluster_buffers = torch.zeros((self.num_speakers, self.num_enhanced_clusters, self.block_size),
                                           device=self.device,
                                           dtype=self.dtype)

        # --- GLOBAL (Cluster HRIRs are a global LUT) ---
        self.cluster_fft_hrirs = torch.zeros((self.num_enhanced_clusters, 2, self.head.fft_size),
                                             device=self.device,
                                             dtype=torch.complex64)

        # --- BATCHIFIED (Cluster overlap state is per-speaker, per-cluster) ---
        self.cluster_overlap = torch.zeros((self.num_speakers, self.num_enhanced_clusters, 2, self.head.overlap_size),
                                           device=self.device,
                                           dtype=self.dtype)

        self._precompute_cluster_fft_hrirs()  # Global
        self._precompute_late_fft_hrirs()  # Global

        # --- GLOBAL (Block indices are global) ---
        self.block_indices = torch.arange(self.block_size, device=self.device, dtype=torch.long)
        self.early_sum = torch.zeros([num_speakers, self.block_size], device=self.device, dtype=torch.float32)
        self.early_sum_delayed_intermediate = torch.zeros([num_speakers, self.block_size], device=self.device,
                                                   dtype=torch.float32)

        # --- BATCHIFIED (Output buffers are per-speaker) ---
        self.stereo_output_buffer = torch.zeros((self.num_speakers, self.block_size, 2), device=self.device,
                                                dtype=self.dtype)

        # --- BATCHIFIED ALL-PASS FILTER SETUP ---
        # self.allpass_delays_s = np.array([0.0051, 0.0037])
        self.allpass_gains_val = np.array([0.7, 0.7])  # Standard all-pass gain
        self.num_allpass_filters = len(self.allpass_delays_s)

        self.allpass_delays_samples = [int(a) for a in (self.allpass_delays_s * self.sample_rate)]

        self.allpass_buffers = []
        self.allpass_indices = []
        self.allpass_gains = []

        for i in range(self.num_allpass_filters):
            delay_len = self.allpass_delays_samples[i]
            gain_val = self.allpass_gains_val[i]

            # State buffer: [num_speakers, N_filters (regions), N_delays]
            buf = torch.zeros((self.num_speakers, self.late_spatial_regions, delay_len),
                              device=self.device, dtype=self.dtype)
            # State indices: [num_speakers, N_filters (regions)]
            idx = torch.zeros((self.num_speakers, self.late_spatial_regions),
                              device=self.device, dtype=torch.int32)
            # Gain: [num_speakers, N_filters (regions)]
            gain = torch.full((self.num_speakers, self.late_spatial_regions),
                              gain_val, device=self.device, dtype=self.dtype)

            self.allpass_buffers.append(buf)
            self.allpass_indices.append(idx)
            self.allpass_gains.append(gain)

        self.allpass_temp_buffer = torch.zeros((self.num_speakers, self.late_spatial_regions, self.block_size),
                                               device=self.device, dtype=self.dtype)

        # --- BATCHIFIED PARAMETER CHANGE CROSSFADE ---
        # Ramps are [1, block_size, 1] for broadcasting with [N, block_size, 2]
        self.param_fade_out_ramp = torch.linspace(1.0, 0.0, self.block_size, device=self.device,
                                                  dtype=self.dtype).view(1, -1, 1)
        self.param_fade_in_ramp = torch.linspace(0.0, 1.0, self.block_size, device=self.device,
                                                 dtype=self.dtype).view(1, -1, 1)

        self.input_padded_zeros = torch.zeros((self.num_speakers * 8, 2, self.head.fft_size), device=self.device,
                                              dtype=self.dtype)
        self.stereo_padded_zeros = torch.zeros((self.num_speakers * 8, 2, self.block_size), device=self.device,
                                               dtype=self.dtype)

        self.input_freq_buf = [torch.empty([self.num_speakers * 8, 2, self.head.fft_size // 2 + 1], device=self.device,
                                           dtype=torch.complex64) for _ in range(3)]
        self.output_freq_new_buf = [
            torch.empty([self.num_speakers * 8, 2, self.head.fft_size // 2 + 1], device=self.device,
                        dtype=torch.complex64) for _ in range(3)]
        self.output_time_new_buf = [
            torch.empty([self.num_speakers * 8, 2, self.head.fft_size], device=self.device, dtype=self.dtype) for _ in
            range(3)]
        self.output_freq_old_buf = [
            torch.empty([self.num_speakers * 8, 2, self.head.fft_size // 2 + 1], device=self.device,
                        dtype=torch.complex64) for _ in range(3)]
        self.output_time_old_buf = [
            torch.empty([self.num_speakers * 8, 2, self.head.fft_size], device=self.device, dtype=self.dtype) for _ in
            range(3)]
        self.crossfade1 = [
            torch.empty([self.num_speakers * 8, 2, self.block_size], device=self.device, dtype=self.dtype) for _ in
            range(3)]
        self.crossfade2 = [
            torch.empty([self.num_speakers * 8, 2, self.block_size], device=self.device, dtype=self.dtype) for _ in
            range(3)]

        # Preallocate buffers
        num_cluster_reflections = self.num_er_sources - self.num_precise_reflections
        self.read_indices_i_cluster = torch.empty(
            [num_speakers, self.block_size, num_cluster_reflections], dtype=torch.long,
            device=self.device)
        self.frac_cluster = torch.empty([num_speakers, self.block_size, num_cluster_reflections],
                                        dtype=self.dtype, device=self.device)

        self.read_indices_i_precise = torch.empty(
            [num_speakers, self.block_size, self.num_precise_reflections], dtype=torch.long,
            device=self.device)
        self.frac_precise = torch.empty([num_speakers, self.block_size, self.num_precise_reflections],
                                        dtype=self.dtype, device=self.device)

        # --- Pre-allocate speaker indices for delay line reading ---
        self.batch_indices = torch.arange(self.num_speakers, device=self.device)
        self.speaker_indices_delay_read = self.batch_indices.view(-1, 1, 1)

        # orders is global [num_er_sources]
        self.orders = torch.cat((
            torch.tensor([0.0], device=self.device, dtype=self.dtype),
            torch.sum(torch.abs(self.er_indices), dim=1)
        ))
        reflection_gain = 10 ** (self.absorption_db / 20.0)  # Global scalar
        self.pow_reflection_gain = torch.pow(reflection_gain, self.orders)
        self.fc = 250
        self.theta = 2 * math.pi * self.fc / self.sample_rate  # scalar
        self.tan_theta_2 = math.tan(self.theta / 2.0)

        # --- HRIR Transition System (Anti-Pop) ---
        self.transition_length = 32  # Samples for smooth HRIR transition
        self.previous_hrir_fft: Dict[
            int, Union[torch.Tensor, None]] = {}  # Dict[int, torch.Tensor] - stores previous HRIR FFTs by ID
        self.hrir_dirty_flags = {}  # Dict[int, bool] - tracks if HRIR changed for each ID
        self.input_history_buffer = {}  # Dict[int, torch.Tensor] - stores input history per context

        # Calculate batch sizes for each context
        # ID 0: Precise reflections
        batch_size_precise = self.num_speakers * self.num_precise_reflections
        # ID 1: Cluster reflections
        batch_size_cluster = self.num_speakers * self.num_enhanced_clusters
        # ID 2: Late reverb
        batch_size_late = self.num_speakers * self.late_spatial_regions

        batch_sizes = [batch_size_precise, batch_size_cluster, batch_size_late]

        # Initialize for 3 IDs: 0=precise, 1=cluster, 2=late
        for hrir_id in range(3):
            self.previous_hrir_fft[hrir_id] = None
            self.hrir_dirty_flags[hrir_id] = False
            # Buffer size must accommodate: overlap_size + transition_length + block_size
            history_size = self.head.overlap_size + self.transition_length + self.block_size
            self.input_history_buffer[hrir_id] = torch.zeros(
                (batch_sizes[hrir_id], history_size),
                device=self.device,
                dtype=self.dtype
            )

        # === EXTENDED CROSSFADE ===
        self.crossfade_len = block_size

        # === FIX: USE RAISED-COSINE (HANN) FADE ===
        t_pi = self.crossfade_linspace
        fade_out = (1 + torch.cos(t_pi)) / 2
        fade_in = (1 - torch.cos(t_pi)) / 2

        # Reshape for broadcasting
        self.fade_out_3d = fade_out.view(1, 1, -1)
        self.fade_in_3d = fade_in.view(1, 1, -1)
        # ============================================

        # --- CUDA Stream and Pre-converted CuPy Buffers ---
        self.stream = stream
        if self.stream is None:
            print("Running on CPU. CUDA streams and kernels will be disabled.")
            self.cupy_stream = None
        else:
            self.cupy_stream = cupy.cuda.ExternalStream(self.stream.cuda_stream)
            # ANSI colors
            RED = "\033[91m"
            ORANGE = "\033[33m"
            YELLOW = "\033[93m"
            GREEN = "\033[92m"
            END = "\033[0m"

            p = self.stream.priority  # range: -3 to 0

            # Label mapping (your custom logic)
            priority_label = {
                -3: "CRITICAL",
                -2: "HIGH",
                -1: "MEDIUM",
                 0: "NORMAL",  # highest number = least important
            }.get(p, "UNKNOWN")

            # Color mapping (green → red)
            priority_color = {
                -3: GREEN,
                -2: YELLOW,
                -1: ORANGE,
                 0: RED,
            }.get(p, END)

            print(f"{priority_color}DSP CUDA stream created (Priority {p}: {priority_label}){END}")

            # --- NEW: Create CuPy views of persistent buffers ONCE ---
            print("Creating persistent CuPy DLPack views...")
            self.direct_path_signal_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.direct_path_signal))

            # Prox Filter
            self.prox_filter_zi_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.prox_filter_zi_torch))
            self.prox_filter_output_buffer_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.prox_filter_output_buffer))
            self.prox_filter_b0_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.prox_filter_b0))
            self.prox_filter_b1_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.prox_filter_b1))
            self.prox_filter_a1_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.prox_filter_a1))

            # Comb Filters
            self.comb_delays_samples_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.comb_delays_samples))
            self.comb_buffer_offsets_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.comb_buffer_offsets))
            self.comb_packed_buffer_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.comb_packed_buffer))
            self.comb_buffer_idx_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.comb_buffer_idx))
            self.comb_output_buffer_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.comb_output_buffer))
            self.early_sum_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.early_sum))

            # --- NEW: Create view for feedback ---
            # This view now points to the one-and-only feedback buffer
            self.comb_feedback_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.comb_feedback))
            # All-pass Filters
            self.allpass_buffers_cupy = [cupy.from_dlpack(torch.utils.dlpack.to_dlpack(b)) for b in
                                         self.allpass_buffers]
            self.allpass_indices_cupy = [cupy.from_dlpack(torch.utils.dlpack.to_dlpack(i)) for i in
                                         self.allpass_indices]
            self.allpass_gains_cupy = [cupy.from_dlpack(torch.utils.dlpack.to_dlpack(g)) for g in
                                       self.allpass_gains]
            self.allpass_temp_buffer_cupy = cupy.from_dlpack(
                torch.utils.dlpack.to_dlpack(self.allpass_temp_buffer))
            print("Persistent CuPy views created.")

        # For backwards compatibility only
        self.compat_speaker_positions = torch.empty((self.num_speakers, 3), dtype=dtype, device=device)

    
    def _update_proximity_filter(self, indexing_values: _indexing_type = slice(None)):
        half_dims = self.room_dimensions / 2.0  # [3]

        speaker_positions = self.speaker_positions[indexing_values]
        self.dist_to_walls[indexing_values, :3] = half_dims - speaker_positions
        self.dist_to_walls[indexing_values, 3:6] = half_dims + speaker_positions

        # Normalized distance: 1.0 (at wall) to 0.0 (1.5m away)
        normalized_dist = 1.0 - self.dist_to_walls[indexing_values] / 1.5

        # --- FIX IS HERE ---
        # 1. Relu handles distance > 1.5m (sets neg to 0)
        # 2. Clamp(max=1.0) handles distance < 0m (sets values > 1.0 to 1.0)
        masked_normalized_dist = torch.clamp(normalized_dist, min=0.0, max=1.0)
        # -------------------

        # 3. Square the result and apply gain.
        gain_boost_contributions = 0.5 * masked_normalized_dist ** 2

        # 4. Sum to get the final boost.
        gain_boost = torch.sum(gain_boost_contributions, dim=1)  # [N]

        shelf_gain = 1.0 + gain_boost  # [N]
        mu = torch.pow(10, shelf_gain / 20.0)  # [N]

        # This part needs to handle division by zero if mu = -1, but tan is small
        gamma_denom = (1.0 + (4.0 / (1.0 + mu)) * self.tan_theta_2)
        # Add epsilon to avoid div by zero if gamma_denom is 0
        gamma = (1.0 - (4.0 / (1.0 + mu)) * self.tan_theta_2) / (gamma_denom + 1e-9)  # [N]

        # --- Set coefficients for CUDA kernel ---
        b0 = (1 + gamma) * mu / 2.0  # [N]
        b1 = -(1 + gamma) * mu / 2.0  # [N]
        a1 = -gamma  # [N]

        # --- CRITICAL: Update IN-PLACE to keep CuPy views valid ---
        self.prox_filter_b0[indexing_values] = b0
        self.prox_filter_b1[indexing_values] = b1
        self.prox_filter_a1[indexing_values] = a1

    def _precompute_cluster_fft_hrirs(self):
        """Pre-compute FFT HRIRs for all clusters - OPTIMIZED (Global)."""
        cluster_azimuths = self.enhanced_clusters[:, 0]
        cluster_elevations = self.enhanced_clusters[:, 1]

        # Returns [N_clusters, 2, fft_size]
        fft_hrirs_batch = self.head.get_cached_fft_hrir(cluster_azimuths, cluster_elevations)
        self.cluster_fft_hrirs[:] = fft_hrirs_batch

    def _precompute_late_fft_hrirs(self):
        """Pre-compute FFT HRIRs for late reverb (4 regions) - OPTIMIZED (Global)."""
        late_directions = torch.tensor(
            [[0, 0], [-45, 0], [45, 0], [-90, 0]],
            device=self.device,
            dtype=self.dtype
        )  # Front, FL, FR, Left

        azimuths = late_directions[:, 0]
        elevations = late_directions[:, 1]

        # Returns [4, 2, fft_size]
        fft_hrirs_batch = self.head.get_cached_fft_hrir(azimuths, elevations)
        self.late_fft_hrirs[:] = fft_hrirs_batch

    
    def _update_input_history(self, hrir_id: int, new_input: torch.Tensor):
        """
        Update rolling input history buffer for transition generation.

        Args:
            hrir_id: Context ID (0=precise, 1=cluster, 2=late)
            new_input: [B, block_size] - New input samples to append
        """
        B, block_size = new_input.shape
        history_buffer = self.input_history_buffer[hrir_id]

        # Roll buffer left by block_size, append new input at end
        history_buffer[:, :-block_size] = history_buffer[:, block_size:]
        history_buffer[:, -block_size:] = new_input

    
    def _fft_convolve_overlap_add_stereo_batched(
            self,
            signal_batch_mono: torch.Tensor,
            fft_hrir_batch_stereo: torch.Tensor,
            overlap_buffer_batch_stereo: torch.Tensor,

            input_freq: torch.Tensor,
            output_freq_new: torch.Tensor,
            output_time_new: torch.Tensor,
            output_freq_old: torch.Tensor,

            output_time_old: torch.Tensor,
            crossfade1: torch.Tensor,
            crossfade2: torch.Tensor,
            hrir_id: int
    ) -> torch.Tensor:
        """
        FFT convolution with smooth HRIR transitions using rfft.
        Uses output crossfading with extended duration for maximum smoothness.
        """
        B = signal_batch_mono.shape[0]
        fft_size = self.head.fft_size
        block_size = self.head.block_size
        overlap_size = self.head.overlap_size
        overlap_add_length = min(overlap_size, block_size)

        # Calculate the size for rfft output
        rfft_size = fft_size // 2 + 1

        # --- Slice the full HRIR FFT to its rfft-compatible half ---
        # We assume fft_hrir_batch_stereo is the full complex FFT.
        # We only need the first rfft_size components for rfft-based convolution.
        hrir_freq_rfft = fft_hrir_batch_stereo[:, :, :rfft_size]

        # Update input history (for potential future use)
        self._update_input_history(hrir_id, signal_batch_mono)

        # === COMPUTE NEW HRIR OUTPUT (ALWAYS NEEDED) ===
        # Pad input for FFT (real tensor)
        input_padded = self.input_padded_zeros[:B]
        input_padded[:, :, :block_size] = signal_batch_mono[:, None]

        # --- FFT of current input (real-to-complex) ---
        torch.fft.rfft(input_padded, n=fft_size, dim=2, out=input_freq)
        # Convolve with NEW HRIR (using the rfft-sliced version)
        torch.mul(input_freq, hrir_freq_rfft, out=output_freq_new)
        # --- Inverse FFT (complex-to-real) ---
        # .real is no longer needed; irfft returns a real tensor
        torch.fft.irfft(output_freq_new, n=fft_size, dim=2, out=output_time_new)
        # Extract block and new overlap
        current_block_new = output_time_new[:, :, :block_size]
        new_overlap = output_time_new[:, :, block_size:block_size + overlap_size]
        # Check transition requirement
        # self.previous_hrir_fft will now store the rfft-sliced version
        need_transition = self.hrir_dirty_flags.get(hrir_id, False) and \
                          self.previous_hrir_fft.get(hrir_id) is not None

        if need_transition:
            # === COMPUTE OUTPUT WITH OLD HRIR ===
            # self.previous_hrir_fft[hrir_id] is already in rfft format
            torch.mul(input_freq, self.previous_hrir_fft[hrir_id], out=output_freq_old)
            # --- Inverse FFT (complex-to-real) ---
            torch.fft.irfft(output_freq_old, n=fft_size, dim=2, out=output_time_old)
            current_block_old = output_time_old[:, :, :block_size]

            # === FIX: APPLY OLA TO OLD STREAM ONLY ===
            current_block_old[:, :, :overlap_add_length] += overlap_buffer_batch_stereo[:, :, :overlap_add_length]

            # === CREATE FINAL BLOCK FROM FADE ===
            output_block = current_block_new
            output_block[:, :, :self.crossfade_len] = (
                torch.mul(current_block_old[:, :, :self.crossfade_len], self.fade_out_3d, out=crossfade1).add_(
                    torch.mul(current_block_new[:, :, :self.crossfade_len], self.fade_in_3d, out=crossfade2))
            )

            # Clear dirty flag
            self.hrir_dirty_flags[hrir_id] = False

            # --- Update previous HRIR reference with the rfft-sliced version ---
            self.previous_hrir_fft[hrir_id] = hrir_freq_rfft
        else:
            # === STANDARD (NO TRANSITION) OVERLAP-ADD ===
            current_block_new[:, :, :overlap_add_length] += overlap_buffer_batch_stereo[:, :, :overlap_add_length]
            output_block = current_block_new

            # --- Update previous HRIR reference if it was empty (with rfft-sliced version) ---
            if self.previous_hrir_fft.get(hrir_id) is None:
                self.previous_hrir_fft[hrir_id] = hrir_freq_rfft

        # === SAVE NEW OVERLAP (COMMON TO BOTH PATHS) ===
        overlap_buffer_batch_stereo[:, :, :overlap_size] = new_overlap
        return output_block.permute(0, 2, 1)

    
    def _read_delay_line(
            self,
            delays: torch.Tensor,
            gains: torch.Tensor,
            read_indices: torch.Tensor,
            read_indices_intermediate: torch.Tensor,
            delayed_signal_out: torch.Tensor,
            speaker_indices: torch.Tensor = None
    ):
        if speaker_indices is None:
            speaker_indices = self.speaker_indices_delay_read  # [N, 1, 1]

        # Use provided speaker indices or default to all
        if speaker_indices is None:
            speaker_indices = self.speaker_indices_delay_read

        torch.sub(read_indices, delays[:, None, :], out=read_indices_intermediate)
        read_indices_intermediate &= self.er_size_mask
        torch.mul(
            self.er_delay_line[speaker_indices, read_indices_intermediate],
            gains[:, None],
            out=delayed_signal_out
        )

    
    def _update_absorption_gains(self):
        self.rot60 = self.compute_rt60()

        # Calculate feedback based on RT60
        delay_samples = self.comb_delays_samples[:self.late_spatial_regions]

        # Handle the infinite RT60 case (0dB absorption) safely
        # if self.rot60 > 1000.0 or torch.isinf(self.rot60):
        #     # If RT60 is massive, default to max safe feedback
        #     feedback_gains = torch.full((self.late_spatial_regions,), 0.995, device=self.device, dtype=self.dtype)
        # else:
        feedback_gains = torch.pow(10.0, -3.0 * delay_samples / (self.rot60 * self.sample_rate))

        # CRITICAL FIX 1: Safety Clamp to 0.995 (approx 60s max tail) instead of 0.99999 (9 hours)
        # CRITICAL FIX 2: Update the TARGET buffer, not the active buffer
        self.target_comb_feedback[:, :self.late_spatial_regions] = torch.clip(feedback_gains, -0.995, 0.995)

    
    def _update_spatial_parameters(self):
        """Update reflection parameters - BATCHIFIED for num_speakers."""
        self.prev_reflection_delays.copy_(self.reflection_delays)
        self.prev_reflection_gains.copy_(self.reflection_gains)
        if self.absorption_gains_dirty:
            self._update_absorption_gains()
        if self.any_reflections_dirty:
            self._update_reflection_parameters(indexing_values=self.reflections_dirty)
        if self.head.head_rotation_dirty:
            self._update_hrirs()
        elif self.hrirs_dirty:
            self._update_hrirs()
        self._update_proximity_filter(indexing_values=self.proximity_dirty)
        self.proximity_dirty[:] = False
        # Remaining resets happen at the end of process call

    
    def _update_reflection_parameters(self, indexing_values: _indexing_type = slice(None, None)):
        # --- NEW: Store old values *before* they are recalculated ---
        self.prev_reflection_delays[indexing_values] = self.reflection_delays[indexing_values]
        self.prev_reflection_gains[indexing_values] = self.reflection_gains[indexing_values]
        speaker_positions = self.speaker_positions[indexing_values]
        sx, sy, sz = speaker_positions.T

        # Image source positions (already correct)
        image_pos_x = self.scaled_er_indices_w[None] - sx[:, None] * self.centered_er_indices_w[None]
        image_pos_y = self.scaled_er_indices_h[None] - sy[:, None] * self.centered_er_indices_h[None]
        image_pos_z = self.scaled_er_indices_d[None] - sz[:, None] * self.centered_er_indices_d[None]

        # Stack image positions: [N, num_images, 3]
        # Update reflection_positions for HRIR calculation
        self.reflection_positions[indexing_values, 0] = speaker_positions
        self.reflection_positions[indexing_values, 1:, 0] = image_pos_x
        self.reflection_positions[indexing_values, 1:, 1] = image_pos_y
        self.reflection_positions[indexing_values, 1:, 2] = image_pos_z

        # --- Calculate delays and gains (BATCHIFIED) ---
        distances = torch.linalg.norm(
            self.reflection_positions[indexing_values] - self.head.head_position.view(1, 1, 3), dim=2)
        self.reflection_delays[indexing_values] = (distances * (self.sample_rate / self.speed_of_sound)).clamp_(
            0.5, self.er_delay_line_size - self.block_size - 1
        ).long()

        # Descriptive all delays to prevent Aliasing
        self.reflection_gains[indexing_values] = self.pow_reflection_gain / distances.clamp_min_(self.min_distance)

    
    def _process_internal(self, input_block: torch.Tensor) -> torch.Tensor:
        """
        Internal processing function with parameter cross-fading.
        input_block: [N, block_size]
        Returns: [N, block_size, 2]
        """
        # --- PARAMETER HANDLING ---
        perform_spatial_crossfade = self.any_reflections_dirty
        if self.params_dirty:
            with self.memory_lock:
                self._update_spatial_parameters()
        self.comb_feedback.lerp_(self.target_comb_feedback, self.param_smoothing_factor)
        main_indices = self.er_write_head + self.block_indices
        read_indices = main_indices.view(1, -1, 1)
        # Flag to trigger one block of crossfading
        internal_input = input_block  # [N, block_size]

        # --- Write to BATCHIFIED delay line ---
        write_indices = (self.er_write_head + self.block_indices) & self.er_size_mask
        self.er_delay_line[:, write_indices] = internal_input

        # Clear BATCHIFIED buffers
        self.cluster_buffers.zero_()  # [N, num_clusters, block_size]
        self.stereo_output_buffer.zero_()  # [N, block_size, 2]
        stereo_output_internal = self.stereo_output_buffer

        # --- PRECISE REFLECTIONS (BATCHIFIED) ---
        num_reflections = self.num_precise_reflections

        # Get CURRENT parameters
        delay_CUR = self.reflection_delays[:, :num_reflections]  # [N, num_reflections]
        gain_CUR = self.reflection_gains[:, :num_reflections]  # [N, num_reflections]
        delayed_signal_CUR = self.delayed_signal_out[0]
        self._read_delay_line(
            delay_CUR, gain_CUR, read_indices=read_indices,
            read_indices_intermediate=self.read_indices_intermediate[0],
            delayed_signal_out=delayed_signal_CUR
        )

        if perform_spatial_crossfade:
            # Get PREVIOUS parameters
            delay_PREV = self.prev_reflection_delays[:, :num_reflections]
            gain_PREV = self.prev_reflection_gains[:, :num_reflections]

            # Calculate delayed signal with PREVIOUS parameters
            delayed_signal_PREV = self.delayed_signal_out[1]
            self._read_delay_line(
                delay_PREV, gain_PREV, read_indices=read_indices,
                read_indices_intermediate=self.read_indices_intermediate[1],
                delayed_signal_out=delayed_signal_PREV
            )

            # Crossfade: [N, block, num_ref] * [1, block, 1]
            delayed_signal = delayed_signal_PREV * self.param_fade_out_ramp + \
                             delayed_signal_CUR * self.param_fade_in_ramp
        else:
            delayed_signal = delayed_signal_CUR  # No crossfade needed

        # [N, num_reflections, block_size]
        delayed_signal_batch = delayed_signal.permute(0, 2, 1)

        # Get direct path for prox filter: [N, block_size]
        self.direct_path_signal[:] = delayed_signal_batch[:, 0]

        # --- Apply Prox Filter (BATCHIFIED) ---
        first_order_iir_cupy_batched(
            self.prox_filter_b0_cupy, self.prox_filter_b1_cupy, self.prox_filter_a1_cupy,
            self.direct_path_signal_cupy,
            self.prox_filter_zi_cupy,
            self.prox_filter_output_buffer_cupy,
            stream=self.cupy_stream
        )
        delayed_signal_batch[:, 0] = self.prox_filter_output_buffer

        # --- Convolve Precise Reflections (BATCHIFIED) ---
        B_precise = self.num_speakers * num_reflections
        signal_flat_precise = delayed_signal_batch.reshape(B_precise, self.block_size)
        hrir_flat_precise = self.precise_fft_hrirs.reshape(B_precise, 2, self.head.fft_size)
        overlap_flat_precise = self.precise_overlap.reshape(B_precise, 2, self.head.overlap_size)

        stereo_out = self._fft_convolve_overlap_add_stereo_batched(
            signal_flat_precise,
            hrir_flat_precise,
            overlap_flat_precise,
            self.input_freq_buf[0][:B_precise],
            self.output_freq_new_buf[0][:B_precise],
            self.output_time_new_buf[0][:B_precise],
            self.output_freq_old_buf[0][:B_precise],

            self.output_time_old_buf[0][:B_precise],
            self.crossfade1[0][:B_precise],
            self.crossfade2[0][:B_precise],

            hrir_id=0
        )  # Output [N*num_reflections, block_size, 2]

        stereo_output_internal.add_(
            stereo_out.reshape(self.num_speakers, num_reflections, self.block_size, 2).sum(dim=1)
        )

        # --- CLUSTER PROCESSING (BATCHIFIED) ---
        num_cluster_reflections = self.num_er_sources - self.num_precise_reflections
        if num_cluster_reflections > 0:
            cluster_indices = self.reflection_clusters[:,
                              self.num_precise_reflections:self.num_er_sources]  # [N, num_cluster]

            # Get CURRENT parameters
            delays_CUR = self.reflection_delays[:, self.num_precise_reflections:self.num_er_sources]  # [N, num_cluster]
            gains_CUR = self.reflection_gains[:, self.num_precise_reflections:self.num_er_sources]  # [N, num_cluster]

            # Calculate delayed signal with CURRENT parameters
            delayed_signals_CUR = self.delayed_signal_out[2]
            self._read_delay_line(
                delays_CUR, gains_CUR, read_indices=read_indices,
                read_indices_intermediate=self.read_indices_intermediate[2],
                delayed_signal_out=delayed_signals_CUR
            )

            if perform_spatial_crossfade:
                # Get PREVIOUS parameters
                delays_PREV = self.prev_reflection_delays[:, self.num_precise_reflections:self.num_er_sources]
                gains_PREV = self.prev_reflection_gains[:, self.num_precise_reflections:self.num_er_sources]

                # Calculate delayed signal with PREVIOUS parameters
                delayed_signals_PREV = self.delayed_signal_out[3]
                self._read_delay_line(
                    delays_PREV, gains_PREV, read_indices=read_indices,
                    read_indices_intermediate=self.read_indices_intermediate[3],
                    delayed_signal_out=delayed_signals_PREV
                )

                # Crossfade: [N, block, num_cluster] * [1, block, 1]
                delayed_signals = delayed_signals_PREV * self.param_fade_out_ramp + \
                                  delayed_signals_CUR * self.param_fade_in_ramp
            else:
                delayed_signals = delayed_signals_CUR  # No crossfade needed

            # --- BATCHIFIED index_add using scatter_add_ ---
            src = delayed_signals.permute(0, 2, 1)  # [N, num_cluster_reflections, block_size]
            index = cluster_indices[:, :, None]
            self.cluster_buffers.scatter_add_(1, index, src)

        # --- Convolve Clusters (BATCHIFIED) ---
        B_cluster = self.num_speakers * self.num_enhanced_clusters
        signal_flat_cluster = self.cluster_buffers.reshape(B_cluster, self.block_size)
        hrir_flat_cluster = self.cluster_fft_hrirs.repeat(self.num_speakers, 1, 1)
        overlap_flat_cluster = self.cluster_overlap.reshape(B_cluster, 2, self.head.overlap_size)

        stereo_out = self._fft_convolve_overlap_add_stereo_batched(
            signal_flat_cluster,
            hrir_flat_cluster,
            overlap_flat_cluster,
            self.input_freq_buf[1][:B_cluster],
            self.output_freq_new_buf[1][:B_cluster],
            self.output_time_new_buf[1][:B_cluster],
            self.output_freq_old_buf[1][:B_cluster],
            self.output_time_old_buf[1][:B_cluster],
            self.crossfade1[1][:B_cluster],
            self.crossfade2[1][:B_cluster],
            hrir_id=1
        )  # Output [N*num_clusters, block_size, 2]

        stereo_output_internal.add_(
            stereo_out.reshape(self.num_speakers, self.num_enhanced_clusters, self.block_size, 2).sum(dim=1)
        )

        # --- LATE REVERB (BATCHIFIED) ---
        torch.sum(self.cluster_buffers, dim=1, out=self.early_sum)
        # Pass the (un-smoothed) target feedback directly
        self.early_sum.add_(torch.sum(delayed_signal_batch, dim=1, out=self.early_sum_delayed_intermediate))
        multi_comb_filter_N_speakers_cupy(
            self.early_sum_cupy,
            self.comb_output_buffer_cupy,
            self.comb_packed_buffer_cupy,
            self.comb_buffer_offsets_cupy,
            self.comb_buffer_idx_cupy,
            self.comb_feedback_cupy,  # This now points to the un-smoothed buffer
            self.comb_delays_samples_cupy,
            self.total_comb_delays,
            stream=self.cupy_stream
        )

        allpass_filter_N_speakers_cupy(
            self.comb_output_buffer_cupy,
            self.allpass_temp_buffer_cupy,
            self.allpass_buffers_cupy[0],
            self.allpass_indices_cupy[0],
            self.allpass_gains_cupy[0],
            self.allpass_delays_samples[0],
            stream=self.cupy_stream
        )

        allpass_filter_N_speakers_cupy(
            self.allpass_temp_buffer_cupy,
            self.comb_output_buffer_cupy,
            self.allpass_buffers_cupy[1],
            self.allpass_indices_cupy[1],
            self.allpass_gains_cupy[1],
            self.allpass_delays_samples[1],
            stream=self.cupy_stream
        )

        # --- Convolve Late Reverb (BATCHIFIED) ---
        region_signal_batched = self.comb_output_buffer.mul_(self.late_weights.view(1, -1, 1)).mul_(
            self.late_reverb_mix)
        B_late = self.num_speakers * self.late_spatial_regions
        signal_flat_late = region_signal_batched.reshape(B_late, self.block_size)
        hrir_flat_late = self.late_fft_hrirs.repeat(self.num_speakers, 1, 1)
        overlap_flat_late = self.late_spatial_overlap.reshape(B_late, 2, self.head.overlap_size)

        stereo_out_batched = self._fft_convolve_overlap_add_stereo_batched(
            signal_flat_late,
            hrir_flat_late,
            overlap_flat_late,
            self.input_freq_buf[2][:B_late],
            self.output_freq_new_buf[2][:B_late],
            self.output_time_new_buf[2][:B_late],
            self.output_freq_old_buf[2][:B_late],

            self.output_time_old_buf[2][:B_late],
            self.crossfade1[2][:B_late],
            self.crossfade2[2][:B_late],

            hrir_id=2
        )  # Output [N*4, block_size, 2]

        stereo_output_internal.add_(
            torch.sum(
                stereo_out_batched.reshape(self.num_speakers, self.late_spatial_regions, self.block_size, 2), dim=1
            )
        )

        # --- FINAL ---
        self.er_write_head = (self.er_write_head + self.block_size) % self.er_delay_line_size
        if self.params_dirty:
            self.hrirs_dirty = False
            self.reflections_dirty[:] = False
            self.params_dirty = False
            self.absorption_gains_dirty = False
            self.head.head_rotation_dirty = False
            self.any_reflections_dirty = False
        return stereo_output_internal

    
    def _update_hrirs(self):
        # Honestly, HRIRs don't need masking, just update all at once
        # --- Get head-relative coordinates (BATCHIFIED) ---
        # Input [N, num_er_sources, 3], Output [N, num_er_sources, 3]
        head_coords_all = self.head.world_to_head_coordinates(self.reflection_positions)
        # Output [N, num_er_sources], [N, num_er_sources]
        azimuths, elevations = self.head.head_coords_to_cipic_angles(head_coords_all)

        # --- Get precise HRIRs (BATCHIFIED) ---
        # Input [N, num_reflections], Output [N, num_reflections, 2, fft_size]
        precise_hrirs = self.head.get_cached_fft_hrir(
            azimuths[:, :self.num_precise_reflections],
            elevations[:, :self.num_precise_reflections]
        )
        self.precise_fft_hrirs = precise_hrirs  # Assign directly
        self._update_reflection_clusters_(azimuths, elevations)
        # Mark all HRIRs as dirty when spatial parameters change
        for hrir_id in range(3):
            self.hrir_dirty_flags[hrir_id] = True

    
    def _update_reflection_clusters_(self, azimuths: torch.Tensor, elevations: torch.Tensor):
        cluster_az = self.enhanced_clusters[..., 0].view(1, 1, -1)  # [1, 1, num_clusters]
        cluster_el = self.enhanced_clusters[..., 1].view(1, 1, -1)  # [1, 1, num_clusters]

        # Compute azimuth differences with wraparound [N, num_er_sources, num_clusters]
        az_diff = torch.abs_(torch.sub(azimuths[:, :, None], cluster_az, out=make_shape(cluster_az, self.cluster_az_buf)))
        az_diff = torch.minimum(
            az_diff, torch.sub(
            360, az_diff, out=make_shape(az_diff, self.az_diff_buf)), out=make_shape(az_diff, self.az_diff_min_buf)
        )

        # Compute elevation differences [N, num_er_sources, num_clusters]
        el_diff = torch.abs_(torch.sub(elevations[:, :, None], cluster_el, out=make_shape(cluster_el, self.cluster_el_buf)))

        # Compute squared distances with elevation weighting
        sq_distances = torch.square_(az_diff).add_(
            torch.square_(el_diff).mul_(1.44))  # [N, num_er_sources, num_clusters]

        # Assign to nearest cluster [N, num_er_sources]
        self.reflection_clusters = torch.argmin(sq_distances, dim=2, out=self.reflection_clusters)
    
    def reset(self):
        """Reset all internal state (BATCHIFIED)."""
        self.er_delay_line.zero_()
        self.er_write_head = 0
        self.cluster_buffers.zero_()
        self.cluster_overlap.zero_()
        self.late_spatial_overlap.zero_()
        self.precise_overlap.zero_()
        self.prox_filter_zi_torch.zero_()
        self.comb_packed_buffer.zero_()
        self.comb_buffer_idx.zero_()
        self.comb_feedback.zero_()
        for i in range(self.num_allpass_filters):
            self.allpass_buffers[i].zero_()
            self.allpass_indices[i].zero_()
        self.proximity_dirty[:] = True
        self.reflections_dirty[:] = True
        self.any_reflections_dirty = True
        self.hrirs_dirty = True
        self.absorption_gains_dirty = True
        self.head.head_rotation_dirty = True

    def reset_nans(self):
        """
        Asynchronously checks for NaNs or Infs in state buffers and resets ONLY the corrupted speakers.
        Zero CPU-GPU synchronization (no blocking, no .item(), no .cpu()).
        """
        # 1. Initialize the Master Reset Mask (N,)
        # We start with false, and accumulate "True" if any buffer contains bad values.
        # This occurs entirely on the GPU.
        reset_mask = self.reset_mask
        reset_mask.zero_()

        # --- CHECKING PHASE (Accumulate "Badness") ---
        # We view everything as (N, -1) to perform a single reduction per buffer.

        # 1. Early Reflection Delay Line [N, Size]
        # Optimization: fusing isnan/isinf check
        reset_mask.logical_or_(self.er_delay_line.isnan().any(dim=1))
        reset_mask.logical_or_(self.er_delay_line.isinf().any(dim=1))

        # 2. Cluster Buffers [N, Clusters, Block]
        flat_cluster = self.cluster_buffers.view(self.num_speakers, -1)
        reset_mask.logical_or_(flat_cluster.isnan().any(dim=1))
        reset_mask.logical_or_(flat_cluster.isinf().any(dim=1))

        # 3. Cluster Overlap [N, Clusters, 2, Overlap]
        flat_cluster_over = self.cluster_overlap.view(self.num_speakers, -1)
        reset_mask.logical_or_(flat_cluster_over.isnan().any(dim=1))
        reset_mask.logical_or_(flat_cluster_over.isinf().any(dim=1))

        # 4. Late Spatial Overlap [N, Regions, 2, Overlap]
        flat_late_over = self.late_spatial_overlap.view(self.num_speakers, -1)
        reset_mask.logical_or_(flat_late_over.isnan().any(dim=1))
        reset_mask.logical_or_(flat_late_over.isinf().any(dim=1))

        # 5. Precise Overlap [N, Refs, 2, Overlap]
        flat_precise_over = self.precise_overlap.view(self.num_speakers, -1)
        reset_mask.logical_or_(flat_precise_over.isnan().any(dim=1))
        reset_mask.logical_or_(flat_precise_over.isinf().any(dim=1))

        # 6. Proximity Filter State [N]
        reset_mask.logical_or_(self.prox_filter_zi_torch.isnan())
        reset_mask.logical_or_(self.prox_filter_zi_torch.isinf())

        # 7. Comb Packed Buffer [N, TotalDelays]
        reset_mask.logical_or_(self.comb_packed_buffer.isnan().any(dim=1))
        reset_mask.logical_or_(self.comb_packed_buffer.isinf().any(dim=1))

        # 8. Comb Buffer Indices (Indices usually don't go NaN, but good to check bounds or corruption logic if float)
        # Skipped as they are int32

        # 9. Comb Feedback [N, Regions]
        reset_mask.logical_or_(self.comb_feedback.isnan().any(dim=1))
        reset_mask.logical_or_(self.comb_feedback.isinf().any(dim=1))

        # 10. Allpass Buffers (List of [N, Regions, Delays])
        for buf in self.allpass_buffers:
            flat_ap = buf.view(self.num_speakers, -1)
            reset_mask.logical_or_(flat_ap.isnan().any(dim=1))
            reset_mask.logical_or_(flat_ap.isinf().any(dim=1))

        # --- RESETTING PHASE (Masked Fill) ---
        # We execute these blindly. If reset_mask is all False, this is a no-op on GPU.
        # This prevents the need for an "if reset_mask.any():" check which would stall the GPU.

        # Prepare broadcast masks
        mask_2d = reset_mask.unsqueeze(1)
        mask_3d = reset_mask.view(-1, 1, 1)
        mask_4d = reset_mask.view(-1, 1, 1, 1)

        # 1. ER Delay Line
        self.er_delay_line.masked_fill_(mask_2d, 0)

        # 2. Cluster Buffers
        self.cluster_buffers.masked_fill_(mask_3d, 0)

        # 3. Cluster Overlap
        self.cluster_overlap.masked_fill_(mask_4d, 0)

        # 4. Late Spatial Overlap
        self.late_spatial_overlap.masked_fill_(mask_4d, 0)

        # 5. Precise Overlap
        self.precise_overlap.masked_fill_(mask_4d, 0)

        # 6. Proximity Filter State
        self.prox_filter_zi_torch.masked_fill_(reset_mask, 0)

        # 7. Comb Packed Buffer
        self.comb_packed_buffer.masked_fill_(mask_2d, 0)

        # 8. Comb Indices
        # We reset indices to 0 for consistency, though not strictly required for NaN removal
        self.comb_buffer_idx.masked_fill_(mask_2d, 0)

        # 9. Comb Feedback
        self.comb_feedback.masked_fill_(mask_2d, 0)

        # 10. Allpass Buffers & Indices
        for i in range(self.num_allpass_filters):
            # Buffer: [N, Regions, Delays]
            self.allpass_buffers[i].masked_fill_(mask_3d, 0)
            # Indices: [N, Regions]
            self.allpass_indices[i].masked_fill_(mask_2d, 0)

        # --- STATE FLAGGING ---
        # If we reset a speaker, we MUST force a parameter recalculation next frame
        # to ensure filters don't stay at 0 coefficients or weird states.
        self.proximity_dirty.masked_fill_(reset_mask, True)
        self.reflections_dirty.masked_fill_(reset_mask, True)

        # If any reset happened, trigger the global dirty flags to ensure flow control picks it up
        # We can't check 'if reset' without blocking, so we assume if we called this,
        # we want to ensure safety. Setting these booleans is cheap.
        # Note: We cannot conditionally set 'self.params_dirty = True' based on GPU tensor content
        # without syncing. However, setting the dirty tensors (above) is sufficient because
        # the engine checks those tensors in the update loop.

    
    def as_indices(self, indexing_values):
        if indexing_values is None:
            return self.speaker_indices
        if isinstance(indexing_values, torch.Tensor) and indexing_values.dtype == torch.bool:
            return indexing_values.nonzero(as_tuple=True)[0]
        if isinstance(indexing_values, torch.Tensor) and indexing_values.dtype in (
        torch.long, torch.int32, torch.int64, torch.int8):
            return indexing_values
        if isinstance(indexing_values, int):
            indexing_values = [indexing_values]
        return self.speaker_indices[indexing_values]

    
    def process(self, input_block: torch.Tensor) -> torch.Tensor:
        if input_block.shape != (self.num_speakers, self.block_size):
            raise ValueError(
                f"Input block shape mismatch. Expected {(self.num_speakers, self.block_size)}, got {input_block.shape}")
        return self._process_internal(input_block)

    
    def is_noticeable_change(self, positions1, positions2, threshold=0.001, buf_id=0):
        # Positions change far away are less noticeable than close by positions
        is_single = len(positions1.shape) == 1
        if is_single:
            positions1 = positions1[None]
            positions2 = positions2[None]
        assert positions1.shape == positions2.shape

        position_sub = self._position_sub_intermediate_buffer[buf_id][:positions1.shape[0]]
        change = self._position_change_intermediate_buffer[buf_id][:positions1.shape[0]]
        minimum = self._position_minimum_intermediate_buffer[buf_id][:positions1.shape[0]]
        thresh_buffer = self._position_thresh_intermediate_buffer[buf_id][:positions1.shape[0]]
        sum_buffer = self._position_sum_intermediate_buffer[buf_id][:positions1.shape[0]]
        output = self._position_greater_output_intermediate_buffer[buf_id][:positions1.shape[0]]
        torch.minimum(positions1, positions2, out=minimum)
        torch.sum(minimum, dim=-1, out=thresh_buffer).mul_(threshold)
        torch.greater(
            torch.sum(torch.abs(torch.sub(positions1, positions2, out=position_sub), out=change), out=sum_buffer,
                      dim=-1), thresh_buffer,
            out=output
        )
        assert len(output.shape) == 1 and output.shape[0] == positions1.shape[0]
        return output



    
    def set_head_position(self, position, should_update_params=True):
        # Only update reverb and hrir, parameters where significant head
        if should_update_params:
            # course_indices = self.is_noticeable_change(
            #     self.speaker_positions, self.speaker_positions
            #     + (self.head.head_position - self.last_head_positions), threshold=0.03, buf_id=0
            # )
            # self.last_head_positions[course_indices] = position
            self.reflections_dirty[:] = True
            # self.reflections_dirty[course_indices] = True
            self.hrirs_dirty = True
            self.params_dirty = True
        else:
            self.last_head_positions[:] = position
        self.head.head_position = position

    @profile
    def set_speaker_positions(self, positions, indexing_values: _indexing_type = None):
        # This alone uses a massive fraction of compute
        with self.memory_lock:
            if indexing_values is None:
                self.speaker_positions = positions
                indexing_values = slice(None, None, None)
                # available_positions = positions
            else:
                available_positions = positions[indexing_values]
                self.speaker_positions[indexing_values] = available_positions
        # available_indices = self.as_indices(indexing_values)
        # fine_mask = self.is_noticeable_change(
        #     available_positions, self.fine_speaker_positions[indexing_values], threshold=0.005, buf_id=1
        # )
        # course_mask = self.is_noticeable_change(
        #     available_positions, self.course_speaker_positions[indexing_values], threshold=0.05, buf_id=2
        # )
        # if len(available_positions.shape) == 1:
        #     available_positions = available_positions[None]
        # fine_indices = available_indices[fine_mask]
        # course_indices = available_indices[course_mask]
        # with self.memory_lock:
            # self.fine_speaker_positions[fine_indices] = available_positions[fine_mask]
            # self.course_speaker_positions[course_indices] = available_positions[course_mask]
            self.proximity_dirty[indexing_values] = True
            # self.proximity_dirty[course_indices] = True
            self.reflections_dirty[indexing_values] = True
            # self.reflections_dirty[fine_indices] = True
        self.hrirs_dirty = True
        self.any_reflections_dirty = True
        self.params_dirty = True


    def set_room_dimensions(self, w, h, d, should_update_params=True):
        """Set room dimensions (Global)."""
        # is_noticeable = self.is_noticeable_change(
        #     dimensions[None], self.room_dimensions[None], threshold=0.05, buf_id=3
        # )[0]
        # if is_noticeable:
        # Always update params regardless
        self.room_dimensions[0] = w
        self.room_dimensions[1] = h
        self.room_dimensions[2] = d
        self.half_dims = self.room_dimensions / 2.0
        self.offsets[0, 0] = -w
        self.offsets[1, 1] = -h
        self.offsets[2, 2] = -d
        self.offsets[3, 0] = w
        self.offsets[4, 1] = h
        self.offsets[5, 2] = d
        self.rot60 = self.compute_rt60()
        # --- Recalculate GLOBAL reflection params ---
        self.scaled_er_indices_w = self.er_indices[:, 0] * w
        self.scaled_er_indices_h = self.er_indices[:, 1] * h
        self.scaled_er_indices_d = self.er_indices[:, 2] * d
        self.centered_er_indices_w = (self.er_indices[:, 0] % 2 * 2 - 1)
        self.centered_er_indices_h = (self.er_indices[:, 1] % 2 * 2 - 1)
        self.centered_er_indices_d = (self.er_indices[:, 2] % 2 - 1)
        if should_update_params:
            with self.memory_lock:
                self.proximity_dirty[:] = True
            self.params_dirty = True

    
    def set_absorption(self, db_value: torch.tensor):
        """Set absorption coefficient in dB (Global)."""
        # ??
        # if torch.abs(self.absorption_db - db_value) > 0.01:
        self.absorption_db = db_value
        reflection_gain = 10 ** (self.absorption_db / 20.0)  # Global scalar
        self.pow_reflection_gain = torch.pow(reflection_gain, self.orders)
        self.absorption_gains_dirty = True
        with self.memory_lock:
            self.reflections_dirty[:] = True
        self.params_dirty = True
        self.any_reflections_dirty = True

    # kept for backwards compatibility
    def set_speaker_position_compat(self, speaker_index: int, x: float, y: float, z: float):
        """Set speaker position for a specific speaker index."""
        if not 0 <= speaker_index < self.num_speakers:
            raise ValueError(f"Invalid speaker_index {speaker_index}. Must be 0 <= index < {self.num_speakers}")

        margin = 0.1
        clamped_pos = torch.tensor([
            torch.clip(torch.as_tensor(x, device=self.device, dtype=self.dtype), -self.room_dimensions[0] / 2 + margin,
                       self.room_dimensions[0] / 2 - margin),
            torch.clip(torch.as_tensor(y, device=self.device, dtype=self.dtype), -self.room_dimensions[1] / 2 + margin,
                       self.room_dimensions[1] / 2 - margin),
            torch.clip(torch.as_tensor(z, device=self.device, dtype=self.dtype), -self.room_dimensions[2] / 2 + margin,
                       self.room_dimensions[2] / 2 - margin)
        ], device=self.device, dtype=self.dtype)
        self.compat_speaker_positions[speaker_index] = clamped_pos
        return self.set_speaker_positions(self.compat_speaker_positions, indexing_values=speaker_index)

    
    def set_room_size(self, w, h, d):
        return self.set_room_dimensions(w, h, d)

    # def estimate_peak_gain(self, head_position: torch.Tensor, speaker_positions: torch.Tensor) -> torch.Tensor:
    #     """
    #     Estimates the 'Perceptual Peak Gain' of the DSP engine for normalization.
    #     Includes Geometric Decay, Proximity Effect, Coherence Weighting, and
    #     Absorption-Curve Scaling.
    #     """
    #     # --- 1. Setup & Constants ---
    #     spk_pos = speaker_positions
    #     head_pos = head_position
    #
    #     W, H, D = self.room_dimensions
    #     half_dims = self.room_dimensions / 2.0
    #
    #     # --- 2. Direct Path (The Anchor) ---
    #     # Exact DSP distance clamping
    #     min_dist_allowed = self.mic_radius + self.speaker_radius
    #     delta_direct = spk_pos - head_pos
    #     dist_direct = torch.linalg.norm(delta_direct, dim=1)
    #     dist_direct = torch.clamp(dist_direct, min=min_dist_allowed)
    #
    #     # Geometric Gain (1/r)
    #     g_direct = 1.0 / dist_direct
    #
    #     # --- Proximity Effect (Exact DSP Match) ---
    #     walls_neg = half_dims - spk_pos
    #     walls_pos = half_dims + spk_pos
    #     all_walls = torch.cat([walls_neg, walls_pos], dim=1)
    #     min_wall_dist, _ = torch.min(all_walls, dim=1)
    #
    #     # Replicating the DSP shelf gain calculation
    #     norm_dist = torch.clamp(1.0 - (min_wall_dist / 1.5), 0.0, 1.0)
    #     shelf_gain_db = 0.5 * (norm_dist ** 2) * 6.0
    #     g_prox = (10 ** (shelf_gain_db / 20.0)) * 1.1  # 10% headroom for filter ringing
    #
    #     # Total Direct Amplitude (Linear)
    #     amp_direct = g_direct * g_prox
    #
    #     # Vectorized Image Sources [N, 6, 3]
    #     img_sources = (spk_pos.unsqueeze(1) * self.mirrors.unsqueeze(0)) + self.offsets.unsqueeze(0)
    #     d_refs = torch.linalg.norm(img_sources - head_pos.unsqueeze(1), dim=2)
    #     d_refs = torch.clamp(d_refs, min=0.1)
    #
    #     # Absorption
    #     absorb_linear = 10 ** (self.absorption_db / 20.0)
    #
    #     # Reflection Amplitude (1/r * absorb)
    #     amp_refs = (1.0 / d_refs) * absorb_linear
    #
    #     # RMS of Early Reflections (Energy Sum)
    #     rms_refs = torch.sqrt(torch.sum(amp_refs ** 2, dim=1))
    #
    #     # Cluster Contributions
    #     dist_cluster = dist_direct * 2.5
    #     amp_cluster = (1.0 / dist_cluster) * (absorb_linear ** 2) * self.num_enhanced_clusters
    #     rms_refs = torch.sqrt(rms_refs ** 2 + amp_cluster ** 2)
    #
    #     # --- 4. Late Reverb (Stability & Weighting) ---
    #     vol = W * H * D
    #     sa = 2 * (W * H + W * D + H * D)
    #     avg_absorb = torch.clamp(1.0 - absorb_linear, min=1e-3)
    #     rt60 = (0.161 * vol) / (sa * avg_absorb)
    #     rt60 = torch.clamp(rt60, min=0.05)
    #
    #     comb_delays_t = self.comb_delays_s[:4]
    #     gains_fb = 10 ** (-3.0 * comb_delays_t / rt60)
    #     gains_fb = torch.clamp(gains_fb, 0.0, 0.95)  # Stability clamp
    #
    #     # Geometric Series of Energy
    #     comb_energy_gain = torch.sum(1.0 / (1.0 - gains_fb ** 2))
    #
    #     # Input to reverb is roughly the Early Energy
    #     reverb_rms = rms_refs * torch.sqrt(comb_energy_gain) * self.late_reverb_mix
    #
    #     # --- 5. Coherence Weighted Combination ---
    #     # Direct (Transient) + Refs (Body) + Reverb (Tail)
    #     weighted_peak = (
    #             amp_direct * 1.0 +  # 100% Coherent
    #             rms_refs * 0.4 +  # 40% Coherent
    #             reverb_rms * 0.2  # 20% Coherent
    #     )
    #
    #     # HRIR Headroom (20%)
    #     final_gain = weighted_peak * 1.2
    #
    #     # --- 6. Final Absorption Weighting Curve ---
    #     # Modeled as Shifted Exponential Decay: y = C + A * e^(kx)
    #     # Fits: 0dB -> 1.8, -6dB -> 1.0, -20dB -> 0.985
    #     k_decay = 0.67
    #     asymptote = 0.985
    #     dynamic_range = 1.8 - asymptote
    #     curve_weighting = asymptote + dynamic_range * torch.exp(self.absorption_db * k_decay)
    #
    #     return torch.clamp(final_gain * curve_weighting, min=1e-3)

    @torch.inference_mode()
    def estimate_peak_gain(self, head_position: torch.Tensor, speaker_positions: torch.Tensor) -> torch.Tensor:
        """
        Estimates 'Perceptual Peak Gain' with Surgical Zero-Alloc Performance.
        """
        N = speaker_positions.shape[0]
        # --- 1. Direct Path (Anchor) ---
        torch.sub(speaker_positions, head_position, out=self.buf_delta[:N])

        torch.linalg.norm(self.buf_delta[:N], dim=1, out=self.buf_dist_direct[:N])

        min_dist = self.mic_radius + self.speaker_radius
        self.buf_dist_direct[:N].clamp_(min=min_dist)

        torch.reciprocal(self.buf_dist_direct[:N], out=self.buf_g_direct[:N])

        # --- 2. Proximity Effect ---
        torch.sub(self.half_dims, speaker_positions, out=self.buf_walls_neg[:N])
        torch.add(self.half_dims, speaker_positions, out=self.buf_walls_pos[:N])
        torch.min(self.buf_walls_neg[:N], self.buf_walls_pos[:N], out=self.buf_walls_neg[:N])

        # Min across axes
        torch.min(self.buf_walls_neg[:N], dim=1, out=(self.buf_min_wall[:N], self.buf_indices[:N]))

        # Shelf Gain
        torch.div(self.buf_min_wall[:N], 1.5, out=self.buf_prox_gain[:N])
        torch.sub(1.0, self.buf_prox_gain[:N], out=self.buf_prox_gain[:N])
        self.buf_prox_gain[:N].clamp_(0.0, 1.0)

        # 3.0 * norm_dist^2 logic
        self.buf_prox_gain[:N].pow_(2).mul_(3.0)

        # Convert dB to Linear
        self.buf_prox_gain[:N].div_(20.0)
        torch.pow(10.0, self.buf_prox_gain[:N], out=self.buf_prox_gain[:N])
        self.buf_prox_gain[:N].mul_(1.1)

        # Combine Direct + Prox
        torch.mul(self.buf_g_direct[:N], self.buf_prox_gain[:N], out=self.buf_amp_direct[:N])

        # --- 3. Image Sources ---
        # Broadcast and compute image sources
        torch.mul(speaker_positions.unsqueeze(1), self.mirrors, out=self.buf_img_src[:N])
        self.buf_img_src[:N].add_(self.offsets)

        self.buf_img_src[:N].sub_(head_position)
        torch.linalg.norm(self.buf_img_src[:N], dim=2, out=self.buf_d_refs[:N])
        self.buf_d_refs[:N].clamp_(min=0.1)

        # Absorption
        absorb_linear = torch.pow(10.0, self.absorption_db / 20.0)

        # Reflection Amplitudes
        torch.reciprocal(self.buf_d_refs[:N], out=self.buf_amp_refs[:N])
        self.buf_amp_refs[:N].mul_(absorb_linear)

        # RMS of Early Reflections
        self.buf_amp_refs[:N].pow_(2)
        torch.sum(self.buf_amp_refs[:N], dim=1, out=self.buf_rms_refs[:N])

        # --- Cluster Contributions ---
        # Transformation: 1/(d*2.5) == (1/d) * 0.4 == g_direct * 0.4
        torch.mul(self.buf_g_direct[:N], 0.4, out=self.buf_dist_sq[:N])
        self.buf_dist_sq[:N].mul_(absorb_linear ** 2)
        self.buf_dist_sq[:N].mul_(self.num_enhanced_clusters)

        # Add Cluster Energy to RMS Energy
        self.buf_dist_sq[:N].pow_(2)
        self.buf_rms_refs[:N].add_(self.buf_dist_sq[:N])

        # Final Sqrt for RMS
        self.buf_rms_refs[:N].sqrt_()

        # --- 4. Late Reverb ---
        # FIX: Replaced .item() calls with Tensor ops to prevent CPU Sync
        # We keep these as 0-dim tensors on GPU
        avg_absorb = (1.0 - absorb_linear).clamp(min=1e-3)

        # Vol/SA are usually python floats from room_dimensions, but dividing by avg_absorb (tensor)
        # creates a GPU tensor.
        W, H, D = self.room_dimensions[0], self.room_dimensions[1], self.room_dimensions[2]
        vol = W * H * D
        sa = 2 * (W * H + W * D + H * D)

        rt60 = (0.161 * vol) / (sa * avg_absorb)
        rt60 = rt60.clamp_(min=0.05)  # Tensor clamp, no CPU sync

        # Feedback Gains
        torch.mul(self.comb_delays_s[:4], -3.0, out=self.buf_reverb_scratch)
        self.buf_reverb_scratch.div_(rt60)
        torch.pow(10.0, self.buf_reverb_scratch, out=self.buf_reverb_scratch)
        self.buf_reverb_scratch.clamp_(0.0, 0.95)

        # Geometric Series
        self.buf_reverb_scratch.pow_(2)
        self.buf_reverb_scratch.neg_().add_(1.0)
        self.buf_reverb_scratch.reciprocal_()
        comb_energy_gain = self.buf_reverb_scratch.sum()

        reverb_scale = torch.sqrt(comb_energy_gain) * self.late_reverb_mix

        # --- 5. Coherence Weighted Combination ---
        combined_scale = 0.4 + (reverb_scale * 0.2)

        # Accumulate Reflections into Direct
        self.buf_amp_direct[:N].add_(self.buf_rms_refs[:N] * combined_scale)

        # HRIR Headroom
        self.buf_amp_direct[:N].mul_(1.2)

        # --- 6. Final Absorption Curve ---
        curve_weighting = self.asymptote + self.dynamic_range * torch.exp(self.absorption_db * self.k_decay)
        self.buf_amp_direct[:N].mul_(curve_weighting).mul_(4.22)
        self.buf_amp_direct[:N].clamp_(min=1e-3)
        return self.buf_amp_direct[:N]

class DopplerEffect:
    """
    Computes a per-sample, phase-accurate Doppler frequency scale ramp
    by smoothly interpolating the sound_scale parameter itself. (Torch version)

    Batched for N speakers.
    """
    def __init__(self,
                 sample_rate,
                 block_size,
                 num_speakers,  # N
                 speed_of_sound=343.0,
                 velocity_smoothing_factor=0.4,
                 dtype=torch.float32,
                 device='cuda'):
        """
        Initializes the Doppler effect processor for N speakers.
        """
        self.sample_rate = float(sample_rate)
        self.speed_of_sound = float(speed_of_sound)
        self.dtype = dtype
        self.device = device
        self.block_size = block_size
        self.num_speakers = num_speakers

        # --- Clamp to prevent unrealistic extremes ---
        self.motion_threshold = 1e-4

        # This is now per-speaker
        if not (0.0 < velocity_smoothing_factor <= 1.0):
            print("Warning: velocity_smoothing_factor must be (0, 1]. Defaulting to 0.3")
            self.velocity_smoothing_factor = 0.3
        else:
            self.velocity_smoothing_factor = velocity_smoothing_factor

        # --- State Variables (Batched) ---
        self.smoothed_radial_velocity = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)
        self.prev_speaker_pos = torch.zeros(self.num_speakers, 3, dtype=self.dtype, device=self.device)
        self.prev_speaker_pos[:, 2] = 15.0  # Default position

        # --- State Variables (Global Head) ---
        self.prev_head_pos = torch.tensor([0.0, 0.0, 0.0], dtype=self.dtype, device=self.device)

        # --- State Variables (Batched) ---
        self.prev_sound_scale = torch.ones(self.num_speakers, dtype=self.dtype, device=self.device)

        # --- Pre-allocated Buffers (Batched) ---
        self.zero_val = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)
        # Buffer for the linspace
        self.sound_scale_ramp_buffer = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype,
                                                   device=self.device)

        # Buffer for manual linspace ramp
        self.lerp_ramp = torch.linspace(0.0, 1.0, self.block_size, dtype=self.dtype, device=self.device)[
            None]  # (1, block_size)
        self.ones = torch.ones(self.num_speakers, dtype=self.dtype, device=self.device)

        # ===
        self.head_pos_exp = torch.empty(1, 3, dtype=self.dtype, device=self.device)
        self.prev_head_pos_exp = torch.empty(1, 3, dtype=self.dtype, device=self.device)
        self.diff_buffer_current = torch.empty(self.num_speakers, 3, dtype=self.dtype, device=self.device)
        self.diff_buffer_prev = torch.empty(self.num_speakers, 3, dtype=self.dtype, device=self.device)
        self.current_distance = torch.empty(self.num_speakers, dtype=self.dtype, device=self.device)
        self.prev_distance = torch.empty(self.num_speakers, dtype=self.dtype, device=self.device)
        self.distance_change = torch.empty(self.num_speakers, dtype=self.dtype, device=self.device)
        self.raw_radial_velocity = torch.empty(self.num_speakers, dtype=self.dtype, device=self.device)
        self.current_sound_scale = torch.empty(self.num_speakers, dtype=self.dtype, device=self.device)
        self.temp_velocity = torch.empty(self.num_speakers, dtype=self.dtype, device=self.device)

    
    def reset(self, prev_speaker_pos, prev_head_pos):
        """
        Instantly resets the 'previous' position state.
        speaker_pos_tensor: (N, 3)
        head_pos_tensor: (3)
        """
        self.prev_speaker_pos = prev_speaker_pos
        self.prev_head_pos = prev_head_pos
        self.prev_sound_scale.fill_(1.0)
        self.smoothed_radial_velocity.fill_(0.0)

    
    def reset_doppler(self, indexing_values=None):
        self.prev_sound_scale[indexing_values] = 1.0
        self.smoothed_radial_velocity[indexing_values] = 0.0
        self.sound_scale_ramp_buffer[indexing_values] = 0.0

    
    def process(self, current_speaker_pos_tensor, current_head_pos_tensor, frames):
        """
        Calculates a smoothly interpolated, per-sample frequency scale ramp.
        Batched for N speakers.
        current_speaker_pos_tensor: (N, 3)
        current_head_pos_tensor: (3)
        frames: int
        """
        # 1. Get current positions (already tensors)
        # Unsqueeze head pos for broadcasting: (1, 3)
        prev_head_pos_exp = self.prev_head_pos[None]
        torch.sub(current_speaker_pos_tensor, current_head_pos_tensor, out=self.diff_buffer_current)
        torch.linalg.norm(self.diff_buffer_current, dim=1, out=self.current_distance)

        # prev_distance: (N)
        torch.sub(self.prev_speaker_pos, prev_head_pos_exp, out=self.diff_buffer_prev)
        torch.linalg.norm(self.diff_buffer_prev, dim=1, out=self.prev_distance)

        # 3. Calculate the *raw* radial velocity over this block
        distance_change = self.current_distance.sub_(self.prev_distance)
        block_time = frames / self.sample_rate
        raw_radial_velocity = distance_change / block_time

        # 4. --- Smooth the velocity ---
        alpha = self.velocity_smoothing_factor

        # smoothed_radial_velocity: (N)
        # We update the full (N) tensor
        self.smoothed_radial_velocity = (
            (alpha * raw_radial_velocity) +
            (1.0 - alpha) * self.smoothed_radial_velocity
        )

        # 6. Calculate the 'current' target sound scale based on this velocity
        denominator = max(1e-6, self.speed_of_sound)
        current_sound_scale = 1.0 - (self.smoothed_radial_velocity * (1.0 / denominator))  # (N)
        # 7. --- INTERPOLATION (Manual batched linspace) ---
        start_scales = self.prev_sound_scale[:, None]  # (N, 1)
        end_scales = current_sound_scale[:, None]  # (N, 1)
        # Use pre-allocated ramp buffer, sliced to correct frame count
        ramp = self.lerp_ramp[:, :frames]  # (1, frames)
        # Get the (N, frames) view of the buffer
        scale_ramps_view = self.sound_scale_ramp_buffer[:, :frames]
        # Batched interpolation: (N, 1) + ((N, 1) - (N, 1)) * (1, frames) -> (N, frames)
        # Write the result directly into the buffer view
        # 1 fused kernel, 0 temporary tensors
        torch.lerp(start_scales, end_scales, ramp, out=scale_ramps_view)
        # 8. Update all state variables for the next block
        self.prev_speaker_pos = current_speaker_pos_tensor.clone()  # Update all N positions
        self.prev_head_pos = current_head_pos_tensor.clone()  # Head is global
        self.prev_sound_scale = current_sound_scale.clone()  # Store the 'end' scale for all N
        return self.sound_scale_ramp_buffer[:, :frames]  # Return view of the (N, frames) buffer

class DefaultCudaStream:
    pass

class RealtimeSpatialPulseEngine:
    
    def __init__(
        self,
        num_speakers=1,  # N
        sample_rate=44100,
        block_size=512,
        dtype=torch.float32,
        device='cuda',
        stream: Union[torch.cuda.Stream, DefaultCudaStream] = DefaultCudaStream(),
    ):
        self.min, self.max = 0, 0
        self.frame_count = 0
        self.num_speakers = num_speakers
        self.sample_rate, self.block_size = sample_rate, block_size
        self.dtype = dtype
        self.device = device
        self.gain_ramp_buf = torch.empty(self.block_size, dtype=self.dtype, device=self.device)
        # --- Batched State Variables (N) ---
        self.phase = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)
        self.vibrato_phase = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)
        self.pulse_phase = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)
        self.pulse_rate_hz = torch.full((self.num_speakers,), 1.0, dtype=self.dtype, device=self.device)
        self.pulse_decay_rate = torch.full((self.num_speakers,), 5.0, dtype=self.dtype, device=self.device)
        self.current_gains = torch.ones(self.num_speakers, dtype=self.dtype, device=self.device)
        self.linear_amps = torch.ones(self.num_speakers, dtype=self.dtype, device=self.device)
        # State variables are (N) tensors for in-place updates
        self.grid_y = torch.full((self.num_speakers,), 50.0, dtype=self.dtype, device=self.device)
        self.grid_x = torch.full((self.num_speakers,), 50.0, dtype=self.dtype, device=self.device)
        self.grid_x_changed_subbed = torch.empty((self.num_speakers,), dtype=self.dtype, device=self.device)
        self.stream = stream
        # Pass num_speakers, dtype and device to assumed-batched SpaceSimulationEngine
        self.space_effect = SpaceSimulationEngine(
            sample_rate=sample_rate,
            block_size=block_size,
            num_speakers=self.num_speakers,
            dtype=self.dtype,
            device=self.device,
            stream=self.stream
        )

        # --- Global Room Properties ---
        self.room_width, self.room_height, self.room_depth = self.space_effect.room_dimensions.clone()
        self.two_pi, self.sample_rate_inv = math.pi * 2.0, 1.0 / self.sample_rate
        self.master_gain = 1.0  # Audible default
        self.gain_linear = 10 ** (self.master_gain / 20.0)  # Global scalar
        self.previous_gain_linear = self.gain_linear

        # --- Torch-based Doppler Engine (Batched) ---
        self.doppler_engine = DopplerEffect(
            self.sample_rate,
            self.block_size,
            self.num_speakers,
            self.space_effect.speed_of_sound,
            dtype=self.dtype,
            device=self.device
        )
        self.doppler_engine.reset(self.space_effect.speaker_positions, torch.zeros(3, dtype=self.dtype, device=self.device))
        self._precompute_luts_and_constants()

        # --- Caching for parameter calculations (Batched) ---
        self.freq_motion_threshold = 0.01

        self.last_grid_y = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)
        self.cached_base_freq = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)

        self.last_grid_x = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)
        self.cached_vibrato_depth = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)

        self.last_base_freq_for_amp = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)
        self.cached_amplitude_gain = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)

        self.changed_mask = torch.empty(num_speakers, dtype=torch.bool, device=device)
        self.freq_diff = torch.empty(num_speakers, dtype=dtype, device=device)
        self.normalized_freq = torch.empty(num_speakers, dtype=dtype, device=device)
        self.lut_indices = torch.empty(num_speakers, dtype=torch.long, device=device)

        # Precompute LUT parameters for direct indexing
        self.lut_size = len(self.amplitude_lut)
        self.lut_freq_min = self.amplitude_lut_frequencies[0]
        self.lut_freq_max = self.amplitude_lut_frequencies[-1]
        self.lut_freq_range = self.lut_freq_max - self.lut_freq_min
        self.lut_size_minus_one = self.lut_size - 1

        # Pre-allocated buffers for _calculate_base_freq
        self.grid_y_diff = torch.empty(num_speakers, dtype=dtype, device=device)
        self.freq_changed_mask = torch.empty(num_speakers, dtype=torch.bool, device=device)
        self.normalized_y = torch.empty(num_speakers, dtype=dtype, device=device)
        self.freq_lut_indices = torch.empty(num_speakers, dtype=torch.long, device=device)

        # Precompute LUT parameters for direct indexing
        self.freq_lut_step_inv = (self.freq_lut_size - 1) / 100.0  # Inverse for multiplication instead of division

        # --- Parameter Interpolation State (Batched Tensors) ---
        self.prev_base_freq = self._calculate_base_freq()  # This will populate the cache (N)
        self.prev_vibrato_depth = self._calculate_vibrato_depth()  # Populates cache (N)
        self.prev_amplitude_gain = self._calculate_amplitude_gain(self.prev_base_freq)  # Populates cache (N)

        # --- Pre-allocate Buffers for process() (Batched) ---
        self.base_freq_ramp = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.amplitude_gain_ramp = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.vibrato_inc_tensor = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.vibrato_phase_signal = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype,
                                                device=self.device)
        self.vibrato_lfo = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.vibrato_depth_ramp = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.vibrato_modulation = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.final_freq_array = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.carrier_phase_incs = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.carrier_phase_signal = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype,
                                                device=self.device)
        self.carrier_signal = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.pulse_inc_tensor = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.pulse_phase_signal = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.saw_wave = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.pulse_env = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.dry_signal = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)
        self.signal = torch.empty(self.num_speakers, self.block_size, dtype=self.dtype, device=self.device)

        self.zero_val = torch.zeros(self.num_speakers, dtype=self.dtype, device=self.device)  # Batched

        # Buffer for manual linspace ramp
        self.lerp_ramp = torch.linspace(
            0.0, 1.0, self.block_size, dtype=self.dtype, device=self.device
        )[None]  # (1, block_size)

        # These are for backwards compatibility only
        self.compat_positions = torch.empty((self.num_speakers, 3), dtype=self.dtype, device=self.device)
        self.compat_rate_hz = torch.empty(self.num_speakers, dtype=self.dtype, device=self.device)
        self.compat_pulse_decay = torch.empty(self.num_speakers, dtype=self.dtype, device=self.device)

    @staticmethod
    def _erb_to_freq(erb):
        """Converts ERB scale to frequency (Hz)."""
        return (10.0 ** (erb / 21.4) - 1.0) / 0.00437

    def _torch_interp(self, x, xp, fp):
        """
        Pure PyTorch linear interpolation (1D) for CUDA tensors.
        x: query points
        xp: known x points (must be sorted)
        fp: known y points
        """
        # bucketize finds indices where elements should be inserted to maintain order
        indices = torch.bucketize(x, xp)
        # Clamp to valid range [1, len-1] for safe indexing
        indices = torch.clamp(indices, 1, len(xp) - 1)

        # Get values for left (lo) and right (hi) boundaries
        xp_lo = xp[indices - 1]
        xp_hi = xp[indices]
        fp_lo = fp[indices - 1]
        fp_hi = fp[indices]

        # Calculate interpolation weight (alpha)
        # Handle potential division by zero (though unlikely with proper frequency ranges)
        denom = xp_hi - xp_lo
        mask = denom == 0
        denom = torch.where(mask, torch.ones_like(denom), denom)

        alpha = (x - xp_lo) / denom

        # Interpolate
        return torch.lerp(fp_lo, fp_hi, alpha)

    def _perfectly_working_equal_loudness(self, freq_tensor):
        """
        Calculates perceptual gain using ISO 226:2003 (80 Phon) entirely in PyTorch/CUDA.
        Includes surgical fixes for specific loud/quiet zones.
        """
        device = freq_tensor.device
        dtype = freq_tensor.dtype

        # --- 1. ISO 226:2003 Data (Defined directly as GPU Tensors) ---
        f_iso = torch.tensor(
            [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
             2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500], device=device, dtype=dtype)
        alpha_iso = torch.tensor(
            [0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301, 0.288, 0.276, 0.267,
             0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301],
            device=device, dtype=dtype)
        LU_iso = torch.tensor(
            [-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5,
             0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1], device=device, dtype=dtype)
        Tf_iso = torch.tensor(
            [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5,
             1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3], device=device, dtype=dtype)

        # --- 2. Log-Space Interpolation (Pure Torch) ---
        log_f_iso = torch.log10(f_iso)
        # Clamp input frequencies to valid ISO range to prevent extrapolation errors
        log_f_in = torch.log10(torch.clamp(freq_tensor, 20.0, 12500.0))

        af = self._torch_interp(log_f_in, log_f_iso, alpha_iso)
        LU = self._torch_interp(log_f_in, log_f_iso, LU_iso)
        Tf = self._torch_interp(log_f_in, log_f_iso, Tf_iso)

        # --- 3. Calculate SPL for 80 Phon ---
        Phon = 80.0
        # Formula: A_f terms
        term1 = 4.47e-3 * (10 ** (0.025 * Phon) - 1.15)
        exponent_inner = ((Tf + LU) / 10.0) - 9.0
        term2 = torch.pow(0.4 * torch.pow(10.0, exponent_inner), af)
        Af = term1 + term2

        # Formula: Lp (Sound Pressure Level)
        Lp = (10.0 / af) * torch.log10(Af) - LU + 94.0

        # --- 4. Normalize (0dB Baseline) ---
        # 3.5kHz is naturally the loudest/most sensitive point. We subtract its level (~77dB)
        # so that it stays at Gain=1.0, and bass/treble get boosted.
        min_spl_iso = 77.6
        gain_db = Lp - min_spl_iso

        # --- 5. Surgical EQ Optimization (The Fixes) ---

        # 530Hz: 2dB boost (Range ~100Hz -> Width 50)
        gain_db += 5.0 * torch.exp(-((freq_tensor - 530.0) / 50.0) ** 2)

        # 1400Hz: 10dB CUT (Range ~1000Hz -> Width 500)
        # This fixes the "unfathomably loud" issue
        gain_db += -19.0 * torch.exp(-((freq_tensor - 1400.0) / 800.0) ** 2)

        # 2100Hz: 2dB boost (Range ~600Hz -> Width 300)
        gain_db += -4.5 * torch.exp(-((freq_tensor - 2200.0) / 400.0) ** 2)

        # 3900Hz: 6dB CUT (Range ~700Hz -> Width 350)
        # This fixes the "painfully loud" issue
        gain_db += -24.0 * torch.exp(-((freq_tensor - 3900.0) / 550.0) ** 2)

        # 30Hz: 6dB Boost (Bass Foundation) - Kept from previous request
        gain_db += 6.0 * torch.exp(-((freq_tensor - 30.0) / 10.0) ** 2)
        gain_db += -3.0 * torch.exp(-((freq_tensor - 2895.0) / 500.0) ** 2)
        gain_db += -3.0 * torch.exp(-((freq_tensor - 2895.0) / 500.0) ** 2)
        gain_db += -3.0 * torch.exp(-((freq_tensor - 1529.0) / 500.0) ** 2)
        gain_db += -3.0 * torch.exp(-((freq_tensor - 2700.0) / 500.0) ** 2)
        gain_db += -9.0 * torch.exp(-((freq_tensor - 3000.0) / 3000.0) ** 2)
        gain_db += -9.8 * torch.exp(-((freq_tensor - 4900.0) / 1000.0) ** 2)
        gain_db += 3.0 * torch.exp(-((freq_tensor - 1000.0) / 700.0) ** 2)

        # --- 6. Convert to Linear Gain ---
        gain_linear = torch.pow(10.0, gain_db / 20.0)

        # Apply safety scaler
        return gain_linear * 0.2


    
    def _perceptual_frequency_amplitude(self, input_tensor):
        """
        Maps a 0-100 input range to the perceptual frequency (ERB) scale.
        Assumes input_tensor is (N) or (N, frames).
        """
        normalized_input = input_tensor / 100.0
        # self.min_erb and self.max_erb are python floats, torch will promote
        erb_values = self.min_erb + normalized_input * (self.max_erb - self.min_erb)
        return RealtimeSpatialPulseEngine._erb_to_freq(erb_values)

    def _precompute_luts_and_constants(self):
        self.freq_lut_size, self.amp_lut_size = 4048 * 16, 4096 * 32

        # Calculate min/max ERB as python floats
        min_freq_float, max_freq_float = 20.0, min(20000.0, self.sample_rate / 2)
        self.min_erb = 21.4 * math.log10(1 + 0.00437 * min_freq_float)
        self.max_erb = 21.4 * math.log10(1 + 0.00437 * max_freq_float)

        # Precompute tensor for perfectly_working_equal_loudness
        self.min_freq_tensor = torch.tensor(min_freq_float, device=self.device, dtype=self.dtype)

        lut_linspace = torch.linspace(0, 100, self.freq_lut_size, device=self.device, dtype=self.dtype)
        self.frequency_lut = self._perceptual_frequency_amplitude(lut_linspace)

        # === FIX IS HERE ===
        # WAS: torch.logspace(...) -> Caused the 3700Hz -> 71Hz lookup error
        # NOW: torch.linspace(...) -> Matches the linear indexing in _calculate_amplitude_gain
        self.amplitude_lut_frequencies = torch.linspace(
            20.0,
            20000.0,
            self.amp_lut_size,
            device=self.device,
            dtype=self.dtype
        )
        # ===================

        self.amplitude_lut = self._perfectly_working_equal_loudness(self.amplitude_lut_frequencies)

        # Precompute xp for _calculate_base_freq
        self.freq_lut_xp = torch.linspace(0, 100, self.freq_lut_size, device=self.device, dtype=self.dtype)

        # --- Global Vibrato Parameters ---
        self.vibrato_rate, self.max_vibrato_q, self.vibrato_exp_factor = 30.0, 51.0, 2.3
        self.vibrato_exp_factor_norm = self.vibrato_exp_factor / self.max_vibrato_q
        self.vibrato_exp_norm = math.exp(2.3) - 1.0

    # def _precompute_luts_and_constants(self):
    #     self.freq_lut_size, self.amp_lut_size = 4048*16, 4096 * 32
    #
    #     # Calculate min/max ERB as python floats
    #     min_freq_float, max_freq_float = 20.0, min(20000.0, self.sample_rate / 2)
    #     self.min_erb = 21.4 * math.log10(1 + 0.00437 * min_freq_float)
    #     self.max_erb = 21.4 * math.log10(1 + 0.00437 * max_freq_float)
    #
    #     # Precompute tensor for perfectly_working_equal_loudness
    #     self.min_freq_tensor = torch.tensor(min_freq_float, device=self.device, dtype=self.dtype)
    #
    #     lut_linspace = torch.linspace(0, 100, self.freq_lut_size, device=self.device, dtype=self.dtype)
    #     self.frequency_lut = self._perceptual_frequency_amplitude(lut_linspace)
    #
    #     self.amplitude_lut_frequencies = torch.logspace(
    #         math.log10(20),
    #         math.log10(20000),
    #         self.amp_lut_size,
    #         device=self.device,
    #         dtype=self.dtype
    #     )
    #
    #     self.amplitude_lut = self._perfectly_working_equal_loudness(self.amplitude_lut_frequencies)
    #
    #     # Precompute xp for _calculate_base_freq
    #     self.freq_lut_xp = torch.linspace(0, 100, self.freq_lut_size, device=self.device, dtype=self.dtype)
    #
    #     # --- Global Vibrato Parameters ---
    #     self.vibrato_rate, self.max_vibrato_q, self.vibrato_exp_factor = 30.0, 51.0, 2.3
    #     self.vibrato_exp_factor_norm = self.vibrato_exp_factor / self.max_vibrato_q
    #     self.vibrato_exp_norm = math.exp(2.3) - 1.0

    
    def _calculate_base_freq(self):
        """Calculates the current target base frequency as an (N) tensor.
        Blazing fast: direct LUT indexing with no bounds checking."""

        # Check which grid_y values changed (reuse buffer)
        torch.sub(self.grid_y, self.last_grid_y, out=self.grid_y_diff)
        torch.abs(self.grid_y_diff, out=self.grid_y_diff)
        torch.gt(self.grid_y_diff, self.freq_motion_threshold, out=self.freq_changed_mask)

        changed_mask = self.freq_changed_mask.nonzero(as_tuple=True)[0]
        if changed_mask.numel() > 0:
            y = self.grid_y[changed_mask]

            # Direct conversion to LUT index: y * (lut_size-1) / 100
            # Using multiplication by inverse instead of division for speed
            torch.mul(y, self.freq_lut_step_inv, out=self.normalized_y[:changed_mask.numel()])

            # Direct cast to long for indexing (no rounding, just truncate)
            self.freq_lut_indices[:changed_mask.numel()] = self.normalized_y[:changed_mask.numel()].long()

            # Direct lookup - raw speed, no safety
            result = self.frequency_lut[self.freq_lut_indices[:changed_mask.numel()]]

            # Fast masked cache update
            self.last_grid_y[changed_mask] = y
            self.cached_base_freq[changed_mask] = result
        return self.cached_base_freq

    
    def _calculate_vibrato_depth(self):
        """Calculates the current target vibrato depth as an (N) tensor."""
        grid_x_changed = self.grid_x
        vibrato_input_norm = torch.abs_(torch.sub(grid_x_changed, 50.0, out=self.grid_x_changed_subbed)).mul_(0.02)
        q_val = vibrato_input_norm.mul_(self.max_vibrato_q)
        return q_val.mul_(self.vibrato_exp_factor_norm).exp_().add_(-1.0).mul_(1.0 / self.vibrato_exp_norm)

    
    def _calculate_amplitude_gain(self, base_freq):
        """Calculates the target amplitude gain based on frequency (N) tensor.
        Blazing fast: direct LUT indexing with no bounds checking or interpolation."""

        # Check which frequencies changed (reuse buffer)
        torch.sub(base_freq, self.last_base_freq_for_amp, out=self.freq_diff)
        torch.abs(self.freq_diff, out=self.freq_diff)
        torch.ge(self.freq_diff, self.freq_motion_threshold, out=self.changed_mask)

        changed_indices = self.changed_mask.nonzero(as_tuple=True)[0]

        if changed_indices.numel() > 0:
            # Update cache for changed frequencies
            self.last_base_freq_for_amp[changed_indices] = base_freq[changed_indices]

            # Direct LUT indexing: normalize to [0, lut_size-1] and cast to int
            # freq -> (freq - min) / range * (size - 1)
            freq_subset = base_freq[changed_indices]

            # Normalize to [0, 1] range
            torch.sub(freq_subset, self.lut_freq_min, out=self.normalized_freq[:changed_indices.numel()])
            torch.div(self.normalized_freq[:changed_indices.numel()],
                      self.lut_freq_range,
                      out=self.normalized_freq[:changed_indices.numel()])

            # Scale to [0, lut_size-1] and convert to int indices
            torch.mul(self.normalized_freq[:changed_indices.numel()],
                      self.lut_size_minus_one,
                      out=self.normalized_freq[:changed_indices.numel()])

            # Direct cast to long for indexing (no rounding, just truncate)
            self.lut_indices[:changed_indices.numel()] = self.normalized_freq[:changed_indices.numel()].long()

            # Direct lookup - raw speed, no safety
            self.cached_amplitude_gain[changed_indices] = self.amplitude_lut[self.lut_indices[:changed_indices.numel()]]

        return self.cached_amplitude_gain

    
    def _process(self, frames, reset_index_values=None):
        """
        Generates one block of audio for N speakers.
        Returns (N, 2, frames) numpy array.
        """
        # --- Create Sliced Views of Buffers ---
        # These are all (N, frames) views
        b_freq_ramp = self.base_freq_ramp[:, :frames]
        b_amp_ramp = self.amplitude_gain_ramp[:, :frames]
        b_vib_inc = self.vibrato_inc_tensor[:, :frames]
        b_vib_phase = self.vibrato_phase_signal[:, :frames]
        b_vib_lfo = self.vibrato_lfo[:, :frames]
        b_vib_depth_ramp = self.vibrato_depth_ramp[:, :frames]
        b_vib_mod = self.vibrato_modulation[:, :frames]
        b_final_freq = self.final_freq_array[:, :frames]
        b_carr_inc = self.carrier_phase_incs[:, :frames]
        b_carr_phase = self.carrier_phase_signal[:, :frames]
        b_carr_sig = self.carrier_signal[:, :frames]
        b_pulse_inc = self.pulse_inc_tensor[:, :frames]
        b_pulse_phase = self.pulse_phase_signal[:, :frames]
        b_saw = self.saw_wave[:, :frames]
        b_pulse_env = self.pulse_env[:, :frames]
        b_dry_sig = self.dry_signal[:, :frames]
        b_signal = self.signal[:, :frames]

        # Re-usable ramp (1, frames)
        ramp = self.lerp_ramp[:, :frames]

        # --- 1. Get Base Frequency (RAMP) ---
        current_base_freq = self._calculate_base_freq()  # (N)
        if reset_index_values is not None:
            self.prev_base_freq[reset_index_values] = current_base_freq[reset_index_values]
        start_freq = self.prev_base_freq[:, None]  # (N, 1)
        end_freq = current_base_freq[:, None]  # (N, 1)
        torch.lerp(start_freq, end_freq, ramp, out=b_freq_ramp)
        self.prev_base_freq = current_base_freq
        # --- 2. Get Amplitude Gain (RAMP) ---
        current_amplitude_gain = (
            self._calculate_amplitude_gain(current_base_freq) * self.gain_linear * self.linear_amps
        ) # (N)
        if reset_index_values is not None:
            self.prev_amplitude_gain[reset_index_values] = current_amplitude_gain[reset_index_values]
        start_amp = self.prev_amplitude_gain[:, None]  # (N, 1)
        end_amp = current_amplitude_gain[:, None]  # (N, 1)
        torch.lerp(start_amp, end_amp, ramp, out=b_amp_ramp)
        self.prev_amplitude_gain = current_amplitude_gain

        # --- 3. Calculate Vibrato LFO (Array) ---
        vibrato_inc = self.vibrato_rate * self.two_pi * self.sample_rate_inv  # (scalar)
        b_vib_inc.fill_(vibrato_inc)

        torch.cumsum(b_vib_inc, dim=1, out=b_vib_phase)
        b_vib_phase.add_(self.vibrato_phase[:, None])  # (N, frames) + (N, 1)

        self.vibrato_phase = b_vib_phase[:, -1].clone()
        torch.sin(b_vib_phase, out=b_vib_lfo)

        # --- 4. Get Vibrato Depth (RAMP) ---
        current_vibrato_depth = self._calculate_vibrato_depth()  # (N)
        if reset_index_values is not None:
            self.prev_vibrato_depth[reset_index_values] = current_vibrato_depth[reset_index_values]
        start_depth = self.prev_vibrato_depth[..., None]  # (N, 1)
        end_depth = current_vibrato_depth[..., None]  # (N, 1)
        torch.lerp(start_depth, end_depth, ramp, out=b_vib_depth_ramp)

        self.prev_vibrato_depth = current_vibrato_depth

        torch.mul(b_vib_depth_ramp, b_vib_lfo, out=b_vib_mod)
        b_vib_mod.add_(1.0)  # In-place
        # --- 5. Calculate Phase-Perfect Doppler Ramp (Array) ---
        # doppler_ramp = self.doppler_engine.process(  # This returns (N, frames) view
        #     self.space_effect.speaker_positions,
        #     self.space_effect.head.head_position,
        #     frames
        # )
        # --- 6. Compute Final Carrier Frequency (All arrays) ---
        torch.mul(b_freq_ramp, b_vib_mod, out=b_final_freq)
        # if doppler_ramp is not None:
        #     if reset_index_values is not None:
        #         doppler_ramp[reset_index_values] = 1.0
        #         self.doppler_engine.reset_doppler(reset_index_values)
        #     b_final_freq.mul_(doppler_ramp)  # In-place

        # --- 7. Generate Carrier Signal (Torch) ---
        torch.mul(b_final_freq, self.two_pi * self.sample_rate_inv, out=b_carr_inc)
        torch.cumsum(b_carr_inc, dim=1, out=b_carr_phase)
        b_carr_phase.add_(self.phase[:, None])  # (N, frames) + (N, 1)

        self.phase = b_carr_phase[:, -1].clone()
        self.phase %= self.two_pi

        torch.sin(b_carr_phase, out=b_carr_sig)

        # --- 8. Generate Pulse Envelope (Torch) ---
        pulse_inc = self.pulse_rate_hz * self.two_pi * self.sample_rate_inv  # (N)
        b_pulse_inc.copy_(pulse_inc[:, None])  # (N, frames) filled with (N, 1)

        torch.cumsum(b_pulse_inc, dim=1, out=b_pulse_phase)
        b_pulse_phase.add_(self.pulse_phase[:, None])  # (N, frames) + (N, 1)
        torch.remainder(b_pulse_phase[:, -1], self.two_pi, out=self.pulse_phase)

        # --- New Sawtooth Algorithm (In-Place, Batched) ---
        torch.remainder(b_pulse_phase, self.two_pi, out=b_saw)
        b_saw.div_(self.two_pi)
        torch.sub(1.0, b_saw, out=b_pulse_env)
        b_pulse_env.mul_(100.0)
        torch.minimum(b_saw, b_pulse_env, out=b_saw)

        # pulse_decay_rate is (N), needs (N, 1) for broadcasting
        torch.mul(b_saw, -self.pulse_decay_rate[:, None], out=b_pulse_env)
        torch.exp(b_pulse_env, out=b_pulse_env)
        # --- End New Sawtooth Algorithm ---

        # --- 9. Create Dry Signal & Apply Gains (Torch) ---
        torch.mul(b_carr_sig, b_pulse_env, out=b_dry_sig)
        torch.mul(b_dry_sig, b_amp_ramp, out=b_signal)
        self.current_gains = b_pulse_env.mean(dim=1)

        # --- 10. Process through Spatializer (Torch) ---
        # Assuming .process takes (N, frames) and returns (N, frames, 2)
        spatial_output = self.space_effect.process(b_signal)
        # spatial_output = b_signal[..., None].repeat(1, 1, 2)
        # min_ = spatial_output.min()
        # max_ = spatial_output.max()
        # if min_ < self.min:
        #     self.min = min_
        # else:
        #     self.min *= 0.992
        # if max_ > self.max:
        #     self.max = max_
        # else:
        #     self.max *= 0.992
        # if self.frame_count % 40 == 0:
            # print(f"max={self.max}, position={self.space_effect.speaker_positions[0]}, head_position={self.space_effect.head.head_position} "
            #       f"absorption_db={self.space_effect.absorption_db}")
        self.frame_count += 1
        # --- 11. Apply Master Gain (Torch) ---
        if abs(self.previous_gain_linear - self.gain_linear) > 0.001:
            torch.linspace(
                self.previous_gain_linear, self.gain_linear, spatial_output.shape[1],
                device=self.device, dtype=self.dtype, out=self.gain_ramp_buf[:spatial_output.shape[1]]
            )
            spatial_output[:] *= self.gain_ramp_buf[..., None]
        else:
            spatial_output.mul_(self.gain_linear)

        # This is not ramped, for performance benefits
        self.previous_gain_linear = self.gain_linear
        # Return (N, frames, 2)
        return spatial_output

    # Usable realtime public functions
    # @check_nans(print_values=True)
    def process(self, frames, reset_index_values=None):
        output = self._process(frames, reset_index_values=reset_index_values)
        return output
    
    def set_head_rotation(self, rotation_matrix):
        """Set the head rotation matrix from the tracker (Global)."""
        self.space_effect.head.set_head_rotation(rotation_matrix)

    
    def set_master_gain(self, gain):
        """Set global master gain."""
        self.master_gain = gain
        self.gain_linear = 10 ** (gain / 20.0)  # Global scalar

    def set_amplitudes(self, linear_amps, indexing_values: _indexing_type = None):
        if indexing_values is None:
            self.linear_amps = linear_amps
        else:
            self.linear_amps[indexing_values] = linear_amps[indexing_values]

    def set_pulse_rates(self, rate_hz, indexing_values=None):
        if indexing_values is None:
            self.pulse_rate_hz = rate_hz
        else:
            self.pulse_rate_hz[indexing_values] = rate_hz[indexing_values]
    
    def set_pulse_decays(self, rates, indexing_values=None):
        if indexing_values is None:
            self.pulse_decay_rate = rates
        else:
            self.pulse_decay_rate[indexing_values] = rates[indexing_values]
    
    def set_frequency_targets(self, coords, indexing_values=None):
        """Set frequency target for a specific speaker."""
        if indexing_values is None:
            self.grid_x = coords[..., 0]
            self.grid_y = coords[..., 1]
        else:
            self.grid_x[indexing_values] = coords[indexing_values, 0]
            self.grid_y[indexing_values] = coords[indexing_values, 1]

    def set_speaker_positions(self, positions, indexing_values=None):
        """Set position for a specific speaker."""
        self.space_effect.set_speaker_positions(
            positions=positions, indexing_values=indexing_values
        )

    def set_head_position(self, position, should_update_params=True):
        self.space_effect.set_head_position(position, should_update_params=should_update_params)
    
    def set_room_dimensions(self, w, h, d, should_update_params=True):
        """Set global room dimensions."""
        self.room_width, self.room_height, self.room_depth = w, h, d
        self.space_effect.set_room_dimensions(w, h, d, should_update_params=should_update_params)

    def set_wall_absorbance(self, db):
        """Set global wall absorbance."""
        self.space_effect.set_absorption(db)

    # Here for backwards compatibility
    def set_frequency_target(self, speaker_index, x, y):
        """Set frequency target for a specific speaker."""
        self.grid_x[speaker_index] = x
        self.grid_y[speaker_index] = y

    
    def set_speaker_position(self, speaker_index, x_percent, y_percent, z_percent):
        """Set position for a specific speaker."""
        room_x = (x_percent / 50.0 - 1.0) * (self.room_width / 2.0)
        room_y = (y_percent / 50.0 - 1.0) * (self.room_height / 2.0)
        room_z = (z_percent / 50.0 - 1.0) * (self.room_depth / 2.0)
        # positions = torch.cat([room_x[..., None], room_y[..., None], room_z[..., None]], dim=-1)
        # if len(positions.shape) == 1:
        #     positions = positions[None]
        # print(positions.shape, "positions")
        # amp_values_norm = self.space_effect.estimate_peak_gain(
        #     torch.zeros([3], dtype=torch.float32, device='cuda'),
        #     positions,
        #     # absorption_db=self.space_effect.absorption_db
        # )
        # print(1/amp_values_norm, "1/amp_values_norm")
        # self.set_amplitudes(1 / amp_values_norm)
        self.space_effect.set_speaker_position_compat(speaker_index, room_x, room_y, room_z)

    
    def set_head_position_percent(self, x_percent, y_percent, z_percent):
        """Set position for the (global) head."""
        room_x = (x_percent / 50.0 - 1.0) * (self.room_width / 2.0)
        room_y = (y_percent / 50.0 - 1.0) * (self.room_height / 2.0)
        room_z = (z_percent / 50.0 - 1.0) * (self.room_depth / 2.0)
        self.set_head_position(
            torch.tensor([room_x, room_y, room_z], dtype=self.dtype, device=self.device)
        )

    
    def set_pulse_rate(self, speaker_index, rate_hz):
        """Set pulse rate for a specific speaker."""
        self.compat_rate_hz[speaker_index] = rate_hz
        return self.set_pulse_rates(self.compat_rate_hz, indexing_values=speaker_index)

    
    def set_pulse_decay(self, speaker_index, delay):
        """Set pulse decay for a specific speaker."""
        self.compat_pulse_decay[speaker_index] = delay
        return self.set_pulse_decays(self.compat_pulse_decay, indexing_values=speaker_index)

    def reset_nans(self):
        self.space_effect.reset_nans()
