import time
import torch
import orjson
import socket
import threading
import numpy as np
from filters import HighPassFilter
from vsa_visualizer import VSAVisualizer

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
DEBUG_IGNORE_SENSOR = False

LAST_KEY = None
def no_repeat_print(*args, **kwargs):
    key = str(args)
    global LAST_KEY
    if key != LAST_KEY:
        print(*args, **kwargs)
    LAST_KEY = key

def make_binary(x):
    return np.where(x < 0, -1, 1)

# Use float64 numpy
class CircularTimeBuffer:
    """Shared time buffer for all circular arrays"""

    def __init__(self, size=20000):
        self.size = size
        self.times = np.zeros(size, dtype=np.float64)
        self.idx = 0
        self.filled = False

    def add(self, time_s):
        """Add timestamp and return current index"""
        self.times[self.idx] = time_s
        current = self.idx
        self.idx = (self.idx + 1) % self.size
        if self.idx == 0:
            self.filled = True
        return current

    def find_indices(self, t_query):
        """Find interpolation indices for given time. Returns (i0, i1, alpha)"""
        if not self.filled and self.idx == 0:
            return 0, 0, 0.0

        # Get valid range
        if self.filled:
            n_valid = self.size
            # Oldest data starts at self.idx, newest at self.idx-1
            start_idx = self.idx
        else:
            n_valid = self.idx
            start_idx = 0

        if n_valid == 0:
            return 0, 0, 0.0

        if n_valid == 1:
            return start_idx, start_idx, 0.0

        # Find time bounds
        t_start = self.times[start_idx]
        t_end = self.times[(start_idx + n_valid - 1) % self.size]

        # Clamp to valid range
        if t_query <= t_start:
            return start_idx, start_idx, 0.0
        if t_query >= t_end:
            last_idx = (start_idx + n_valid - 1) % self.size
            return last_idx, last_idx, 0.0

        # Binary search in circular buffer
        left, right = 0, n_valid - 1
        while right - left > 1:
            mid = (left + right) // 2
            mid_idx = (start_idx + mid) % self.size
            if self.times[mid_idx] < t_query:
                left = mid
            else:
                right = mid

        # Convert to circular indices
        i0 = (start_idx + left) % self.size
        i1 = (start_idx + right) % self.size

        # Compute interpolation factor
        t0 = self.times[i0]
        t1 = self.times[i1]
        alpha = (t_query - t0) / (t1 - t0) if t1 > t0 else 0.0
        return i0, i1, alpha

class CircularRotationBuffer:
    """Circular buffer for rotation matrices with SLERP interpolation"""

    def __init__(self, time_buffer):
        self.time_buffer = time_buffer
        self.size = time_buffer.size
        # Store as quaternions for efficient SLERP
        self.large_tensor = np.zeros((self.size, 3, 3), dtype=np.float32)
        self.large_tensor[0] = np.eye(3, dtype=np.float32)  # Initialize to identity (w=1, x=y=z=0)

    def add(self, rotation_matrix, time_s):
        """Add rotation matrix at given time"""
        # Optional: Validate monotonic time
        if self.time_buffer.idx > 0 or self.time_buffer.filled:
            last_idx = (self.time_buffer.idx - 1) % self.size
            if time_s <= self.time_buffer.times[last_idx]:
                # if abs(time_s - self.time_buffer.times[last_idx]) < 0.0001:
                print(f"WARNING: Timestamp {time_s} is not after last timestamp {self.time_buffer.times[last_idx]}")
                return

        idx = self.time_buffer.add(time_s)
        if isinstance(rotation_matrix, np.ndarray):
            tensor = rotation_matrix.astype(self.large_tensor.dtype)
        else:
            tensor = rotation_matrix.astype(self.large_tensor.dtype)

        self.large_tensor[idx] = tensor

    @staticmethod
    def _matrix_to_quat(R):
        """Robust rotation matrix to quaternion (w, x, y, z)"""
        trace = np.trace(R)

        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])

    @staticmethod
    def _quat_to_matrix(q):
        """Fast quaternion to rotation matrix (q = [w, x, y, z])"""
        w, x, y, z = q

        # Precompute repeated terms
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])

    @staticmethod
    def _slerp(q0, q1, t):
        """Spherical linear interpolation between quaternions"""
        # Normalize inputs
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)

        dot = np.dot(q0, q1)

        # Ensure the shortest path (avoid long way around sphere)
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        # Clamp for numerical stability
        dot = np.clip(dot, -1.0, 1.0)

        # For very close quaternions, use linear interpolation
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)

        # Standard SLERP
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q0 + s1 * q1

    def at(self, t_query):
        """Get rotation matrix at time t_query with SLERP interpolation"""
        i0, i1, alpha = self.time_buffer.find_indices(t_query)
        if i0 == i1 or alpha == 0.0:
            return self.large_tensor[i0]
        return self.large_tensor[i0]

class CircularPositionBuffer:
    """Circular buffer for 3D positions with linear interpolation"""

    def __init__(self, time_buffer):
        self.time_buffer = time_buffer
        self.size = time_buffer.size
        self.positions = np.zeros((self.size, 3), dtype=np.float32)

    def add(self, position, time_s):
        """Add position at given time"""
        idx = self.time_buffer.add(time_s)
        if isinstance(position, np.ndarray):
            tensor = position.astype(self.positions.dtype)
        else:
            tensor = position.astype(self.positions.dtype)

        self.positions[idx] = tensor

    def at(self, t_query):
        """Get position at time t_query with linear interpolation"""
        i0, i1, alpha = self.time_buffer.find_indices(t_query)

        if i0 == i1 or alpha == 0.0:
            return self.positions[i0]

        # Linear interpolation
        return (1 - alpha) * self.positions[i0] + alpha * self.positions[i1]

class IMUSensing:
    def __init__(self, *, parent=None, verbose=False, dtype=torch.float32, device='cuda',
                 host="0.0.0.0", port=8090, mode="portrait"):  # Added host/port for UDP
        print(f"Sensing on Mode | {mode}")
        self.parent = parent
        self.failed_count = 0
        self.running = True
        self.rotation_matrix = np.eye(3, dtype=np.float64)
        self.verbose = verbose
        self.dtype = dtype
        self.closed = False
        self.device = device
        self.mode = mode
        self.pinned_rotation = torch.zeros((3, 3), dtype=dtype, device='cpu').pin_memory()
        self.on_device_rotation_buffer = torch.zeros((3, 3), dtype=dtype, device=device)
        self.cpu_pinned = [
            torch.eye(3, device='cpu').pin_memory() for _ in range(4)
        ]

        # 2. GPU BUFFERS (The "Storage" Area)
        # Buffer 0: Currently being read by DSP
        # Buffer 1: Being updated by CPU
        self.gpu_buffers = [
            torch.eye(3, device=device) for _ in range(4)
        ]

        # 0 or 1
        self.idx = 0
        self.latest_sensor_timestamp_ns = 0

        # --- UDP Socket Setup ---
        print(f"IMU: Binding to UDP socket at {host}:{port}")
        self.listen_port = port
        self.listen_host = host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_host, self.listen_port))
        self.url_info = (self.listen_host, self.listen_port)
        self.sock.setblocking(False)  # CRITICAL: Set to non-blocking
        print("IMU: Socket bound successfully.")

        # Raw sensor data
        self._raw_accel_data = None
        self._raw_rotation_data = None
        # self.data_lock = threading.Lock() # Removed

        # Processed data timestamps
        self.last_processed_accel_timestamp_ns = 0
        self.last_processed_rotation_timestamp_ns = 0

        # Tracking data
        self.readings = {}
        # self.readings_lock = threading.Lock() # Removed

        # --- CONSTANTS ---
        self.NS_TO_S = 1_000_000_000.0
        # Android (ENU) world frame: Gravity is 9.80665 m/s^2 on the Z-axis
        self.GRAVITY_WORLD = np.array([0, 0, 9.80665], dtype=np.float64)

        # --- ZUPT (Zero-Velocity Update) State ---
        self.stationary_frames = 0
        self.is_stationary = False
        self.STATIONARY_FRAMES_MIN = 10
        self.STATIONARY_THRESHOLD_MS2 = 0.05

        # --- Automatic Bias Calibration ---
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.bias_calibration_buffer = []
        self.BIAS_CALIBRATION_FRAMES = 100
        self.rotation_time_buffer = CircularTimeBuffer(size=20000)
        self.rotation_buffer = CircularRotationBuffer(self.rotation_time_buffer)
        self.accel_time_buffer = CircularTimeBuffer(size=20000)
        self.current_rotation_matrix = np.eye(3, dtype=np.float64)

        # Start processing thread (Removed)

    # --- New method to fetch data from UDP socket ---
    def _fetch_latest_data(self):
        """
        Instantly drain the UDP socket buffer, keeping only the *newest*
        packet of each sensor type. This is non-blocking and
        handles out-of-order packets.
        """
        latest_accel_ts = -1
        latest_rotation_ts = -1
        latest_accel_msg = None
        latest_rotation_msg = None
        current_time_stamp = time.time_ns()
        if self.closed:
            return

        try:
            while True:
                data, addr = self.sock.recvfrom(4096)
                message = orjson.loads(data)
                # --- THIS IS THE FIX ---
                # We use an *exact match* to avoid mixing sensor types
                sensor_type = message.get('type', '')
                new_ts = message.get('timestamp', 0)
                if sensor_type == 'android.sensor.accelerometer':
                    if new_ts > latest_accel_ts:
                        latest_accel_ts = new_ts
                        latest_accel_msg = message

                # We ONLY listen for 'rotation_vector', just like your original code
                elif sensor_type == 'android.sensor.rotation_vector':
                    if new_ts > latest_rotation_ts:
                        latest_rotation_ts = new_ts
                        latest_rotation_msg = message

        except BlockingIOError:
            pass  # Socket buffer is empty. This is normal.
        # except Exception as e:
        #     print(f"\rError fetching sensor data: {e}", end="", flush=True)

        # Only update our internal data if the packet we just found
        # is *actually newer* than the last one we fully processed.
        if latest_accel_msg and latest_accel_ts > self.last_processed_accel_timestamp_ns:
            latest_accel_msg['pc_timestamp'] = current_time_stamp
            self._raw_accel_data = latest_accel_msg

        if latest_rotation_msg and latest_rotation_ts > self.last_processed_rotation_timestamp_ns:
            latest_rotation_msg['pc_timestamp'] = current_time_stamp
            # print(latest_rotation_msg, "latest_rotation_msg")
            self._raw_rotation_data = latest_rotation_msg

    def precise_time_ns(self):
        if self._raw_accel_data is not None:
            return self._raw_accel_data['timestamp']
        return None

    def rotation_matrix_when(self, t_seconds, device=None):
        """Get rotation matrix at any given time with interpolation"""
        if t_seconds == "now":
            return self.rotation_matrix_now(device=device)

        if device is None:
            device = 'cpu'

        numpy_array = self.rotation_buffer.at(t_seconds)
        # We return CPU tensor (0-Copy, 0-latency)
        if device == 'cpu':
            return torch.from_numpy(numpy_array)
        self.pinned_rotation.copy_(torch.from_numpy(numpy_array))
        self.on_device_rotation_buffer.copy_(self.pinned_rotation, non_blocking=True)
        return self.gpu_buffers[self.idx]

    def rotation_matrix_now(self, device=None):
        # This returns the fastest | 0 Buffering blazing fast DSP copy implementation available
        if device == 'cpu':
            return torch.from_numpy(self.rotation_matrix).to(dtype=torch.float32)
        return self.gpu_buffers[self.idx]

    def update_async(self):
        """
        Call this every frame on the CPU.
        It triggers the upload for the *NEXT* frame, but returns the *CURRENT* safe frame.
        """
        t_seconds = time.time_ns() * 1e-9
        # The buffer index we are about to write to (The Back Buffer)
        write_idx = (self.idx + 1) % 2
        # The buffer index the GPU should use right now (The Front Buffer)
        read_idx = self.idx
        # 1. Get data from your source (Interpolation happen here)
        # Direct copy into the Pinned CPU buffer
        # Note: We assume rotation_source.at(t) returns a numpy array or tensor
        data = self.rotation_buffer.at(t_seconds)
        # Fast CPU-to-CPU copy
        # We construct a view to avoid allocation, then copy into pinned memory
        self.cpu_pinned[write_idx].copy_(torch.from_numpy(data))
        self.gpu_buffers[write_idx].copy_(self.cpu_pinned[write_idx], non_blocking=True)
        self.idx = write_idx
        return self.gpu_buffers[read_idx]

    def get_latest_sensor_time_ns(self):
        """Thread-safe way to get the latest sensor time"""
        return self.latest_sensor_timestamp_ns

    def _remap_rotation_matrix(self, R_enu):
        """
        Apply your empirical coordinate remapping to rotation matrix.
        Input: R in Android ENU frame
        Output: R in your coordinate system where Y is always "device up"
        """
        R = R_enu.T
        R_remapped = np.array([
            [R[0, 0], R[0, 2], -R[0, 1]],
            [R[1, 0], R[1, 2], -R[1, 1]],
            [R[2, 0], R[2, 2], -R[2, 1]]
        ], dtype=np.float64)
        R_remapped[:, 0] = -R_remapped[:, 0]

        # Fix: Phone appears upside down - rotate 180° around Z axis (viewing direction)
        flip_180 = np.array([
            [-1, 0, 0],  # Flip X
            [0, -1, 0],  # Flip Y
            [0, 0, 1]  # Keep Z
        ], dtype=np.float64)
        R_remapped = flip_180 @ R_remapped

        if self.mode == 'landscape':
            # Landscape mode: right side becomes up
            landscape_rotation = np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            R_remapped = landscape_rotation @ R_remapped

        elif self.mode == 'flat':
            # Flat mode: screen facing up (90° rotation around X axis)
            # Portrait Y (top of phone) → stays horizontal
            # Portrait Z (out of screen) → becomes up (+Y)
            flat_rotation = np.array([
                [1, 0, 0],  # X stays X
                [0, 0, -1],  # Y = -Z (screen out becomes up)
                [0, 1, 0]  # Z = Y (top of phone becomes forward)
            ], dtype=np.float64)
            R_remapped = flat_rotation @ R_remapped

        return (-R_remapped).T

    @staticmethod
    def remap_position_to_coordinate_system(position_enu):
        position_remapped = np.array([
            -position_enu[0],
            position_enu[2],
            -position_enu[1]
        ], dtype=np.float64)
        return position_remapped

    def update(self):
        # --- 0. Fetch latest data from socket ---
        if self.closed or DEBUG_IGNORE_SENSOR:
            return

        self._fetch_latest_data()
        current_rotation_data = self._raw_rotation_data
        current_accel_data = self._raw_accel_data
        if current_rotation_data is None or current_accel_data is None:
            if self.failed_count % 100 == 0:
                print("Failed to Retrieve Sensor Data")
            self.failed_count += 1
            return
        self.failed_count = 0
        rotation_timestamp_ns = current_rotation_data['timestamp']
        rotation_pc_timestamp_ns = current_rotation_data['pc_timestamp']
        accel_timestamp_ns = current_accel_data['timestamp']
        # --- 1. Process Rotation (Always first!) ---
        if self.last_processed_rotation_timestamp_ns != rotation_timestamp_ns:
            q = current_rotation_data['values'][:4]  # [x, y, z, w]
            self.current_rotation_matrix = IMUSensing.quaternion_to_rotation_matrix(q)
            remapped = self._remap_rotation_matrix(self.current_rotation_matrix)
            self.rotation_buffer.add(remapped, rotation_pc_timestamp_ns / self.NS_TO_S)
            self.rotation_matrix = remapped
            self.last_processed_rotation_timestamp_ns = rotation_timestamp_ns

        # --- 2. Process Acceleration ---
        if self.last_processed_accel_timestamp_ns != accel_timestamp_ns:
            current_time_sec = accel_timestamp_ns / self.NS_TO_S
            last_time_sec = self.last_processed_accel_timestamp_ns / self.NS_TO_S
            delta_t = current_time_sec - last_time_sec
            raw_accel_device = np.array(
                current_accel_data['values'][:3],
                dtype=np.float64
            )
            # --- MANUAL GRAVITY REMOVAL ---
            gravity_in_device_frame = self.current_rotation_matrix.T @ self.GRAVITY_WORLD
            linear_accel_with_bias = raw_accel_device - gravity_in_device_frame

            # --- ZUPT & Bias Calibration Logic ---
            accel_magnitude = np.linalg.norm(linear_accel_with_bias)
            if accel_magnitude > self.STATIONARY_THRESHOLD_MS2:
                # ----- MOVING -----
                self.is_stationary = False
                self.stationary_frames = 0
                self.bias_calibration_buffer = []

                corrected_accel_device = linear_accel_with_bias - self.accel_bias
                linear_accel_world = self.current_rotation_matrix @ corrected_accel_device

            else:
                # ----- STATIONARY -----
                self.stationary_frames += 1
                linear_accel_world = np.zeros(3, dtype=np.float64)

                # with self.readings_lock: # Removed
                for reading_data in self.readings.values():
                    reading_data['velocity'] = np.zeros(3, dtype=np.float64)
                    reading_data['rotation_matrix_'] = self.rotation_matrix

                if self.stationary_frames >= self.STATIONARY_FRAMES_MIN:
                    self.is_stationary = True
                    self.bias_calibration_buffer.append(linear_accel_with_bias)
                    if len(self.bias_calibration_buffer) >= self.BIAS_CALIBRATION_FRAMES:
                        new_bias = np.mean(self.bias_calibration_buffer, axis=0)
                        self.accel_bias = self.accel_bias * 0.8 + new_bias * 0.2
                        self.bias_calibration_buffer = []

            # --- 3. Integrate for all readings ---
            if self.last_processed_accel_timestamp_ns > 0 and 0 < delta_t < 1.0:
                # with self.readings_lock: # Removed
                for reading_id, reading_data in self.readings.items():
                    v_old = reading_data['velocity']
                    v_new = v_old + linear_accel_world * delta_t
                    reading_data['position_'] += (v_old + v_new) * 0.5 * delta_t
                    reading_data['position'] = IMUSensing.remap_position_to_coordinate_system(reading_data['position_'])
                    reading_data['position_at'].add(
                        reading_data['position'],
                        current_time_sec
                    )
                    reading_data['position_highpass'].add_node(
                        reading_data['position'],
                        current_time_sec
                    )
                    reading_data['velocity'] = v_new * 0.999
                    reading_data['velocity'][reading_data['velocity'] > 0] -= 0.01
                    reading_data['velocity'][reading_data['velocity'] < 0] += 0.01
                    reading_data['rotation_matrix_'] = self.rotation_matrix

                    if self.verbose:
                        vel_mag = np.linalg.norm(reading_data['velocity'])
                        status = "STATIONARY" if self.is_stationary else "MOVING    "
                        cal_status = f"CALIBRATING ({len(self.bias_calibration_buffer)}/{self.BIAS_CALIBRATION_FRAMES})" if 0 < len(
                            self.bias_calibration_buffer
                        ) else ""
                        print(
                            f"\r{status} | "
                            f"V: {vel_mag:6.3f} m/s | "
                            f"P: [{reading_data['position'][0]:7.3f}, "
                            f"{reading_data['position'][1]:7.3f}, "
                            f"{reading_data['position'][2]:7.3f}] m | "
                            f"dt: {delta_t * 1000:5.1f} ms | "
                            f"{cal_status}",
                            end="", flush=True
                        )

            self.last_processed_accel_timestamp_ns = accel_timestamp_ns
            self.latest_sensor_timestamp_ns = max(rotation_timestamp_ns, accel_timestamp_ns)
            self.update_async()

    def make_reading(self, reading_id, cutoff_freq=0.06):
        """Create a new reading tracker"""
        # with self.readings_lock: # Removed
        position_buffer = CircularPositionBuffer(self.accel_time_buffer)
        highpass_filter = HighPassFilter(cutoff_freq, dtype=self.dtype, device=self.device)
        self.readings[reading_id] = {
            'position_': np.zeros(3, dtype=np.float64),
            'position': np.zeros(3, dtype=np.float64),
            'velocity': np.zeros(3, dtype=np.float64),
            'position_at': position_buffer,
            'position_highpass': highpass_filter
        }

    def reset_bias_calibration(self):
        """Manually reset the calculated bias"""
        print("\n*** BIAS RESET ***\n")
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.bias_calibration_buffer = []
        self.stationary_frames = 0
        self.is_stationary = False

    def capture(self):
        """Capture current state (thread-safe)"""
        R = self.rotation_matrix
        return {'rotation_matrix': R}

    def close(self):
        self.running = False
        self.closed = True
        self.sock.close()
        print("\nIMU: Socket closed.")

    def get_highpass_position(self, reading_id, t):
        """Get highpass filtered position at time t"""
        if reading_id not in self.readings:
            raise AttributeError(f"reading_id: {reading_id} not in readings")
        return self.readings[reading_id]['position_highpass'].get_value(t)

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        Convert quaternion to rotation matrix (device-to-world).
        Assumes q = [x, y, z, w] from Android's ROTATION_VECTOR
        """
        x, y, z, w = q
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        xx, yy, zz = x * x, y * y, z * z

        # **FIX for AttributeError**:
        # This is the correct calculation from your original code.
        xy, xz, xw = x * y, x * z, x * w
        yz, yw, zw = y * z, y * w, z * w

        # Standard ENU (East-North-Up) convention
        R = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
        ], dtype=np.float64)
        return R

# def create_pose_matrix(position, rotation_matrix=None):
#     if rotation_matrix is None:
#         rotation_matrix = np.eye(3)
#     # Create 4x4 pose matrix
#     pose = np.eye(4)
#     pose[:3, :3] = rotation_matrix
#     pose[:3, 3] = rotation_matrix @ position
#     return pose


def main():
    # Initialize IMU sensor
    mode = 'portrait'
    sensor = IMUSensing(mode=mode)

    # Connect to sensor (update IP address as needed)
    print(f"Connecting to sensor at {sensor.url_info}...")

    # Create reading with ID 0
    sensor.make_reading(0)

    # Initialize visualizer
    print("Initializing visualizer...")
    visualizer = VSAVisualizer(width=1200, height=800, mode=mode, num_landmarks=1, intrinsics=np.array([
        [1346.2626115837832, 0.0, 971.1651220753282],
        [0.0, 1345.375887563327, 531.1205273630911],
        [0.0, 0.0, 1.0]
    ]))

    print("Starting visualization loop...")
    print("Controls:")
    print("  Arrow Keys - Pan view")
    print("  +/- or Mouse Wheel - Zoom in/out")
    print("  Q - Quit")

    # Main visualization loop
    frame_count = 0
    last_frame_time = time.time()

    try:
        while True:
            sensor.update()
            # Get position from reading ID 0
            # if 0 in sensor.readings:
            position =  sensor.get_highpass_position(0, sensor.latest_sensor_timestamp_ns*1e-9)
            # print(position,"position")
            if position is None:
                position = [0.0, 0.0, 0.0]
            else:
                position = position.cpu().numpy()
            # position = sensor.readings[0]['position']
            # pose_matrix = create_pose_matrix(position, rotation_matrix=sensor.rotation_matrix_when(time.time_ns() * 1e-9))
            # pose_matrix = create_pose_matrix(position, rotation_matrix=sensor.rotation_matrix_when(sensor.latest_sensor_timestamp_ns * 1e-9))

            # Render visualization
            if not visualizer.render(position, sensor.rotation_matrix_when(time.time_ns() * 1e-9)):
                break  # User closed window or pressed Q

            frame_count += 1

            # measure time since last frame
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            # prevent divide-by-zero on very fast loops
            if dt > 0:
                fps = 1.0 / dt
            else:
                fps = float('inf')

            if frame_count % 5 == 0:
                print(f"\rFPS: {fps:.1f} | thread_count: {threading.active_count()}",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        visualizer.close()
        print("Visualizer closed.")


if __name__ == '__main__':
    main()

# Is there anyway to keep "velocity" rigid in world space?
# The way it works is we have a position where points must += in world space
# That position must remain consistent, yes may drift but is indeed consistent.
# if X is towards a building. no matter how I rotate the phone scale
# Is depending on accelerometer useful?
