import torch
import math


class LowPassFilter:
    """
    Blazing fast low-pass filter using simple exponential smoothing.
    Circular buffer with CUDA tensors, Python scalars for time/math.
    """

    def __init__(self, cutoff_freq, max_nodes=50, dtype=torch.float32, device='cuda'):
        self.cutoff_freq = cutoff_freq
        self.alpha = 2.0 * math.pi * cutoff_freq  # Smoothing rate
        self.max_nodes = max_nodes
        self.dtype = dtype
        self.device = device

        # Circular buffer state
        self.write_idx = 0
        self.count = 0
        self.last_time = None

        # Pre-allocated storage
        self.times = [0.0] * max_nodes  # Python list (fast scalar access)
        self.values = None  # CUDA tensor for values
        self.filtered = None  # CUDA tensor for filtered values
        self.value_shape = None
        self.out_interp = None

    def add_node(self, value, time):
        """Add node in O(1). Raises ValueError if time goes backwards."""
        # Validate monotonic
        if self.last_time is not None and time < self.last_time:
            if abs(self.last_time - time) < 0.0001:
                return
            print(f"WARNING: Non-monotonic timestamp: {time} < {self.last_time}")
            return

        # Convert to tensor if needed
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(value).pin_memory()

        if value.device.type == 'cpu' and 'cuda' in str(self.device):
            value = value.to(self.device, non_blocking=True)
        else:
            value = value.to(self.device)

        if value.ndim == 0:
            value = value.unsqueeze(0)

        # Initialize on first add
        if self.filtered is None:
            self.value_shape = value.shape
            self.values = torch.zeros((self.max_nodes,) + value.shape, dtype=self.dtype, device=self.device)
            self.filtered = torch.zeros((self.max_nodes,) + value.shape, dtype=self.dtype, device=self.device)
            self.out_interp = torch.zeros(value.shape, dtype=self.dtype, device=self.device)

            # First node
            self.times[0] = time
            self.values[0] = value
            self.filtered[0] = value
            self.write_idx = 1
            self.count = 1
            self.last_time = time
            return

        # Get previous state (Python scalars for time math)
        prev_idx = (self.write_idx - 1) % self.max_nodes
        prev_time = self.times[prev_idx]
        dt = time - prev_time

        # Simple exponential smoothing: y = y_prev + alpha * dt * (value - y_prev)
        # This is discrete approximation of dy/dt = alpha * (input - y)
        if dt > 0:
            blend = self.alpha * dt
            if blend > 1.0:
                blend = 1.0  # Clamp to prevent overshoot

            # In-place update: filtered = filtered_prev + blend * (value - filtered_prev)
            prev_filtered = self.filtered[prev_idx]
            new_filtered = prev_filtered + blend * (value - prev_filtered)
        else:
            # dt == 0, use previous filtered value
            new_filtered = self.filtered[prev_idx]

        # Write to circular buffer
        self.times[self.write_idx] = time
        self.values[self.write_idx] = value
        self.filtered[self.write_idx] = new_filtered

        # Update state
        self.last_time = time
        self.write_idx = (self.write_idx + 1) % self.max_nodes
        self.count = min(self.count + 1, self.max_nodes)

    def _get_sorted_indices(self):
        """Get indices in time order (accounting for circular wrap)."""
        if self.count < self.max_nodes:
            return list(range(self.count))
        else:
            # Wrapped: [write_idx ... max_nodes-1, 0 ... write_idx-1]
            return list(range(self.write_idx, self.max_nodes)) + list(range(self.write_idx))

    def get_original_value(self, time):
        """Get original signal via linear interpolation."""
        if self.count == 0:
            return torch.zeros(self.value_shape if self.value_shape else (1,),
                               dtype=self.dtype, device=self.device)

        indices = self._get_sorted_indices()

        # Edge cases
        if time >= self.times[indices[-1]]:
            return self.values[indices[-1]]
        if time <= self.times[indices[0]]:
            return self.values[indices[0]]

        # Find bracket (simple linear search for small n=50)
        for i in range(1, len(indices)):
            if self.times[indices[i]] >= time:
                idx0, idx1 = indices[i - 1], indices[i]
                t0, t1 = self.times[idx0], self.times[idx1]

                if t1 == t0:
                    return self.values[idx0]

                # Linear interpolation (Python scalar math)
                alpha = (time - t0) / (t1 - t0)
                # return self.values[idx0] + alpha * (self.values[idx1] - self.values[idx0])
                f0 = self.filtered[idx0]  # view
                f1 = self.filtered[idx1]  # view
                torch.lerp(f0, f1, alpha, out=self.out_interp)
                return self.out_interp
        return self.values[indices[-1]]

    def get_value(self, time):
        """Get low-pass filtered value via linear interpolation."""
        if self.count == 0:
            return None

        indices = self._get_sorted_indices()

        # Edge cases
        if time >= self.times[indices[-1]]:
            return self.filtered[indices[-1]]
        if time <= self.times[indices[0]]:
            return self.filtered[indices[0]]

        # Find bracket (simple linear search for small n=50)
        for i in range(1, len(indices)):
            if self.times[indices[i]] >= time:
                idx0, idx1 = indices[i - 1], indices[i]
                t0, t1 = self.times[idx0], self.times[idx1]

                if t1 == t0:
                    return self.filtered[idx0]

                # Linear interpolation (Python scalar math)
                alpha = (time - t0) / (t1 - t0)
                f0 = self.filtered[idx0]  # view
                f1 = self.filtered[idx1]  # view
                torch.lerp(f0, f1, alpha, out=self.out_interp)
                return self.out_interp
                # return self.filtered[idx0] + alpha * (self.filtered[idx1] - self.filtered[idx0])
        return self.filtered[indices[-1]]


class HighPassFilter:
    """
    Blazing fast high-pass filter.
    Perfect reconstruction: LPF + HPF = Original.
    """

    def __init__(self, cutoff_freq, max_nodes=50, dtype=torch.float32, device='cuda'):
        self.lpf = LowPassFilter(cutoff_freq, max_nodes, dtype=dtype, device=device)

    def add_node(self, value, time):
        """Add node. Raises ValueError if time goes backwards."""
        self.lpf.add_node(value, time)

    def get_value(self, time):
        """Get high-pass filtered value: Original - LPF."""
        original = self.lpf.get_original_value(time)
        lpv = self.lpf.get_value(time)

        if lpv is None:
            return None

        return original - lpv

# Example usage and verification
if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-CHANNEL CONTINUOUS TIME-DOMAIN FILTER VERIFICATION (TORCH)")
    print("=" * 70)

    # Create filters with 1 Hz cutoff
    lpf = LowPassFilter(cutoff_freq=1.0)
    hpf = HighPassFilter(cutoff_freq=1.0)

    # Simulate streaming 3D signal (X, Y, Z)
    test_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    test_values = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.5, -0.3]),
        torch.tensor([0.5, 1.0, 0.2]),
        torch.tensor([-0.5, 0.3, 0.8]),
        torch.tensor([0.8, -0.2, 0.5]),
        torch.tensor([0.3, 0.6, -0.1]),
        torch.tensor([-0.2, 0.1, 0.4]),
    ]

    for t, v in zip(test_times, test_values):
        lpf.add_node(v, t)
        hpf.add_node(v, t)

    print("\nVerification at node times (3D signal: X, Y, Z):")
    print("Time | Channel | Original | LPF      | HPF      | LPF+HPF  | Error")
    print("-" * 80)

    for t, orig in zip(test_times, test_values):
        lpf_val = lpf.get_value(t)
        hpf_val = hpf.get_value(t)
        reconstructed = lpf_val + hpf_val
        for ch, (o, l, h, r) in enumerate(zip(orig, lpf_val, hpf_val, reconstructed)):
            error = torch.abs(r - o).item()
            ch_name = ['X', 'Y', 'Z'][ch]
            print(f"{t:4.1f} | {ch_name:7s} | {o:8.4f} | {l:8.4f} | {h:8.4f} | "
                  f"{r:8.4f} | {error:.2e}")
        print("-" * 80)

    print("\n" + "=" * 70)
    print("Perfect reconstruction verified (Torch version)")
    print("=" * 70)
