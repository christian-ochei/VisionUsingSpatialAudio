import torch
import functools
import traceback
import inspect
from typing import Any, Callable, Set


class NaNDetector:
    """Context manager that intercepts ALL PyTorch operations to detect NaN/Inf values.

    Automatically discovers and wraps all tensor methods and torch functions.
    Works with in-place operations and CUDA tensors.

    Usage:
        with NaNDetector():
            x = torch.randn(100, device='cuda')
            x.mul_(2.0)  # All operations monitored
            y = torch.fft.fft(x)
    """

    def __init__(self, check_inf=True, print_values=False, exclude_ops=None):
        self.check_inf = check_inf
        self.print_values = print_values
        self.exclude_ops = exclude_ops or set()
        self.original_methods = {}
        self.operation_count = 0

        # Store original methods we'll need for checking
        self._original_numel = torch.Tensor.numel
        self._original_isnan = torch.isnan
        self._original_isinf = torch.isinf
        self._original_any = torch.Tensor.any
        self._original_item = torch.Tensor.item
        self._original_nonzero = torch.nonzero

    def _check_tensor(self, tensor, op_name, is_output=True):
        """Check a single tensor for NaN/Inf using original unwrapped methods."""
        if not isinstance(tensor, torch.Tensor):
            return

        # Use original methods to avoid recursion
        if self._original_numel(tensor) == 0:  # Empty tensor
            return

        has_nan = self._original_item(self._original_any(self._original_isnan(tensor)))
        has_inf = self.check_inf and self._original_item(self._original_any(self._original_isinf(tensor)))

        if has_nan or has_inf:
            issue = "NaN" if has_nan else "Inf"
            location = "output" if is_output else "input"

            error_msg = [
                f"\n{'=' * 70}",
                f"{issue} DETECTED in {location} of operation: {op_name}",
                f"Operation count: {self.operation_count}",
                f"Tensor shape: {tensor.shape}",
                f"Tensor device: {tensor.device}",
                f"Tensor dtype: {tensor.dtype}",
            ]

            if self.print_values:
                error_msg.append(f"Tensor values:\n{tensor}")

            # Get indices where NaN/Inf occurs
            if has_nan:
                nan_indices = self._original_nonzero(self._original_isnan(tensor))
                error_msg.append(f"NaN count: {nan_indices.shape[0]}")
                error_msg.append(f"First NaN indices: {nan_indices[:5].tolist()}")

            if has_inf:
                inf_indices = self._original_nonzero(self._original_isinf(tensor))
                error_msg.append(f"Inf count: {inf_indices.shape[0]}")
                error_msg.append(f"First Inf indices: {inf_indices[:5].tolist()}")

            error_msg.append(f"{'=' * 70}")
            error_msg.append("\nStack trace:")

            print("\n".join(error_msg))
            traceback.print_stack()

            raise RuntimeError(f"{issue} detected in {op_name}")

    def _wrap_method(self, method_name: str, original_method: Callable) -> Callable:
        """Wrap a method to add NaN checking."""

        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            self.operation_count += 1

            # Check inputs
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    self._check_tensor(arg, f"{method_name} (input {i})", is_output=False)

            # Execute original operation
            result = original_method(*args, **kwargs)

            # Check outputs
            if isinstance(result, torch.Tensor):
                self._check_tensor(result, method_name, is_output=True)
            elif isinstance(result, (tuple, list)):
                for i, r in enumerate(result):
                    if isinstance(r, torch.Tensor):
                        self._check_tensor(r, f"{method_name} (output {i})", is_output=True)

            return result

        return wrapper

    def _get_all_tensor_methods(self) -> Set[str]:
        """Discover all tensor methods dynamically."""
        methods = set()

        # Core methods we must exclude to avoid recursion
        critical_exclude = {
            'numel', 'any', 'item', '__str__', '__repr__',
            '__format__', '__sizeof__', '__hash__'
        }

        for name in dir(torch.Tensor):
            # Skip private methods and properties
            if name.startswith('_'):
                continue

            # Skip excluded operations
            if name in self.exclude_ops or name in critical_exclude:
                continue

            attr = getattr(torch.Tensor, name)

            # Only wrap callable methods
            if callable(attr):
                methods.add(name)

        return methods

    def _get_all_torch_functions(self) -> Set[str]:
        """Discover all torch module functions dynamically."""
        functions = set()

        # Functions to exclude to avoid recursion
        critical_exclude = {'isnan', 'isinf', 'nonzero'}

        for name in dir(torch):
            # Skip private and special methods
            if name.startswith('_'):
                continue

            # Skip excluded operations
            if name in self.exclude_ops or name in critical_exclude:
                continue

            attr = getattr(torch, name)

            # Only wrap functions (not classes, modules, etc.)
            if callable(attr) and not inspect.isclass(attr):
                functions.add(name)

        return functions

    def _get_torch_submodule_functions(self) -> dict:
        """Get functions from torch submodules like torch.fft, torch.linalg, etc."""
        submodules = {}

        # Important torch submodules for DSP
        module_names = ['fft', 'linalg', 'special', 'nn.functional']

        for mod_name in module_names:
            try:
                # Handle nested modules like nn.functional
                parts = mod_name.split('.')
                mod = torch
                for part in parts:
                    mod = getattr(mod, part)

                funcs = set()
                for name in dir(mod):
                    if name.startswith('_'):
                        continue
                    if name in self.exclude_ops:
                        continue

                    attr = getattr(mod, name)
                    if callable(attr) and not inspect.isclass(attr):
                        funcs.add(name)

                if funcs:
                    submodules[mod_name] = (mod, funcs)
            except AttributeError:
                pass

        return submodules

    def __enter__(self):
        """Monkey-patch all PyTorch operations."""

        # Patch all tensor methods
        # print("Discovering tensor methods...")
        tensor_methods = self._get_all_tensor_methods()
        # print(f"Found {len(tensor_methods)} tensor methods")

        for method_name in tensor_methods:
            try:
                original = getattr(torch.Tensor, method_name)
                self.original_methods[f'Tensor.{method_name}'] = original
                setattr(torch.Tensor, method_name, self._wrap_method(f'Tensor.{method_name}', original))
            except Exception:
                pass

        # Patch all torch functions
        # print("Discovering torch functions...")
        torch_functions = self._get_all_torch_functions()
        # print(f"Found {len(torch_functions)} torch functions")

        for func_name in torch_functions:
            try:
                original = getattr(torch, func_name)
                self.original_methods[f'torch.{func_name}'] = original
                setattr(torch, func_name, self._wrap_method(f'torch.{func_name}', original))
            except Exception:
                pass

        # Patch submodule functions (torch.fft, torch.linalg, etc.)
        # print("Discovering submodule functions...")
        submodules = self._get_torch_submodule_functions()
        for mod_name, (mod, funcs) in submodules.items():
            # print(f"Found {len(funcs)} functions in torch.{mod_name}")
            for func_name in funcs:
                try:
                    original = getattr(mod, func_name)
                    key = f'torch.{mod_name}.{func_name}'
                    self.original_methods[key] = original
                    setattr(mod, func_name, self._wrap_method(key, original))
                except Exception:
                    pass

        # print(f"Monitoring {len(self.original_methods)} operations total\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original methods."""
        for full_name, original_method in self.original_methods.items():
            try:
                if full_name.startswith('Tensor.'):
                    method_name = full_name.replace('Tensor.', '')
                    setattr(torch.Tensor, method_name, original_method)
                elif full_name.startswith('torch.'):
                    parts = full_name.replace('torch.', '').split('.')
                    if len(parts) == 1:
                        # torch.function
                        setattr(torch, parts[0], original_method)
                    else:
                        # torch.submodule.function
                        mod = torch
                        for part in parts[:-1]:
                            mod = getattr(mod, part)
                        setattr(mod, parts[-1], original_method)
            except Exception:
                pass

        self.original_methods.clear()
        # print(f"\nNaNDetector: Monitored {self.operation_count} operations")


# Decorator version for functions
def check_nans(check_inf=True, print_values=False, exclude_ops=None):
    """Decorator that wraps a function with NaN detection.

    Usage:
        @check_nans()
        def my_dsp_function(signal):
            signal = signal.mul_(2.0)
            return torch.fft.fft(signal)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with NaNDetector(check_inf=check_inf, print_values=print_values, exclude_ops=exclude_ops):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    print("Example 1: Comprehensive operation monitoring")
    try:
        with NaNDetector():
            # Various operations
            x = torch.randn(100, device='cuda')
            x.mul_(2.0)
            x = torch.fft.fft(x)
            x = torch.abs(x)
            x.add_(torch.tensor(float('nan'), device='cuda'))  # This will be caught
    except RuntimeError as e:
        print(f"Caught error: {e}\n")

    print("\nExample 2: DSP pipeline")


    @check_nans(print_values=False)
    def dsp_pipeline(signal):
        # Normalize
        signal = signal.clone()
        signal.div_(signal.abs().max())

        # FFT
        spectrum = torch.fft.rfft(signal)

        # Filter (apply window)
        window = torch.hann_window(spectrum.shape[-1], device=signal.device)
        spectrum.mul_(window)

        # IFFT
        output = torch.fft.irfft(spectrum, n=signal.shape[-1])

        return output


    signal = torch.randn(1024, device='cuda')
    result = dsp_pipeline(signal)
    print(f"Processed signal shape: {result.shape}")