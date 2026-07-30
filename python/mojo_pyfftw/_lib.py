from __future__ import annotations

import ctypes
import math
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYFFTW_LIB", os.path.join(ROOT, "dist", "libmojo-pyfftw.so"))
I = ctypes.c_int64
F = ctypes.c_double

_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    sources = [
        os.path.join(ROOT, "src", name)
        for name in os.listdir(os.path.join(ROOT, "src"))
        if name.endswith(".mojo")
    ]
    stale = not os.path.exists(LIB) or any(
        os.path.getmtime(path) > os.path.getmtime(LIB) for path in sources
    )
    if force or stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        prepare = _library.mpf_prepare_workspace_f64
        prepare.argtypes = [I, I, I]
        prepare.restype = I
        rfft = _library.mpf_rfft_power2_f64
        rfft.argtypes = [I, I, I, I, I, F, I]
        rfft.restype = I
        fn = _library.mpf_transform_axis_f64
        fn.argtypes = [I, I, I, I, I, I, I, F, I]
        fn.restype = I
    return _library


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _scratch_size(n: int) -> int:
    n = int(n)
    if n <= 0:
        raise ValueError("transform length must be positive")
    if not (n & (n - 1)):
        return 2 * n
    return 3 * _next_power_of_two(2 * n - 1)


def _require_buffer(
    array: np.ndarray,
    *,
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...] | None = None,
    minimum_size: int | None = None,
    writable: bool = False,
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if minimum_size is not None and array.size < minimum_size:
        raise ValueError(f"{name} is too small")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _require_no_overlap(
    first: np.ndarray,
    second: np.ndarray,
    names: str,
    *,
    allow_exact: bool = False,
) -> None:
    exact = (
        first.ctypes.data == second.ctypes.data
        and first.shape == second.shape
        and first.strides == second.strides
        and first.dtype == second.dtype
    )
    if np.shares_memory(first, second) and not (allow_exact and exact):
        raise ValueError(f"{names} must not overlap")


def prepare_workspace(scratch: np.ndarray, n: int, direction: int) -> None:
    n = int(n)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    _require_buffer(
        scratch,
        name="scratch",
        dtype=np.dtype(np.complex128),
        minimum_size=_scratch_size(n),
        writable=True,
    )
    ok = lib().mpf_prepare_workspace_f64(scratch.ctypes.data, n, direction)
    if not ok:
        raise ValueError("invalid transform length")


def transform_axis(
    source: np.ndarray,
    axis: int,
    direction: int,
    scale: float = 1.0,
    result: np.ndarray | None = None,
    scratch: np.ndarray | None = None,
    prepared: bool = False,
    threads: int = 1,
) -> np.ndarray:
    if not isinstance(source, np.ndarray):
        source = np.asarray(source)
    axis = np.lib.array_utils.normalize_axis_index(axis, source.ndim)
    source = np.ascontiguousarray(source, dtype=np.complex128)
    if source.size == 0:
        raise ValueError("source must be non-empty")
    if result is None:
        result = np.empty_like(source)
    n = source.shape[axis]
    inner = math.prod(source.shape[axis + 1 :])
    _require_buffer(
        result,
        name="result",
        dtype=np.dtype(np.complex128),
        shape=source.shape,
        writable=True,
    )
    if scratch is None:
        scratch = np.empty(_scratch_size(n), dtype=np.complex128)
    _require_buffer(
        scratch,
        name="scratch",
        dtype=np.dtype(np.complex128),
        minimum_size=_scratch_size(n),
        writable=True,
    )
    _require_no_overlap(source, result, "source and result", allow_exact=True)
    _require_no_overlap(source, scratch, "source and scratch")
    _require_no_overlap(result, scratch, "result and scratch")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if not prepared:
        prepare_workspace(scratch, n, direction)
    ok = lib().mpf_transform_axis_f64(
        source.ctypes.data,
        result.ctypes.data,
        scratch.ctypes.data,
        source.size,
        n,
        inner,
        direction,
        scale,
        max(1, int(threads)),
    )
    if not ok:
        raise ValueError("invalid transform dimensions")
    return result


def rfft_power2(
    source: np.ndarray,
    result: np.ndarray,
    scratch: np.ndarray,
    scale: float = 1.0,
    threads: int = 1,
) -> np.ndarray:
    _require_buffer(
        source,
        name="source",
        dtype=np.dtype(np.float64),
        writable=False,
    )
    if source.ndim == 0:
        raise ValueError("source must have at least one dimension")
    n = source.shape[-1]
    expected = source.shape[:-1] + (n // 2 + 1,)
    _require_buffer(
        result,
        name="result",
        dtype=np.dtype(np.complex128),
        shape=expected,
        writable=True,
    )
    _require_buffer(
        scratch,
        name="scratch",
        dtype=np.dtype(np.complex128),
        minimum_size=_scratch_size(n),
        writable=True,
    )
    _require_no_overlap(source, result, "source and result")
    _require_no_overlap(source, scratch, "source and scratch")
    _require_no_overlap(result, scratch, "result and scratch")
    ok = lib().mpf_rfft_power2_f64(
        source.ctypes.data,
        result.ctypes.data,
        scratch.ctypes.data,
        source.size,
        n,
        scale,
        max(1, int(threads)),
    )
    if not ok:
        raise ValueError("invalid real transform dimensions")
    return result


def transform(
    source: np.ndarray,
    axes: tuple[int, ...],
    direction: int,
    scale: float = 1.0,
) -> np.ndarray:
    result = np.ascontiguousarray(source, dtype=np.complex128)
    for index, axis in enumerate(axes):
        result = transform_axis(
            result,
            axis,
            direction,
            scale if index == len(axes) - 1 else 1.0,
        )
    if not axes and scale != 1.0:
        result *= scale
    return result
