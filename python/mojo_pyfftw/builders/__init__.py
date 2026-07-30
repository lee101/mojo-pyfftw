from __future__ import annotations

import numpy as np

from .. import config
from ..core import FFTW, FFTW_BACKWARD, FFTW_FORWARD, empty_aligned


def _dtype_for_complex(array):
    dtype = np.asarray(array).dtype
    if (
        np.issubdtype(dtype, np.floating) and dtype.itemsize > 8
        or np.issubdtype(dtype, np.complexfloating) and dtype.itemsize > 16
    ):
        raise TypeError("long-double transforms are not supported")
    return np.complex64 if dtype in (np.float32, np.complex64) else np.complex128


def _dtype_for_real(array):
    dtype = np.asarray(array).dtype
    if np.issubdtype(dtype, np.floating) and dtype.itemsize > 8:
        raise TypeError("long-double transforms are not supported")
    return np.float32 if dtype == np.float32 else np.float64


def _normalization(plan: FFTW, norm):
    if norm in (None, "backward"):
        plan.normalise_idft, plan.ortho = True, False
    elif norm == "forward":
        plan.normalise_idft, plan.ortho = False, False
    elif norm == "ortho":
        plan.normalise_idft, plan.ortho = False, True
    else:
        raise ValueError("Invalid norm value")
    return plan


def _axes_and_shape(array, s, axes, default_axes=None):
    ndim = array.ndim
    if axes is None:
        axes = tuple(range(ndim)) if s is None else tuple(range(ndim - len(s), ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    axes = tuple(np.lib.array_utils.normalize_axis_index(axis, ndim) for axis in axes)
    if s is None:
        shape = tuple(array.shape[axis] for axis in axes)
    else:
        shape = tuple(int(n) for n in s)
        if len(shape) != len(axes):
            raise ValueError("Shape and axes have different lengths")
    return axes, shape


def _resize(array, axes, shape):
    target_shape = list(array.shape)
    for axis, n in zip(axes, shape):
        target_shape[axis] = n
    result = np.zeros(target_shape, dtype=array.dtype)
    source_slices = [slice(None)] * array.ndim
    target_slices = [slice(None)] * array.ndim
    for axis, n in zip(axes, shape):
        stop = min(array.shape[axis], n)
        source_slices[axis] = slice(0, stop)
        target_slices[axis] = slice(0, stop)
    result[tuple(target_slices)] = array[tuple(source_slices)]
    return result


def _plan(
    source,
    axes,
    shape,
    kind,
    norm,
    planner_effort,
    threads,
):
    source = np.asarray(source)
    if any(n <= 0 for n in shape):
        raise ValueError("transform lengths must be positive")
    if kind in ("fft", "ifft"):
        input_dtype = _dtype_for_complex(source)
        planned_input = _resize(source.astype(input_dtype), axes, shape)
        output_shape = planned_input.shape
        output_dtype = input_dtype
        direction = FFTW_FORWARD if kind == "fft" else FFTW_BACKWARD
    elif kind == "rfft":
        if np.iscomplexobj(source):
            raise TypeError("rfft input must be real")
        input_dtype = _dtype_for_real(source)
        planned_input = _resize(source.astype(input_dtype), axes, shape)
        output_shape = list(planned_input.shape)
        output_shape[axes[-1]] = shape[-1] // 2 + 1
        output_shape = tuple(output_shape)
        output_dtype = np.complex64 if input_dtype == np.float32 else np.complex128
        direction = FFTW_FORWARD
    else:
        if not np.iscomplexobj(source):
            source = source.astype(_dtype_for_complex(source))
        input_dtype = _dtype_for_complex(source)
        output_shape = list(source.shape)
        for axis, n in zip(axes, shape):
            output_shape[axis] = n
        output_shape = tuple(output_shape)
        input_shape = list(output_shape)
        input_shape[axes[-1]] = shape[-1] // 2 + 1
        planned_input = _resize(source.astype(input_dtype), axes, tuple(input_shape[a] for a in axes))
        output_dtype = np.float32 if input_dtype == np.complex64 else np.float64
        direction = FFTW_BACKWARD
    input_buffer = empty_aligned(planned_input.shape, input_dtype)
    input_buffer[...] = planned_input
    output_buffer = empty_aligned(output_shape, output_dtype)
    flags = (planner_effort or config.PLANNER_EFFORT,)
    plan = FFTW(
        input_buffer,
        output_buffer,
        axes=axes,
        direction=direction,
        flags=flags,
        threads=threads or config.NUM_THREADS,
    )
    return _normalization(plan, norm)


def fft(
    a, n=None, axis=-1, overwrite_input=False, planner_effort=None, threads=None,
    auto_align_input=True, auto_contiguous=True, avoid_copy=False, norm=None,
):
    source = np.asarray(a)
    axis = np.lib.array_utils.normalize_axis_index(axis, source.ndim)
    return _plan(source, (axis,), (source.shape[axis] if n is None else int(n),), "fft", norm, planner_effort, threads)


def ifft(
    a, n=None, axis=-1, overwrite_input=False, planner_effort=None, threads=None,
    auto_align_input=True, auto_contiguous=True, avoid_copy=False, norm=None,
):
    source = np.asarray(a)
    axis = np.lib.array_utils.normalize_axis_index(axis, source.ndim)
    return _plan(source, (axis,), (source.shape[axis] if n is None else int(n),), "ifft", norm, planner_effort, threads)


def rfft(
    a, n=None, axis=-1, overwrite_input=False, planner_effort=None, threads=None,
    auto_align_input=True, auto_contiguous=True, avoid_copy=False, norm=None,
):
    source = np.asarray(a)
    axis = np.lib.array_utils.normalize_axis_index(axis, source.ndim)
    return _plan(source, (axis,), (source.shape[axis] if n is None else int(n),), "rfft", norm, planner_effort, threads)


def irfft(
    a, n=None, axis=-1, overwrite_input=False, planner_effort=None, threads=None,
    auto_align_input=True, auto_contiguous=True, avoid_copy=False, norm=None,
):
    source = np.asarray(a)
    axis = np.lib.array_utils.normalize_axis_index(axis, source.ndim)
    length = 2 * (source.shape[axis] - 1) if n is None else int(n)
    return _plan(source, (axis,), (length,), "irfft", norm, planner_effort, threads)


def fftn(
    a, s=None, axes=None, overwrite_input=False, planner_effort=None, threads=None,
    auto_align_input=True, auto_contiguous=True, avoid_copy=False, norm=None,
):
    source = np.asarray(a)
    axes, shape = _axes_and_shape(source, s, axes)
    return _plan(source, axes, shape, "fft", norm, planner_effort, threads)


def ifftn(
    a, s=None, axes=None, overwrite_input=False, planner_effort=None, threads=None,
    auto_align_input=True, auto_contiguous=True, avoid_copy=False, norm=None,
):
    source = np.asarray(a)
    axes, shape = _axes_and_shape(source, s, axes)
    return _plan(source, axes, shape, "ifft", norm, planner_effort, threads)


def rfftn(
    a, s=None, axes=None, overwrite_input=False, planner_effort=None, threads=None,
    auto_align_input=True, auto_contiguous=True, avoid_copy=False, norm=None,
):
    source = np.asarray(a)
    axes, shape = _axes_and_shape(source, s, axes)
    return _plan(source, axes, shape, "rfft", norm, planner_effort, threads)


def irfftn(
    a, s=None, axes=None, overwrite_input=False, planner_effort=None, threads=None,
    auto_align_input=True, auto_contiguous=True, avoid_copy=False, norm=None,
):
    source = np.asarray(a)
    if axes is None:
        axes = tuple(range(source.ndim)) if s is None else tuple(range(source.ndim - len(s), source.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    axes = tuple(np.lib.array_utils.normalize_axis_index(axis, source.ndim) for axis in axes)
    if s is None:
        shape = tuple(source.shape[axis] for axis in axes[:-1]) + (2 * (source.shape[axes[-1]] - 1),)
    else:
        shape = tuple(int(n) for n in s)
    return _plan(source, axes, shape, "irfft", norm, planner_effort, threads)


def fft2(a, s=None, axes=(-2, -1), **kwargs):
    return fftn(a, s=s, axes=axes, **kwargs)


def ifft2(a, s=None, axes=(-2, -1), **kwargs):
    return ifftn(a, s=s, axes=axes, **kwargs)


def rfft2(a, s=None, axes=(-2, -1), **kwargs):
    return rfftn(a, s=s, axes=axes, **kwargs)


def irfft2(a, s=None, axes=(-2, -1), **kwargs):
    return irfftn(a, s=s, axes=axes, **kwargs)


__all__ = [
    "fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "rfft2", "irfft2",
    "fftn", "ifftn", "rfftn", "irfftn",
]
