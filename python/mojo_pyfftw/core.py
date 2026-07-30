from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ._lib import _scratch_size, prepare_workspace, rfft_power2, transform_axis

FFTW_FORWARD = "FFTW_FORWARD"
FFTW_BACKWARD = "FFTW_BACKWARD"
_DEFAULT = object()


def _axes_tuple(axes, ndim: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, (int, np.integer)):
        axes = (int(axes),)
    result = tuple(np.lib.array_utils.normalize_axis_index(int(axis), ndim) for axis in axes)
    if len(set(result)) != len(result):
        raise ValueError("axes must be unique")
    return result


def _full_spectrum(
    half: np.ndarray,
    full_shape: tuple[int, ...],
    axes: tuple[int, ...],
    full: np.ndarray | None = None,
) -> np.ndarray:
    last = axes[-1]
    n = full_shape[last]
    m = half.shape[last]
    if full is None:
        full = np.empty(full_shape, dtype=np.complex128)
    head = [slice(None)] * half.ndim
    head[last] = slice(0, m)
    full[tuple(head)] = half
    if m < n:
        partner = half
        for axis in axes[:-1]:
            indices = (-np.arange(full_shape[axis])) % full_shape[axis]
            partner = np.take(partner, indices, axis=axis)
        tail_indices = n - np.arange(m, n)
        tail = np.conjugate(np.take(partner, tail_indices, axis=last))
        target = [slice(None)] * half.ndim
        target[last] = slice(m, n)
        full[tuple(target)] = tail
    return full


class FFTW:
    def __init__(
        self,
        input_array,
        output_array,
        axes=(-1,),
        direction=FFTW_FORWARD,
        flags=("FFTW_MEASURE",),
        threads=1,
        planning_timelimit=None,
    ):
        if not isinstance(input_array, np.ndarray) or not isinstance(output_array, np.ndarray):
            raise ValueError("input_array and output_array must be numpy arrays")
        if input_array.ndim != output_array.ndim:
            raise ValueError("input and output arrays must have equal dimensionality")
        for array in (input_array, output_array):
            if (
                np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > 8
                or np.issubdtype(array.dtype, np.complexfloating)
                and array.dtype.itemsize > 16
            ):
                raise TypeError("long-double transforms are not supported")
        self.axes = _axes_tuple(axes, input_array.ndim)
        if not self.axes:
            raise ValueError("at least one transform axis is required")
        self.flags = tuple(flags)
        self._threads = int(threads)
        self._planning_timelimit = planning_timelimit
        self.input_array = input_array
        self.output_array = output_array
        self._scheme = self._infer_scheme(direction)
        self.direction = (
            FFTW_FORWARD if self._scheme == "r2c" else
            FFTW_BACKWARD if self._scheme == "c2r" else direction
        )
        if self.direction not in (FFTW_FORWARD, FFTW_BACKWARD):
            raise ValueError("only FFTW_FORWARD and FFTW_BACKWARD are supported")
        logical_shape = self.output_array.shape if self._scheme == "c2r" else self.input_array.shape
        self.N = math.prod(logical_shape[axis] for axis in self.axes)
        self.normalise_idft = True
        self.ortho = False
        self._work = (
            np.empty(logical_shape, dtype=np.complex128),
            np.empty(logical_shape, dtype=np.complex128),
        )
        self._scratch = []
        for axis in self.axes:
            n = logical_shape[axis]
            scratch = np.empty(_scratch_size(n), dtype=np.complex128)
            prepare_workspace(
                scratch,
                n,
                -1 if self.direction == FFTW_FORWARD else 1,
            )
            self._scratch.append(scratch)

    def _infer_scheme(self, direction: str) -> str:
        input_complex = np.issubdtype(self.input_array.dtype, np.complexfloating)
        output_complex = np.issubdtype(self.output_array.dtype, np.complexfloating)
        last = self.axes[-1]
        if input_complex and output_complex:
            if self.input_array.shape != self.output_array.shape:
                raise ValueError("complex transforms require equal input and output shapes")
            return "c2c"
        if not input_complex and output_complex:
            expected = list(self.input_array.shape)
            expected[last] = expected[last] // 2 + 1
            if tuple(expected) != self.output_array.shape:
                raise ValueError("invalid output shape for a real forward transform")
            return "r2c"
        if input_complex and not output_complex:
            expected = list(self.output_array.shape)
            expected[last] = expected[last] // 2 + 1
            if tuple(expected) != self.input_array.shape:
                raise ValueError("invalid input shape for a real backward transform")
            return "c2r"
        raise ValueError("real-to-real FFTW schemes are not covered")

    @property
    def input_shape(self):
        return self.input_array.shape

    @property
    def output_shape(self):
        return self.output_array.shape

    @property
    def input_dtype(self):
        return self.input_array.dtype

    @property
    def output_dtype(self):
        return self.output_array.dtype

    @property
    def input_strides(self):
        return self.input_array.strides

    @property
    def output_strides(self):
        return self.output_array.strides

    @property
    def input_alignment(self):
        return _alignment(self.input_array)

    @property
    def output_alignment(self):
        return _alignment(self.output_array)

    @property
    def simd_aligned(self):
        return is_byte_aligned(self.input_array) and is_byte_aligned(self.output_array)

    def get_input_array(self):
        return self.input_array

    def get_output_array(self):
        return self.output_array

    def update_arrays(self, new_input_array, new_output_array):
        if (
            new_input_array.shape != self.input_shape
            or new_input_array.dtype != self.input_dtype
            or new_output_array.shape != self.output_shape
            or new_output_array.dtype != self.output_dtype
        ):
            raise ValueError("new arrays must match the planned shapes and dtypes")
        self.input_array = new_input_array
        self.output_array = new_output_array

    def _transform_full(self, source: np.ndarray, direction: int, scale: float):
        current = source
        direct_output = (
            self._scheme == "c2c"
            and self.output_array.dtype == np.complex128
            and self.output_array.flags.c_contiguous
        )
        for index, axis in enumerate(self.axes):
            if index == len(self.axes) - 1 and direct_output:
                destination = self.output_array
            else:
                destination = self._work[0] if current is not self._work[0] else self._work[1]
            transform_axis(
                current,
                axis,
                direction,
                scale if index == len(self.axes) - 1 else 1.0,
                result=destination,
                scratch=self._scratch[index],
                prepared=True,
                threads=self._threads,
            )
            current = destination
        return current

    def _execute(self, scale=1.0):
        if self._scheme == "c2c":
            if self.input_array.dtype == np.complex128 and self.input_array.flags.c_contiguous:
                source = self.input_array
            else:
                self._work[0][...] = self.input_array
                source = self._work[0]
            result = self._transform_full(
                source,
                -1 if self.direction == FFTW_FORWARD else 1,
                scale,
            )
        elif self._scheme == "r2c":
            n = self.input_shape[-1]
            if (
                self.axes == (self.input_array.ndim - 1,)
                and self.input_array.dtype == np.float64
                and self.output_array.dtype == np.complex128
                and self.input_array.flags.c_contiguous
                and self.output_array.flags.c_contiguous
                and n > 0
                and not (n & (n - 1))
            ):
                result = rfft_power2(
                    self.input_array,
                    self.output_array,
                    self._scratch[0],
                    scale,
                    self._threads,
                )
            else:
                self._work[0].real[...] = self.input_array
                self._work[0].imag.fill(0)
                result = self._transform_full(self._work[0], -1, scale)
                selection = [slice(None)] * result.ndim
                selection[self.axes[-1]] = slice(0, self.output_shape[self.axes[-1]])
                result = result[tuple(selection)]
        else:
            spectrum = _full_spectrum(
                np.asarray(self.input_array, dtype=np.complex128),
                self.output_shape,
                self.axes,
                self._work[0],
            )
            result = self._transform_full(spectrum, 1, scale).real
        if result is not self.output_array:
            self.output_array[...] = result
        return self.output_array

    def execute(self):
        self._execute()
        return None

    def __call__(
        self,
        input_array=None,
        output_array=None,
        normalise_idft=_DEFAULT,
        ortho=_DEFAULT,
    ):
        if input_array is not None:
            converted = np.asarray(input_array, dtype=self.input_dtype)
            if converted.shape != self.input_shape:
                raise ValueError("input shape differs from the planned shape")
            self.input_array[...] = converted
        if output_array is not None:
            if output_array.shape != self.output_shape or output_array.dtype != self.output_dtype:
                raise ValueError("output shape or dtype differs from the plan")
            self.output_array = output_array
        normalise = self.normalise_idft if normalise_idft is _DEFAULT else bool(normalise_idft)
        use_ortho = self.ortho if ortho is _DEFAULT else bool(ortho)
        if use_ortho:
            scale = 1.0 / math.sqrt(self.N)
        elif normalise and self.direction == FFTW_BACKWARD:
            scale = 1.0 / self.N
        elif not normalise and self.direction == FFTW_FORWARD:
            scale = 1.0 / self.N
        else:
            scale = 1.0
        result = self._execute(scale)
        return result


def _alignment(array: np.ndarray) -> int:
    address = array.ctypes.data
    alignment = 1
    while alignment < 4096 and address % (alignment * 2) == 0:
        alignment *= 2
    return alignment


simd_alignment = 32


def empty_aligned(shape, dtype="float64", order="C", n=None):
    dtype = np.dtype(dtype)
    alignment = simd_alignment if n is None else int(n)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("n must be a positive power of two")
    count = math.prod(shape) if isinstance(shape, Iterable) else int(shape)
    storage = np.empty(count * dtype.itemsize + alignment, dtype=np.uint8)
    offset = (-storage.ctypes.data) % alignment
    return storage[offset : offset + count * dtype.itemsize].view(dtype).reshape(shape, order=order)


def zeros_aligned(shape, dtype="float64", order="C", n=None):
    result = empty_aligned(shape, dtype, order, n)
    result.fill(0)
    return result


def ones_aligned(shape, dtype="float64", order="C", n=None):
    result = empty_aligned(shape, dtype, order, n)
    result.fill(1)
    return result


def is_byte_aligned(array, n=None):
    alignment = simd_alignment if n is None else int(n)
    return np.asarray(array).ctypes.data % alignment == 0


def byte_align(array, n=None, dtype=None):
    source = np.asarray(array, dtype=dtype)
    alignment = simd_alignment if n is None else int(n)
    if is_byte_aligned(source, alignment):
        return source
    result = empty_aligned(source.shape, source.dtype, "F" if source.flags.f_contiguous else "C", alignment)
    result[...] = source
    return result


def n_byte_align_empty(shape, n, dtype="float64", order="C"):
    return empty_aligned(shape, dtype, order, n)


def n_byte_align(array, n, dtype=None):
    return byte_align(array, n, dtype)
