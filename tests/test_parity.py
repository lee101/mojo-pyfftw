import numpy as np
import pyfftw
import pytest

import mojo_pyfftw as mfftw
from mojo_pyfftw import _lib
from mojo_pyfftw.interfaces import numpy_fft as mojo_fft
from pyfftw.interfaces import numpy_fft as upstream_fft

rng = np.random.default_rng(20260730)


@pytest.mark.parametrize("n", [1, 2, 8, 64, 1024])
def test_power_of_two_fft_matches_pyfftw(n):
    values = rng.normal(size=n) + 1j * rng.normal(size=n)
    assert mojo_fft.fft(values) == pytest.approx(upstream_fft.fft(values), rel=2e-11, abs=2e-11)


@pytest.mark.parametrize("n", [3, 7, 31, 127, 1000])
def test_bluestein_fft_matches_pyfftw(n):
    values = rng.normal(size=n) + 1j * rng.normal(size=n)
    assert mojo_fft.fft(values) == pytest.approx(upstream_fft.fft(values), rel=2e-10, abs=2e-10)


@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
def test_fft_normalization_matches_pyfftw(norm):
    values = rng.normal(size=97) + 1j * rng.normal(size=97)
    assert mojo_fft.fft(values, norm=norm) == pytest.approx(upstream_fft.fft(values, norm=norm), rel=2e-11)
    assert mojo_fft.ifft(values, norm=norm) == pytest.approx(upstream_fft.ifft(values, norm=norm), rel=2e-11)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_complex_builder_preserves_pyfftw_dtype(dtype):
    values = rng.normal(size=32).astype(dtype)
    ours = mfftw.builders.fft(values)
    theirs = pyfftw.builders.fft(values)
    assert ours.input_dtype == theirs.input_dtype
    assert ours.output_dtype == theirs.output_dtype
    assert ours() == pytest.approx(theirs(), rel=2e-5 if dtype in (np.float32, np.complex64) else 1e-12)


@pytest.mark.parametrize("n", [7, 16, 63])
def test_rfft_and_irfft_match_pyfftw(n):
    values = rng.normal(size=n)
    spectrum = mojo_fft.rfft(values)
    assert spectrum == pytest.approx(upstream_fft.rfft(values), rel=2e-11, abs=2e-11)
    assert mojo_fft.irfft(spectrum, n=n) == pytest.approx(values, rel=2e-11, abs=2e-11)


def test_real_float32_dtype_and_values_match_pyfftw():
    values = rng.normal(size=35).astype(np.float32)
    ours = mojo_fft.rfft(values)
    theirs = upstream_fft.rfft(values)
    assert ours.dtype == theirs.dtype == np.complex64
    assert ours == pytest.approx(theirs, rel=2e-5, abs=2e-5)
    restored = mojo_fft.irfft(ours, n=values.size)
    assert restored.dtype == np.float32
    assert restored == pytest.approx(values, rel=2e-5, abs=2e-5)


def test_length_padding_and_truncation_match_pyfftw():
    values = rng.normal(size=19) + 1j * rng.normal(size=19)
    for n in (12, 27):
        assert mojo_fft.fft(values, n=n) == pytest.approx(upstream_fft.fft(values, n=n), rel=2e-11)


def test_non_last_axis_and_batch_match_pyfftw():
    values = rng.normal(size=(7, 5, 3)) + 1j * rng.normal(size=(7, 5, 3))
    assert mojo_fft.fft(values, axis=0) == pytest.approx(upstream_fft.fft(values, axis=0), rel=2e-11)
    assert mojo_fft.ifft(values, axis=1) == pytest.approx(upstream_fft.ifft(values, axis=1), rel=2e-11)


def test_simd_scalar_tail_matches_pyfftw():
    values = rng.normal(size=(1024, 5)) + 1j * rng.normal(size=(1024, 5))
    ours = mfftw.builders.fft(values, axis=0, threads=1)
    theirs = pyfftw.builders.fft(values, axis=0, threads=1)
    assert ours() == pytest.approx(theirs(), rel=2e-11, abs=2e-11)


@pytest.mark.parametrize("shape", [(2, 8192), (8, 65536), (1048576,)])
def test_parallel_threshold_paths_match_pyfftw(shape):
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ours = mfftw.builders.fft(values, threads=2)
    theirs = pyfftw.builders.fft(values, threads=1)
    assert ours() == pytest.approx(theirs(), rel=2e-11, abs=2e-11)


@pytest.mark.parametrize(
    ("shape", "axes"),
    [((4, 8), None), ((3, 5), None), ((2, 3, 7), (0, 2)), ((2, 4, 9), (-2, -1))],
)
def test_fftn_matches_pyfftw(shape, axes):
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    ours = mojo_fft.fftn(values, axes=axes)
    theirs = upstream_fft.fftn(values, axes=axes)
    assert ours == pytest.approx(theirs, rel=3e-11, abs=3e-11)
    assert mojo_fft.ifftn(ours, axes=axes) == pytest.approx(values, rel=3e-11, abs=3e-11)


def test_fftn_shape_and_axes_match_pyfftw():
    values = rng.normal(size=(5, 6, 7))
    ours = mojo_fft.fftn(values, s=(4, 9), axes=(0, 2))
    theirs = upstream_fft.fftn(values, s=(4, 9), axes=(0, 2))
    assert ours.shape == theirs.shape
    assert ours == pytest.approx(theirs, rel=3e-11, abs=3e-11)


@pytest.mark.parametrize(
    ("shape", "axes"),
    [((3, 8), (-1,)), ((3, 5), (-1,)), ((3, 4, 7), (1, 2)), ((2, 5, 6), (0, 2))],
)
def test_real_nd_roundtrip_and_parity(shape, axes):
    values = rng.normal(size=shape)
    transformed = mojo_fft.rfftn(values, axes=axes)
    reference = upstream_fft.rfftn(values, axes=axes)
    assert transformed == pytest.approx(reference, rel=3e-11, abs=3e-11)
    lengths = tuple(shape[axis] for axis in axes)
    restored = mojo_fft.irfftn(transformed, s=lengths, axes=axes)
    assert restored == pytest.approx(values, rel=3e-11, abs=3e-11)


def test_fft2_family_matches_pyfftw():
    real = rng.normal(size=(7, 10))
    complex_values = real + 1j * rng.normal(size=real.shape)
    transformed = mojo_fft.fft2(complex_values)
    assert transformed == pytest.approx(upstream_fft.fft2(complex_values), rel=3e-11)
    assert mojo_fft.ifft2(transformed) == pytest.approx(complex_values, rel=3e-11)
    half = mojo_fft.rfft2(real)
    assert half == pytest.approx(upstream_fft.rfft2(real), rel=3e-11)
    assert mojo_fft.irfft2(half, s=real.shape) == pytest.approx(real, rel=3e-11)


def test_hfft_and_ihfft_match_pyfftw():
    hermitian_half = rng.normal(size=17) + 1j * rng.normal(size=17)
    hermitian_half[0] = hermitian_half[0].real
    hermitian_half[-1] = hermitian_half[-1].real
    assert mojo_fft.hfft(hermitian_half) == pytest.approx(upstream_fft.hfft(hermitian_half), rel=2e-11)
    real = rng.normal(size=32)
    assert mojo_fft.ihfft(real) == pytest.approx(upstream_fft.ihfft(real), rel=2e-11)


def test_raw_fftw_execute_is_unnormalized_like_upstream():
    source = mfftw.empty_aligned(32, dtype="complex128")
    source[:] = rng.normal(size=32) + 1j * rng.normal(size=32)
    output = mfftw.empty_aligned(32, dtype="complex128")
    plan = mfftw.FFTW(source, output, direction="FFTW_BACKWARD", flags=("FFTW_ESTIMATE",))
    assert plan.execute() is None
    got = output.copy()
    upstream_source = pyfftw.byte_align(source.copy())
    upstream_output = pyfftw.empty_aligned(32, dtype="complex128")
    upstream_plan = pyfftw.FFTW(
        upstream_source,
        upstream_output,
        direction="FFTW_BACKWARD",
        flags=("FFTW_ESTIMATE",),
    )
    assert upstream_plan.execute() is None
    reference = upstream_output
    assert got == pytest.approx(reference, rel=2e-11)


def test_fftw_call_normalizes_inverse():
    source = mfftw.empty_aligned(19, dtype="complex128")
    source[:] = rng.normal(size=19) + 1j * rng.normal(size=19)
    output = mfftw.empty_aligned(19, dtype="complex128")
    plan = mfftw.FFTW(source, output, direction="FFTW_BACKWARD", flags=("FFTW_ESTIMATE",))
    assert plan() == pytest.approx(upstream_fft.ifft(source), rel=2e-11)


def test_fftw_properties_and_array_updates():
    source = mfftw.empty_aligned((3, 8), dtype="complex128")
    output = mfftw.empty_aligned((3, 8), dtype="complex128")
    plan = mfftw.FFTW(source, output, axes=(1,), flags=("FFTW_ESTIMATE",))
    assert plan.axes == (1,)
    assert plan.direction == "FFTW_FORWARD"
    assert plan.N == 8
    assert plan.input_shape == plan.output_shape == (3, 8)
    assert plan.input_dtype == plan.output_dtype == np.dtype("complex128")
    replacement_in = mfftw.empty_aligned((3, 8), dtype="complex128")
    replacement_out = mfftw.empty_aligned((3, 8), dtype="complex128")
    replacement_in[:] = rng.normal(size=(3, 8))
    plan.update_arrays(replacement_in, replacement_out)
    assert plan.get_input_array() is replacement_in
    assert plan.get_output_array() is replacement_out
    assert plan() == pytest.approx(upstream_fft.fft(replacement_in, axis=1), rel=2e-11)


def test_call_accepts_new_input_and_output():
    initial = mfftw.zeros_aligned(16, dtype="complex128")
    target = mfftw.empty_aligned(16, dtype="complex128")
    plan = mfftw.FFTW(initial, target, flags=("FFTW_ESTIMATE",))
    values = rng.normal(size=16) + 1j * rng.normal(size=16)
    new_output = mfftw.empty_aligned(16, dtype="complex128")
    returned = plan(values, new_output)
    assert returned is new_output
    assert returned == pytest.approx(upstream_fft.fft(values), rel=2e-11)


@pytest.mark.parametrize("order", ["C", "F"])
def test_aligned_allocators(order):
    for factory, fill in ((mfftw.empty_aligned, None), (mfftw.zeros_aligned, 0), (mfftw.ones_aligned, 1)):
        array = factory((5, 7), dtype="float32", order=order, n=64)
        assert mfftw.is_byte_aligned(array, 64)
        assert array.flags.c_contiguous if order == "C" else array.flags.f_contiguous
        if fill is not None:
            assert np.all(array == fill)


def test_byte_align_matches_values_and_dtype():
    storage = np.arange(101, dtype=np.uint8)
    misaligned = storage[1:].view(np.float32)
    aligned = mfftw.byte_align(misaligned, n=32, dtype=np.float64)
    assert mfftw.is_byte_aligned(aligned, 32)
    assert aligned.dtype == np.float64
    assert aligned == pytest.approx(misaligned)


def test_frequency_and_shift_helpers_match_numpy():
    values = np.arange(12).reshape(3, 4)
    assert np.array_equal(mojo_fft.fftfreq(9, 0.2), np.fft.fftfreq(9, 0.2))
    assert np.array_equal(mojo_fft.rfftfreq(10, 0.2), np.fft.rfftfreq(10, 0.2))
    assert np.array_equal(mojo_fft.fftshift(values), np.fft.fftshift(values))
    assert np.array_equal(mojo_fft.ifftshift(values), np.fft.ifftshift(values))


def test_impulse_and_constant_published_dft_vectors():
    impulse = np.array([1, 0, 0, 0], dtype=np.complex128)
    constant = np.ones(4, dtype=np.complex128)
    assert np.array_equal(mojo_fft.fft(impulse), np.ones(4))
    assert mojo_fft.fft(constant) == pytest.approx([4, 0, 0, 0], abs=1e-15)


def test_invalid_duplicate_axes_and_real_shape_raise():
    with pytest.raises(ValueError):
        mfftw.builders.fftn(np.ones((3, 4)), axes=(0, 0))
    with pytest.raises(ValueError):
        mfftw.FFTW(np.ones(8), np.empty(8, dtype=np.complex128))


def test_ffi_rejects_unsafe_buffers_before_calling_mojo():
    source = np.ones(8, dtype=np.complex128)
    scratch = np.empty(_lib._scratch_size(8), dtype=np.complex128)
    with pytest.raises(TypeError):
        _lib.transform_axis(source, 0, -1, result=np.empty(8, dtype=np.complex64))
    with pytest.raises(ValueError):
        _lib.transform_axis(source, 0, -1, result=np.empty(7, dtype=np.complex128))
    with pytest.raises(ValueError):
        _lib.transform_axis(source, 0, -1, scratch=scratch[:-1])
    overlapping = np.ones(9, dtype=np.complex128)
    with pytest.raises(ValueError, match="must not overlap"):
        _lib.transform_axis(overlapping[:-1], 0, -1, result=overlapping[1:])
    readonly = np.empty_like(source)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        _lib.transform_axis(source, 0, -1, result=readonly)


def test_ffi_accepts_exact_in_place_complex_transform():
    values = (rng.normal(size=16) + 1j * rng.normal(size=16)).astype(np.complex128)
    expected = upstream_fft.fft(values)
    scratch = np.empty(_lib._scratch_size(values.size), dtype=np.complex128)
    _lib.transform_axis(values, 0, -1, result=values, scratch=scratch)
    assert values == pytest.approx(expected, rel=2e-11, abs=2e-11)


def test_unsupported_long_double_is_not_silently_narrowed():
    with pytest.raises(TypeError, match="long-double"):
        mfftw.builders.fft(np.ones(8, dtype=np.clongdouble))
    with pytest.raises(TypeError, match="long-double"):
        mfftw.builders.rfft(np.ones(8, dtype=np.longdouble))


def test_wisdom_compatibility_stubs_are_explicit():
    assert mfftw.export_wisdom() == (b"", b"", b"")
    assert mfftw.import_wisdom((b"a", b"b", b"c")) == (False, False, False)
    with pytest.raises(ValueError):
        mfftw.import_wisdom((b"only one",))
    assert mfftw.forget_wisdom() is None


def test_legacy_alignment_aliases():
    array = mfftw.n_byte_align_empty(17, 64, dtype="float64")
    assert mfftw.is_byte_aligned(array, 64)
    misaligned = np.arange(18, dtype=np.float64)[1:]
    aligned = mfftw.n_byte_align(misaligned, 64)
    assert mfftw.is_byte_aligned(aligned, 64)
    assert np.array_equal(aligned, misaligned)
