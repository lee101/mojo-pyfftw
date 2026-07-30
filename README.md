# mojo-pyfftw

FFT planning and execution in [Mojo](https://www.modular.com/mojo), callable from
Python through a pyFFTW-shaped API.

This is a standalone implementation, not bindings to FFTW. Power-of-two transforms
use an iterative radix-2 Cooley-Tukey kernel; every other positive length uses
Bluestein's algorithm backed by the same radix-2 convolution kernel. The covered API
is tested numerically and behaviorally against pyFFTW 0.15.1.

```python
import numpy as np
import mojo_pyfftw as pyfftw

x = np.sin(2 * np.pi * np.arange(16) / 16)
forward = pyfftw.builders.rfft(x)
spectrum = forward()
restored = pyfftw.builders.irfft(spectrum, n=x.size)()

np.testing.assert_allclose(restored, x, atol=1e-12)
```

## Coverage

The following public surface mirrors pyFFTW names and signatures for the covered
arguments:

| area | covered |
| --- | --- |
| plans | `FFTW` complex-to-complex, real-to-complex, and complex-to-real plans; arbitrary positive lengths; selected axes; batched and multidimensional arrays; `execute`, callable normalization, `update_arrays`, and plan properties |
| builders | `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn` |
| NumPy interface | the same transform family, plus `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, and `ifftshift` |
| data types | `float32`, `float64`, `complex64`, and `complex128`; 32-bit inputs retain pyFFTW's 32-bit output dtype and are computed internally at 64-bit precision |
| allocation | `empty_aligned`, `zeros_aligned`, `ones_aligned`, `byte_align`, `is_byte_aligned`, and the legacy `n_byte_align*` aliases |
| normalization | `None`/`"backward"`, `"forward"`, and `"ortho"` |

Not covered are DCT/DST and other real-to-real FFTW schemes, long-double transforms,
the SciPy interfaces, interface caching, and direct zero-copy execution over
arbitrary byte strides. Non-contiguous inputs are accepted but staged through a
contiguous plan buffer. `threads` selects thresholded CPU parallel execution for
large power-of-two transforms; planner flags and `planning_timelimit` remain
compatibility arguments. FFTW wisdom has no Mojo equivalent: exported wisdom is
empty and imports report that no entries were consumed.

## Install and run

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles the single C-ABI compilation unit to
`dist/libmojo-pyfftw.so`. Importing the Python package also rebuilds a missing or
stale library. Set `MOJO_PYFFTW_LIB` to use an already-built shared library at a
different path.

The Python package is named `mojo_pyfftw`, allowing an explicit replacement import:

```python
import mojo_pyfftw as pyfftw
from mojo_pyfftw.interfaces import numpy_fft
```

## Benchmarks

Measured with `pixi run bench` on this machine (`x86_64`,
Linux 6.8.0-136-generic). The first five rows use one execution thread for both
implementations; the final pair shows the same large batch with one and four
threads. These are the best of seven warm planned executions; allocation and
initialization of plan work, twiddle, and scratch buffers is outside the timed
region.

| case | Mojo | pyFFTW 0.15.1 | pyFFTW/Mojo |
| --- | ---: | ---: | ---: |
| complex FFT, 262144 | 8.887 ms | 3.831 ms | 0.43x, Mojo slower |
| real RFFT, 262144 | 3.965 ms | 1.475 ms | 0.37x, Mojo slower |
| batched FFT, 64 x 4096 | 11.644 ms | 1.982 ms | 0.17x, Mojo slower |
| Bluestein FFT, 100003 | 29.029 ms | 11.387 ms | 0.39x, Mojo slower |
| FFT2, 512 x 512 | 11.530 ms | 10.101 ms | 0.88x, Mojo slower |
| batched FFT, 16 x 65536, 1 thread | 44.968 ms | 18.775 ms | 0.42x, Mojo slower |
| batched FFT, 16 x 65536, 4 threads | 25.916 ms | 7.686 ms | 0.30x, Mojo slower |

pyFFTW is faster on every case in this run; its generated plans use mature
architecture-specific kernels. The four-thread row uses four threads on both sides
and shows Mojo's thresholded parallel path reducing its large-batch execution time.

No GPU path is included.

## How it works

Python owns every input, output, work, twiddle, and scratch allocation. A plan
preallocates two contiguous complex128 work arrays and one workspace per transformed
axis, then reuses them on every execution. Twiddles and the Bluestein convolution
kernel are initialized once during planning. Complex values use NumPy's interleaved
real/imaginary memory layout.

The shared-library boundary is one `ctypes` call per axis. NumPy buffers cross the C
ABI zero-copy as 64-bit addresses, and the exported Mojo function reconstructs each
as `UnsafePointer[Float64, AnyOrigin[mut=True]]`. Shape information crosses as
scalar integers. Contiguous power-of-two transforms execute directly in the
destination; strided transforms vectorize across adjacent independent columns.
Power-of-two float64 RFFT packs pairs into a half-sized complex FFT and reconstructs
the Hermitian output with SIMD. Arbitrary-length Bluestein transforms reuse plan
scratch.

For non-power-of-two lengths, the plan workspace holds Bluestein's chirped input,
pretransformed convolution kernel, and radix-2 twiddles at the next power-of-two
length. Forward transforms are unnormalized. Inverse, forward-normalized, and
orthonormal scaling is fused into the final Mojo store.

## License

MIT
