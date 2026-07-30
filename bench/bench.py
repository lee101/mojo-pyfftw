from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pyfftw

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_pyfftw as mfftw  # noqa: E402


def timeit(function, repeat=7):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def plans(builder, values, threads=1, **kwargs):
    ours = getattr(mfftw.builders, builder)(
        values, planner_effort="FFTW_ESTIMATE", threads=threads, **kwargs
    )
    theirs = getattr(pyfftw.builders, builder)(
        values, planner_effort="FFTW_ESTIMATE", threads=threads, **kwargs
    )
    ours()
    theirs()
    return ours, theirs


def main():
    rng = np.random.default_rng(0)
    cases = []

    values = rng.normal(size=262_144) + 1j * rng.normal(size=262_144)
    cases.append(("complex fft, 262144", *plans("fft", values)))

    real = rng.normal(size=262_144)
    cases.append(("real rfft, 262144", *plans("rfft", real)))

    batched = rng.normal(size=(64, 4096)) + 1j * rng.normal(size=(64, 4096))
    cases.append(("batched fft, 64 x 4096", *plans("fft", batched, axis=-1)))

    prime = rng.normal(size=100_003) + 1j * rng.normal(size=100_003)
    cases.append(("Bluestein fft, 100003", *plans("fft", prime)))

    matrix = rng.normal(size=(512, 512)) + 1j * rng.normal(size=(512, 512))
    cases.append(("fft2, 512 x 512", *plans("fft2", matrix)))

    parallel_batch = rng.normal(size=(16, 65_536)) + 1j * rng.normal(
        size=(16, 65_536)
    )
    cases.append(
        ("batched fft, 16 x 65536, 1t", *plans("fft", parallel_batch, threads=1))
    )
    cases.append(
        ("batched fft, 16 x 65536, 4t", *plans("fft", parallel_batch, threads=4))
    )

    print(f"Machine: {platform.processor() or platform.machine()} ({platform.platform()})")
    print(f"{'case':<29} {'Mojo':>11} {'pyFFTW':>11} {'pyFFTW/Mojo':>14}")
    print("-" * 69)
    for name, ours, theirs in cases:
        mojo_time = timeit(ours)
        pyfftw_time = timeit(theirs)
        ratio = pyfftw_time / mojo_time
        label = "faster" if ratio > 1 else "slower"
        print(
            f"{name:<29} {mojo_time * 1e3:>9.3f} ms "
            f"{pyfftw_time * 1e3:>9.3f} ms {ratio:>9.2f}x {label}"
        )


if __name__ == "__main__":
    main()
