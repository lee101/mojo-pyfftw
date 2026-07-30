from __future__ import annotations

import numpy as np

from .. import builders


def _run(name, a, **kwargs):
    return getattr(builders, name)(a, **kwargs)()


def fft(a, n=None, axis=-1, norm=None, overwrite_input=False, planner_effort=None,
        threads=None, auto_align_input=True, auto_contiguous=True):
    return _run("fft", a, n=n, axis=axis, norm=norm, overwrite_input=overwrite_input,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input, auto_contiguous=auto_contiguous)


def ifft(a, n=None, axis=-1, norm=None, overwrite_input=False, planner_effort=None,
         threads=None, auto_align_input=True, auto_contiguous=True):
    return _run("ifft", a, n=n, axis=axis, norm=norm, overwrite_input=overwrite_input,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input, auto_contiguous=auto_contiguous)


def rfft(a, n=None, axis=-1, norm=None, overwrite_input=False, planner_effort=None,
         threads=None, auto_align_input=True, auto_contiguous=True):
    return _run("rfft", a, n=n, axis=axis, norm=norm, overwrite_input=overwrite_input,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input, auto_contiguous=auto_contiguous)


def irfft(a, n=None, axis=-1, norm=None, overwrite_input=False, planner_effort=None,
          threads=None, auto_align_input=True, auto_contiguous=True):
    return _run("irfft", a, n=n, axis=axis, norm=norm, overwrite_input=overwrite_input,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input, auto_contiguous=auto_contiguous)


def fftn(a, s=None, axes=None, norm=None, overwrite_input=False, planner_effort=None,
         threads=None, auto_align_input=True, auto_contiguous=True):
    return _run("fftn", a, s=s, axes=axes, norm=norm, overwrite_input=overwrite_input,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input, auto_contiguous=auto_contiguous)


def ifftn(a, s=None, axes=None, norm=None, overwrite_input=False, planner_effort=None,
          threads=None, auto_align_input=True, auto_contiguous=True):
    return _run("ifftn", a, s=s, axes=axes, norm=norm, overwrite_input=overwrite_input,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input, auto_contiguous=auto_contiguous)


def rfftn(a, s=None, axes=None, norm=None, overwrite_input=False, planner_effort=None,
          threads=None, auto_align_input=True, auto_contiguous=True):
    return _run("rfftn", a, s=s, axes=axes, norm=norm, overwrite_input=overwrite_input,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input, auto_contiguous=auto_contiguous)


def irfftn(a, s=None, axes=None, norm=None, overwrite_input=False, planner_effort=None,
           threads=None, auto_align_input=True, auto_contiguous=True):
    return _run("irfftn", a, s=s, axes=axes, norm=norm, overwrite_input=overwrite_input,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input, auto_contiguous=auto_contiguous)


def fft2(a, s=None, axes=(-2, -1), **kwargs):
    return fftn(a, s=s, axes=axes, **kwargs)


def ifft2(a, s=None, axes=(-2, -1), **kwargs):
    return ifftn(a, s=s, axes=axes, **kwargs)


def rfft2(a, s=None, axes=(-2, -1), **kwargs):
    return rfftn(a, s=s, axes=axes, **kwargs)


def irfft2(a, s=None, axes=(-2, -1), **kwargs):
    return irfftn(a, s=s, axes=axes, **kwargs)


def _swap_norm(norm):
    if norm in (None, "backward"):
        return "forward"
    if norm == "forward":
        return "backward"
    return norm


def hfft(a, n=None, axis=-1, norm=None, **kwargs):
    return irfft(np.conjugate(a), n=n, axis=axis, norm=_swap_norm(norm), **kwargs)


def ihfft(a, n=None, axis=-1, norm=None, **kwargs):
    return np.conjugate(rfft(a, n=n, axis=axis, norm=_swap_norm(norm), **kwargs))


fftfreq = np.fft.fftfreq
rfftfreq = np.fft.rfftfreq
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

__all__ = [
    "fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "rfft2", "irfft2",
    "fftn", "ifftn", "rfftn", "irfftn", "hfft", "ihfft",
    "fftfreq", "rfftfreq", "fftshift", "ifftshift",
]
