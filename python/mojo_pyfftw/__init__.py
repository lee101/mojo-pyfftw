from . import builders, config, interfaces
from .core import (
    FFTW,
    FFTW_BACKWARD,
    FFTW_FORWARD,
    byte_align,
    empty_aligned,
    is_byte_aligned,
    n_byte_align,
    n_byte_align_empty,
    ones_aligned,
    simd_alignment,
    zeros_aligned,
)

__version__ = "0.1.0"


def export_wisdom():
    return (b"", b"", b"")


def import_wisdom(wisdom):
    if len(wisdom) != 3:
        raise ValueError("wisdom must contain double, single, and long-double entries")
    return (False, False, False)


def forget_wisdom():
    return None


__all__ = [
    "FFTW",
    "FFTW_FORWARD",
    "FFTW_BACKWARD",
    "builders",
    "interfaces",
    "config",
    "empty_aligned",
    "zeros_aligned",
    "ones_aligned",
    "byte_align",
    "is_byte_aligned",
    "n_byte_align_empty",
    "n_byte_align",
    "simd_alignment",
    "export_wisdom",
    "import_wisdom",
    "forget_wisdom",
]
