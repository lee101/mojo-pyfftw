from fft import (
    Ptr,
    prepare_workspace,
    rfft_power2,
    transform_axis,
    transform_vectors_range,
)


@export("mpf_prepare_workspace_f64")
def mpf_prepare_workspace_f64(
    scratch: Int,
    n: Int,
    direction: Int,
) abi("C") -> Int:
    if scratch == 0 or n <= 0 or (direction != -1 and direction != 1):
        return 0
    return prepare_workspace(
        Ptr(unsafe_from_address=scratch),
        n,
        direction,
    )


@export("mpf_rfft_power2_f64")
def mpf_rfft_power2_f64(
    src: Int,
    dst: Int,
    scratch: Int,
    total: Int,
    n: Int,
    scale: Float64,
    threads: Int,
) abi("C") -> Int:
    if (
        src == 0
        or dst == 0
        or scratch == 0
        or total <= 0
        or n <= 0
        or n > total
        or total % n != 0
        or threads <= 0
    ):
        return 0
    return rfft_power2(
        Ptr(unsafe_from_address=src),
        Ptr(unsafe_from_address=dst),
        Ptr(unsafe_from_address=scratch),
        total,
        n,
        scale,
        threads,
    )


@export("mpf_transform_axis_f64")
def mpf_transform_axis_f64(
    src: Int,
    dst: Int,
    scratch: Int,
    total: Int,
    n: Int,
    inner: Int,
    direction: Int,
    scale: Float64,
    threads: Int,
) abi("C") -> Int:
    if (
        src == 0
        or dst == 0
        or scratch == 0
        or total <= 0
        or n <= 0
        or inner <= 0
        or n > total
        or inner > total // n
        or total % (n * inner) != 0
        or (direction != -1 and direction != 1)
        or threads <= 0
    ):
        return 0
    return transform_axis(
        Ptr(unsafe_from_address=src),
        Ptr(unsafe_from_address=dst),
        Ptr(unsafe_from_address=scratch),
        total,
        n,
        inner,
        direction,
        scale,
        threads,
        src == dst,
    )


@export("mpf_transform_vectors_f64")
def mpf_transform_vectors_f64(
    src: Int,
    dst: Int,
    scratch: Int,
    total: Int,
    n: Int,
    direction: Int,
    scale: Float64,
    first: Int,
    last: Int,
) abi("C") -> Int:
    """Transform vectors [first, last) of a power-of-two, unit-stride axis.

    Vectors read and write disjoint spans of src and dst, so the Python shim
    can hand disjoint ranges to separate threads.
    """
    if (
        src == 0
        or dst == 0
        or scratch == 0
        or total <= 0
        or n <= 0
        or total % n != 0
        or (direction != -1 and direction != 1)
        or first < 0
        or last > total // n
        or first > last
    ):
        return 0
    transform_vectors_range(
        Ptr(unsafe_from_address=src),
        Ptr(unsafe_from_address=dst),
        Ptr(unsafe_from_address=scratch) + 2 * n,
        n,
        direction,
        scale,
        src == dst,
        first,
        last,
    )
    return 1
