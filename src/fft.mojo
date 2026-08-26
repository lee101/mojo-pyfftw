from std.bit import bit_reverse
from std.math import cos, sin
from std.runtime import initialize_runtime
from std.runtime.asyncrt import TaskGroup
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime PI = 3.141592653589793238462643383279502884


@always_inline
def parallelize[FuncType: def(Int) -> None](
    func: FuncType, num_work_items: Int, num_workers: Int
):
    """Run work on the Mojo runtime without depending on the MAX package."""
    if num_work_items <= 0:
        return
    var workers = min(num_work_items, num_workers)
    if workers <= 1:
        for i in range(num_work_items):
            func(i)
        return

    initialize_runtime()
    var chunk_size, extra_items = divmod(num_work_items, workers)

    @always_inline
    async def task(worker: Int) {imm}:
        var start = worker * chunk_size + min(worker, extra_items)
        for i in range(chunk_size + Int(worker < extra_items)):
            func(start + i)

    var tasks = TaskGroup()
    for worker in range(workers):
        tasks.create_task(task(worker))
    tasks.wait()


def is_power_of_two(n: Int) -> Bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: Int) -> Int:
    var value = 1
    while value < n:
        value *= 2
    return value


def swap_complex(data: Ptr, a: Int, b: Int):
    var ar = data[2 * a]
    var ai = data[2 * a + 1]
    data[2 * a] = data[2 * b]
    data[2 * a + 1] = data[2 * b + 1]
    data[2 * b] = ar
    data[2 * b + 1] = ai


def prepare_twiddles(twiddles: Ptr, n: Int):
    var offset = 0
    var length = 2
    while length <= n:
        var half = length >> 1
        for k in range(half):
            var angle = -2.0 * PI * Float64(k) / Float64(length)
            twiddles[2 * (offset + k)] = cos(angle)
            twiddles[2 * (offset + k) + 1] = sin(angle)
        offset += half
        length *= 2


def radix2_stages(
    data: Ptr,
    twiddles: Ptr,
    base_start: Int,
    base_end: Int,
    start_length: Int,
    end_length: Int,
    direction: Int,
):
    comptime W = simd_width_of[DType.float64]()
    comptime C = W // 2
    var length = start_length
    if length == 2 and end_length >= 4:
        var base = base_start
        while base < base_end:
            var ar = data[2 * base]
            var ai = data[2 * base + 1]
            var br = data[2 * (base + 1)]
            var bi = data[2 * (base + 1) + 1]
            var cr = data[2 * (base + 2)]
            var ci = data[2 * (base + 2) + 1]
            var dr = data[2 * (base + 3)]
            var di = data[2 * (base + 3) + 1]
            var p0r = ar + br
            var p0i = ai + bi
            var p1r = ar - br
            var p1i = ai - bi
            var q0r = cr + dr
            var q0i = ci + di
            var q1r = cr - dr
            var q1i = ci - di
            data[2 * base] = p0r + q0r
            data[2 * base + 1] = p0i + q0i
            data[2 * (base + 2)] = p0r - q0r
            data[2 * (base + 2) + 1] = p0i - q0i
            data[2 * (base + 1)] = p1r - Float64(direction) * q1i
            data[2 * (base + 1) + 1] = p1i + Float64(direction) * q1r
            data[2 * (base + 3)] = p1r + Float64(direction) * q1i
            data[2 * (base + 3) + 1] = p1i - Float64(direction) * q1r
            base += 4
        length = 8
    while 2 * length <= end_length:
        var quarter = length >> 1
        var first_twiddle_offset = quarter - 1
        var second_twiddle_offset = length - 1
        var base = base_start
        while base < base_end:
            var k = 0
            while k + C <= quarter:
                var a = data.load[width=W](2 * (base + k)).deinterleave()
                var b = data.load[width=W](
                    2 * (base + quarter + k)
                ).deinterleave()
                var c = data.load[width=W](
                    2 * (base + length + k)
                ).deinterleave()
                var d = data.load[width=W](
                    2 * (base + length + quarter + k)
                ).deinterleave()
                var w1 = twiddles.load[width=W](
                    2 * (first_twiddle_offset + k)
                ).deinterleave()
                var w1i = w1[1] * Float64(-direction)
                var bwr = b[0] * w1[0] - b[1] * w1i
                var bwi = b[0] * w1i + b[1] * w1[0]
                var dwr = d[0] * w1[0] - d[1] * w1i
                var dwi = d[0] * w1i + d[1] * w1[0]
                var p0r = a[0] + bwr
                var p0i = a[1] + bwi
                var p1r = a[0] - bwr
                var p1i = a[1] - bwi
                var q0r = c[0] + dwr
                var q0i = c[1] + dwi
                var q1r = c[0] - dwr
                var q1i = c[1] - dwi
                var w20 = twiddles.load[width=W](
                    2 * (second_twiddle_offset + k)
                ).deinterleave()
                var w21 = twiddles.load[width=W](
                    2 * (second_twiddle_offset + quarter + k)
                ).deinterleave()
                var w20i = w20[1] * Float64(-direction)
                var w21i = w21[1] * Float64(-direction)
                var q0wr = q0r * w20[0] - q0i * w20i
                var q0wi = q0r * w20i + q0i * w20[0]
                var q1wr = q1r * w21[0] - q1i * w21i
                var q1wi = q1r * w21i + q1i * w21[0]
                data.store(
                    2 * (base + k),
                    (p0r + q0wr).interleave(p0i + q0wi),
                )
                data.store(
                    2 * (base + length + k),
                    (p0r - q0wr).interleave(p0i - q0wi),
                )
                data.store(
                    2 * (base + quarter + k),
                    (p1r + q1wr).interleave(p1i + q1wi),
                )
                data.store(
                    2 * (base + length + quarter + k),
                    (p1r - q1wr).interleave(p1i - q1wi),
                )
                k += C
            while k < quarter:
                var ai = base + k
                var bi = ai + quarter
                var ci = ai + length
                var di = ci + quarter
                var ar = data[2 * ai]
                var aim = data[2 * ai + 1]
                var br = data[2 * bi]
                var bim = data[2 * bi + 1]
                var cr = data[2 * ci]
                var cim = data[2 * ci + 1]
                var dr = data[2 * di]
                var dim = data[2 * di + 1]
                var w1r = twiddles[2 * (first_twiddle_offset + k)]
                var w1i = twiddles[2 * (first_twiddle_offset + k) + 1] * Float64(-direction)
                var bwr = br * w1r - bim * w1i
                var bwi = br * w1i + bim * w1r
                var dwr = dr * w1r - dim * w1i
                var dwi = dr * w1i + dim * w1r
                var p0r = ar + bwr
                var p0i = aim + bwi
                var p1r = ar - bwr
                var p1i = aim - bwi
                var q0r = cr + dwr
                var q0i = cim + dwi
                var q1r = cr - dwr
                var q1i = cim - dwi
                var w20r = twiddles[2 * (second_twiddle_offset + k)]
                var w20i = twiddles[2 * (second_twiddle_offset + k) + 1] * Float64(-direction)
                var w21r = twiddles[2 * (second_twiddle_offset + quarter + k)]
                var w21i = twiddles[2 * (second_twiddle_offset + quarter + k) + 1] * Float64(-direction)
                var q0wr = q0r * w20r - q0i * w20i
                var q0wi = q0r * w20i + q0i * w20r
                var q1wr = q1r * w21r - q1i * w21i
                var q1wi = q1r * w21i + q1i * w21r
                data[2 * ai] = p0r + q0wr
                data[2 * ai + 1] = p0i + q0wi
                data[2 * ci] = p0r - q0wr
                data[2 * ci + 1] = p0i - q0wi
                data[2 * bi] = p1r + q1wr
                data[2 * bi + 1] = p1i + q1wi
                data[2 * di] = p1r - q1wr
                data[2 * di + 1] = p1i - q1wi
                k += 1
            base += 2 * length
        length *= 4

    if length <= end_length:
        var half = length >> 1
        var twiddle_offset = half - 1
        var base = base_start
        while base < base_end:
            var k = 0
            while k + C <= half:
                var even = base + k
                var odd = even + half
                var u = data.load[width=W](2 * even).deinterleave()
                var x = data.load[width=W](2 * odd).deinterleave()
                var w = twiddles.load[width=W](
                    2 * (twiddle_offset + k)
                ).deinterleave()
                var wi = w[1] * Float64(-direction)
                var vr = x[0] * w[0] - x[1] * wi
                var vi = x[0] * wi + x[1] * w[0]
                data.store(2 * even, (u[0] + vr).interleave(u[1] + vi))
                data.store(2 * odd, (u[0] - vr).interleave(u[1] - vi))
                k += C
            while k < half:
                var even = base + k
                var odd = even + half
                var ur = data[2 * even]
                var ui = data[2 * even + 1]
                var xr = data[2 * odd]
                var xi = data[2 * odd + 1]
                var wr = twiddles[2 * (twiddle_offset + k)]
                var wi = twiddles[2 * (twiddle_offset + k) + 1] * Float64(-direction)
                var vr = xr * wr - xi * wi
                var vi = xr * wi + xi * wr
                data[2 * even] = ur + vr
                data[2 * even + 1] = ui + vi
                data[2 * odd] = ur - vr
                data[2 * odd + 1] = ui - vi
                k += 1
            base += length


def radix2(
    data: Ptr, twiddles: Ptr, n: Int, direction: Int, threads: Int = 1
):
    var j = 0
    for i in range(1, n):
        var bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            swap_complex(data, i, j)

    if n <= 512:
        var length = 2
        while length <= n:
            var angle = Float64(direction) * 2.0 * PI / Float64(length)
            var step_r = cos(angle)
            var step_i = sin(angle)
            var base = 0
            var half = length >> 1
            while base < n:
                var wr = 1.0
                var wi = 0.0
                for k in range(half):
                    var even = base + k
                    var odd = even + half
                    var ur = data[2 * even]
                    var ui = data[2 * even + 1]
                    var xr = data[2 * odd]
                    var xi = data[2 * odd + 1]
                    var vr = xr * wr - xi * wi
                    var vi = xr * wi + xi * wr
                    data[2 * even] = ur + vr
                    data[2 * even + 1] = ui + vi
                    data[2 * odd] = ur - vr
                    data[2 * odd + 1] = ui - vi
                    var next_wr = wr * step_r - wi * step_i
                    wi = wr * step_i + wi * step_r
                    wr = next_wr
                base += length
            length *= 2
        return

    var parallel_block = 4096
    if threads > 1 and n >= 1048576:
        var blocks = n // parallel_block

        @always_inline
        def transform_block(
            block: Int,
        ) {imm data, imm twiddles, imm parallel_block, imm direction}:
            var start = block * parallel_block
            radix2_stages(
                data,
                twiddles,
                start,
                start + parallel_block,
                2,
                parallel_block,
                direction,
            )

        parallelize(transform_block, blocks, min(threads, blocks))
        radix2_stages(
            data,
            twiddles,
            0,
            n,
            2 * parallel_block,
            n,
            direction,
        )
    else:
        radix2_stages(data, twiddles, 0, n, 2, n, direction)


def bit_reversed_copy(src: Ptr, dst: Ptr, source: Int, target: Int, n: Int):
    var bits = 0
    var bit_count = n
    while bit_count > 1:
        bits += 1
        bit_count >>= 1
    for i in range(n):
        var reversed = Int(bit_reverse(UInt64(i)) >> UInt64(64 - bits))
        dst[2 * (target + reversed)] = src[2 * (source + i)]
        dst[2 * (target + reversed) + 1] = src[2 * (source + i) + 1]


def radix2_ordered(
    data: Ptr, twiddles: Ptr, n: Int, direction: Int, threads: Int
):
    var parallel_block = 4096
    if threads > 1 and n >= 262144:
        var blocks = n // parallel_block

        @always_inline
        def transform_block(
            block: Int,
        ) {imm data, imm twiddles, imm parallel_block, imm direction}:
            var start = block * parallel_block
            radix2_stages(
                data,
                twiddles,
                start,
                start + parallel_block,
                2,
                parallel_block,
                direction,
            )

        parallelize(transform_block, blocks, min(threads, blocks))
        radix2_stages(
            data,
            twiddles,
            0,
            n,
            2 * parallel_block,
            n,
            direction,
        )
    else:
        radix2_stages(data, twiddles, 0, n, 2, n, direction)


def swap_complex_blocks(data: Ptr, a: Int, b: Int, count: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= 2 * count:
        var av = data.load[width=W](2 * a + i)
        var bv = data.load[width=W](2 * b + i)
        data.store(2 * a + i, bv)
        data.store(2 * b + i, av)
        i += W
    while i < 2 * count:
        var value = data[2 * a + i]
        data[2 * a + i] = data[2 * b + i]
        data[2 * b + i] = value
        i += 1


def radix2_strided(
    data: Ptr,
    twiddles: Ptr,
    n: Int,
    inner: Int,
    outer: Int,
    direction: Int,
):
    for left in range(outer):
        var outer_base = left * n * inner
        var j = 0
        for i in range(1, n):
            var bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                swap_complex_blocks(
                    data,
                    outer_base + i * inner,
                    outer_base + j * inner,
                    inner,
                )

    comptime W = simd_width_of[DType.float64]()
    comptime C = W // 2
    var length = 2
    var twiddle_offset = 0
    while length <= n:
        var half = length >> 1
        for left in range(outer):
            var outer_base = left * n * inner
            var base = 0
            while base < n:
                for k in range(half):
                    var even = outer_base + (base + k) * inner
                    var odd = even + half * inner
                    var wr = twiddles[2 * (twiddle_offset + k)]
                    var wi = twiddles[2 * (twiddle_offset + k) + 1] * Float64(
                        -direction
                    )
                    var right = 0
                    while right + C <= inner:
                        var u = data.load[width=W](
                            2 * (even + right)
                        ).deinterleave()
                        var x = data.load[width=W](
                            2 * (odd + right)
                        ).deinterleave()
                        var vr = x[0] * wr - x[1] * wi
                        var vi = x[0] * wi + x[1] * wr
                        data.store(
                            2 * (even + right),
                            (u[0] + vr).interleave(u[1] + vi),
                        )
                        data.store(
                            2 * (odd + right),
                            (u[0] - vr).interleave(u[1] - vi),
                        )
                        right += C
                    while right < inner:
                        var ur = data[2 * (even + right)]
                        var ui = data[2 * (even + right) + 1]
                        var xr = data[2 * (odd + right)]
                        var xi = data[2 * (odd + right) + 1]
                        var vr = xr * wr - xi * wi
                        var vi = xr * wi + xi * wr
                        data[2 * (even + right)] = ur + vr
                        data[2 * (even + right) + 1] = ui + vi
                        data[2 * (odd + right)] = ur - vr
                        data[2 * (odd + right) + 1] = ui - vi
                        right += 1
                base += length
        twiddle_offset += half
        length *= 2


def clear_complex(data: Ptr, n: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var zero = SIMD[DType.float64, W](0.0)
    while i + W <= 2 * n:
        data.store(i, zero)
        i += W
    while i < 2 * n:
        data[i] = 0.0
        i += 1


def copy_complex(src: Ptr, dst: Ptr, source: Int, target: Int, n: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= 2 * n:
        dst.store(2 * target + i, src.load[width=W](2 * source + i))
        i += W
    while i < 2 * n:
        dst[2 * target + i] = src[2 * source + i]
        i += 1


def scale_complex(data: Ptr, base: Int, n: Int, scale: Float64):
    if scale == 1.0:
        return
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= 2 * n:
        var values = data.load[width=W](2 * base + i)
        data.store(2 * base + i, values * scale)
        i += W
    while i < 2 * n:
        data[2 * base + i] *= scale
        i += 1


def pack_real(src: Ptr, work: Ptr, source: Int, n: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        work.store(i, src.load[width=W](source + i))
        i += W
    while i < n:
        work[i] = src[source + i]
        i += 1


def pack_real_bit_reversed(src: Ptr, work: Ptr, source: Int, n: Int):
    var bits = 0
    var bit_count = n
    while bit_count > 1:
        bits += 1
        bit_count >>= 1
    for i in range(n):
        var reversed = Int(bit_reverse(UInt64(i)) >> UInt64(64 - bits))
        work[2 * reversed] = src[source + 2 * i]
        work[2 * reversed + 1] = src[source + 2 * i + 1]


def rfft_power2(
    src: Ptr,
    dst: Ptr,
    scratch: Ptr,
    total: Int,
    n: Int,
    scale: Float64,
    threads: Int,
) -> Int:
    if n <= 0 or total % n != 0 or not is_power_of_two(n):
        return 0
    if n == 1:
        for vector in range(total):
            dst[2 * vector] = src[vector] * scale
            dst[2 * vector + 1] = 0.0
        return 1

    var vectors = total // n
    var m = n >> 1
    var output_n = m + 1
    var twiddles = scratch + 2 * n
    var final_twiddles = twiddles + 2 * (m - 1)
    for vector in range(vectors):
        pack_real_bit_reversed(src, scratch, vector * n, m)
        radix2_ordered(scratch, twiddles, m, -1, threads)

        var y0r = scratch[0]
        var y0i = scratch[1]
        var target = vector * output_n
        dst[2 * target] = (y0r + y0i) * scale
        dst[2 * target + 1] = 0.0
        dst[2 * (target + m)] = (y0r - y0i) * scale
        dst[2 * (target + m) + 1] = 0.0

        comptime W = simd_width_of[DType.float64]()
        comptime C = W // 2
        var k = 1
        while k + C <= m:
            var a = scratch.load[width=W](2 * k).deinterleave()
            var b = scratch.load[width=W](
                2 * (m - k - C + 1)
            ).deinterleave()
            var br = b[0].reversed()
            var bi = -b[1].reversed()
            var w = final_twiddles.load[width=W](2 * k).deinterleave()
            var sum_r = a[0] + br
            var sum_i = a[1] + bi
            var diff_r = a[0] - br
            var diff_i = a[1] - bi
            var rotated_r = w[0] * diff_i + w[1] * diff_r
            var rotated_i = w[1] * diff_i - w[0] * diff_r
            dst.store(
                2 * (target + k),
                (0.5 * (sum_r + rotated_r) * scale).interleave(
                    0.5 * (sum_i + rotated_i) * scale
                ),
            )
            k += C
        while k < m:
            var ar = scratch[2 * k]
            var ai = scratch[2 * k + 1]
            var br = scratch[2 * (m - k)]
            var bi = -scratch[2 * (m - k) + 1]
            var sum_r = ar + br
            var sum_i = ai + bi
            var diff_r = ar - br
            var diff_i = ai - bi
            var wr = final_twiddles[2 * k]
            var wi = final_twiddles[2 * k + 1]
            var rotated_r = wr * diff_i + wi * diff_r
            var rotated_i = wi * diff_i - wr * diff_r
            dst[2 * (target + k)] = 0.5 * (sum_r + rotated_r) * scale
            dst[2 * (target + k) + 1] = 0.5 * (
                sum_i + rotated_i
            ) * scale
            k += 1
    return 1


def prepare_bluestein(
    kernel: Ptr, twiddles: Ptr, chirp: Ptr, n: Int, m: Int, direction: Int
):
    clear_complex(kernel, m)
    kernel[0] = 1.0
    chirp[0] = 1.0
    chirp[1] = 0.0
    for j in range(1, n):
        var jf = Float64(j)
        var angle = -Float64(direction) * PI * jf * jf / Float64(n)
        var re = cos(angle)
        var im = sin(angle)
        kernel[2 * j] = re
        kernel[2 * j + 1] = im
        kernel[2 * (m - j)] = re
        kernel[2 * (m - j) + 1] = im
        chirp[2 * j] = re
        chirp[2 * j + 1] = -im
    radix2(kernel, twiddles, m, -1)


def bluestein_vector(
    src: Ptr,
    dst: Ptr,
    work: Ptr,
    kernel: Ptr,
    twiddles: Ptr,
    chirp: Ptr,
    base: Int,
    n: Int,
    m: Int,
    inner: Int,
    direction: Int,
    scale: Float64,
    threads: Int,
):
    clear_complex(work, m)
    comptime W = simd_width_of[DType.float64]()
    comptime C = W // 2
    var j = 0
    if inner == 1:
        while j + C <= n:
            var x = src.load[width=W](2 * (base + j)).deinterleave()
            var w = chirp.load[width=W](2 * j).deinterleave()
            work.store(
                2 * j,
                (x[0] * w[0] - x[1] * w[1]).interleave(
                    x[0] * w[1] + x[1] * w[0]
                ),
            )
            j += C
    while j < n:
        var source = base + j * inner
        var xr = src[2 * source]
        var xi = src[2 * source + 1]
        var wr = chirp[2 * j]
        var wi = chirp[2 * j + 1]
        work[2 * j] = xr * wr - xi * wi
        work[2 * j + 1] = xr * wi + xi * wr
        j += 1

    radix2(work, twiddles, m, -1, threads)
    j = 0
    while j + C <= m:
        var a = work.load[width=W](2 * j).deinterleave()
        var b = kernel.load[width=W](2 * j).deinterleave()
        work.store(
            2 * j,
            (a[0] * b[0] - a[1] * b[1]).interleave(
                a[0] * b[1] + a[1] * b[0]
            ),
        )
        j += C
    while j < m:
        var ar = work[2 * j]
        var ai = work[2 * j + 1]
        var br = kernel[2 * j]
        var bi = kernel[2 * j + 1]
        work[2 * j] = ar * br - ai * bi
        work[2 * j + 1] = ar * bi + ai * br
        j += 1
    radix2(work, twiddles, m, 1, threads)

    var convolution_scale = scale / Float64(m)
    var k = 0
    if inner == 1:
        while k + C <= n:
            var a = work.load[width=W](2 * k).deinterleave()
            var w = chirp.load[width=W](2 * k).deinterleave()
            dst.store(
                2 * (base + k),
                ((a[0] * w[0] - a[1] * w[1]) * convolution_scale).interleave(
                    (a[0] * w[1] + a[1] * w[0]) * convolution_scale
                ),
            )
            k += C
    while k < n:
        var wr = chirp[2 * k]
        var wi = chirp[2 * k + 1]
        var ar = work[2 * k]
        var ai = work[2 * k + 1]
        var target = base + k * inner
        dst[2 * target] = (ar * wr - ai * wi) * convolution_scale
        dst[2 * target + 1] = (ar * wi + ai * wr) * convolution_scale
        k += 1


def prepare_workspace(scratch: Ptr, n: Int, direction: Int) -> Int:
    if n <= 0:
        return 0
    if is_power_of_two(n):
        prepare_twiddles(scratch + 2 * n, n)
    else:
        var m = next_power_of_two(2 * n - 1)
        var kernel = scratch + 2 * m
        var twiddles = scratch + 4 * m
        var chirp = scratch + 6 * m
        prepare_twiddles(twiddles, m)
        prepare_bluestein(kernel, twiddles, chirp, n, m, direction)
    return 1


def transform_axis(
    src: Ptr,
    dst: Ptr,
    scratch: Ptr,
    total: Int,
    n: Int,
    inner: Int,
    direction: Int,
    scale: Float64,
    threads: Int,
    in_place: Bool,
) -> Int:
    if n <= 0 or inner <= 0 or total % (n * inner) != 0:
        return 0
    var outer = total // (n * inner)
    if is_power_of_two(n):
        var twiddles = scratch + 2 * n
        if (
            threads > 1
            and inner == 1
            and outer >= 2
            and total >= 524288
        ):

            @always_inline
            def transform_vector(
                left: Int,
            ) {
                imm src,
                imm dst,
                imm twiddles,
                imm n,
                imm direction,
                imm scale,
                imm in_place,
            }:
                var base = left * n
                if n > 512 and not in_place:
                    bit_reversed_copy(src, dst, base, base, n)
                    radix2_ordered(
                        dst + 2 * base,
                        twiddles,
                        n,
                        direction,
                        1,
                    )
                else:
                    if not in_place:
                        copy_complex(src, dst, base, base, n)
                    radix2(dst + 2 * base, twiddles, n, direction)
                scale_complex(dst, base, n, scale)

            parallelize(transform_vector, outer, min(threads, outer))
        else:
            if inner > 1:
                copy_complex(src, dst, 0, 0, total)
                radix2_strided(
                    dst,
                    twiddles,
                    n,
                    inner,
                    outer,
                    direction,
                )
                scale_complex(dst, 0, total, scale)
            else:
                for left in range(outer):
                    var base = left * n
                    if n > 512 and not in_place:
                        bit_reversed_copy(src, dst, base, base, n)
                        radix2_ordered(
                            dst + 2 * base,
                            twiddles,
                            n,
                            direction,
                            threads if outer == 1 else 1,
                        )
                    else:
                        if not in_place:
                            copy_complex(src, dst, base, base, n)
                        radix2(
                            dst + 2 * base,
                            twiddles,
                            n,
                            direction,
                            threads,
                        )
                    scale_complex(dst, base, n, scale)
    else:
        var m = next_power_of_two(2 * n - 1)
        var kernel = scratch + 2 * m
        var twiddles = scratch + 4 * m
        var chirp = scratch + 6 * m
        for left in range(outer):
            for right in range(inner):
                var base = left * n * inner + right
                bluestein_vector(
                    src,
                    dst,
                    scratch,
                    kernel,
                    twiddles,
                    chirp,
                    base,
                    n,
                    m,
                    inner,
                    direction,
                    scale,
                    threads,
                )
    return 1
