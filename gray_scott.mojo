# Shared Gray-Scott math, used by both rd_cpu.mojo and rd_gpu.mojo.
#
# Model (Karl Sims' formulation, dt = 1):
#     u' = u + Du * lap(u) - u*v*v + F * (1 - u)
#     v' = v + Dv * lap(v) + u*v*v - (F + k) * v
# with a 9-point Laplacian and wrap-around edges.

from std.python import PythonObject

comptime FloatPtr = Pointer[Float32, MutAnyOrigin]
comptime BytePtr = Pointer[UInt8, MutAnyOrigin]

comptime DU: Float32 = 1.0
comptime DV: Float32 = 0.5
comptime EDGE: Float32 = 0.2  # weight of the 4 direct neighbours
comptime CORNER: Float32 = 0.05  # weight of the 4 diagonal neighbours
comptime WIDTH = 8  # SIMD lanes for the CPU backend


def ptr_from(obj: PythonObject) raises -> FloatPtr:
    return FloatPtr(unsafe_from_address=Int(py=obj))


@always_inline
def update_cell[
    n: Int
](
    u: SIMD[DType.float32, n],
    v: SIMD[DType.float32, n],
    lap_u: SIMD[DType.float32, n],
    lap_v: SIMD[DType.float32, n],
    feed: Float32,
    kill: Float32,
) -> Tuple[SIMD[DType.float32, n], SIMD[DType.float32, n]]:
    var uvv = u * v * v
    var new_u = u + DU * lap_u - uvv + feed * (1.0 - u)
    var new_v = v + DV * lap_v + uvv - (feed + kill) * v
    return (new_u, new_v)


@always_inline
def laplacian[
    n: Int
](p: FloatPtr, up: Int, mid: Int, down: Int, x: Int) -> SIMD[DType.float32, n]:
    """9-point Laplacian of n consecutive cells starting at column x."""
    var c = p.unsafe_load[width=n](mid + x)
    var edges = (
        p.unsafe_load[width=n](up + x)
        + p.unsafe_load[width=n](down + x)
        + p.unsafe_load[width=n](mid + x - 1)
        + p.unsafe_load[width=n](mid + x + 1)
    )
    var corners = (
        p.unsafe_load[width=n](up + x - 1)
        + p.unsafe_load[width=n](up + x + 1)
        + p.unsafe_load[width=n](down + x - 1)
        + p.unsafe_load[width=n](down + x + 1)
    )
    return EDGE * edges + CORNER * corners - c


@always_inline
def laplacian_wrapped(
    p: FloatPtr, up: Int, mid: Int, down: Int, x: Int, w: Int
) -> Float32:
    """Scalar Laplacian at column x, wrapping around the left/right edges.

    Uses comparisons instead of `%`: integer modulo is slow, especially on GPUs.
    """
    var xl = x - 1 if x > 0 else w - 1
    var xr = x + 1 if x < w - 1 else 0
    var edges = p[unsafe_offset = up + x] + p[unsafe_offset = down + x] + p[
        unsafe_offset = mid + xl
    ] + p[unsafe_offset = mid + xr]
    var corners = p[unsafe_offset = up + xl] + p[unsafe_offset = up + xr] + p[
        unsafe_offset = down + xl
    ] + p[unsafe_offset = down + xr]
    return EDGE * edges + CORNER * corners - p[unsafe_offset = mid + x]


@always_inline
def wrap_rows(y: Int, w: Int, h: Int) -> Tuple[Int, Int, Int]:
    """Offsets of the rows above, at and below y, wrapping top/bottom."""
    var up = (y - 1 if y > 0 else h - 1) * w
    var down = (y + 1 if y < h - 1 else 0) * w
    return (up, y * w, down)
