# Gray-Scott reaction-diffusion on the CPU: one thread per row (all cores),
# SIMD across each row. The state lives in NumPy float32 arrays owned by
# Python; we get their addresses and work on the memory in place.

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from max.algorithm import parallelize

from gray_scott import *


@export
def PyInit_rd_cpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("rd_cpu")
        m.def_function[step_cpu](
            "step", docstring="step(u, v, u2, v2, w, h, steps, feed, kill)"
        )
        m.def_function[render](
            "render", docstring="render(v, out, n): map v to 0..255 bytes"
        )
        return m.finalize()
    except e:
        abort(String("failed to create rd_cpu module: ", e))


def cpu_step_once(
    u: FloatPtr,
    v: FloatPtr,
    u2: FloatPtr,
    v2: FloatPtr,
    w: Int,
    h: Int,
    feed: Float32,
    kill: Float32,
):
    def row(y: Int) {u, v, u2, v2, w, h, feed, kill}:
        var up = ((y - 1 + h) % h) * w
        var mid = y * w
        var down = ((y + 1) % h) * w

        # Wrap-around columns, one cell at a time.
        for x in [0, w - 1]:
            var cu = u[unsafe_offset = mid + x]
            var cv = v[unsafe_offset = mid + x]
            var r = update_cell[1](
                cu,
                cv,
                laplacian_wrapped(u, up, mid, down, x, w),
                laplacian_wrapped(v, up, mid, down, x, w),
                feed,
                kill,
            )
            u2[unsafe_offset = mid + x] = r[0][0]
            v2[unsafe_offset = mid + x] = r[1][0]

        # Interior: WIDTH cells per instruction.
        var x = 1
        while x + WIDTH <= w - 1:
            var r = update_cell[WIDTH](
                u.unsafe_load[width=WIDTH](mid + x),
                v.unsafe_load[width=WIDTH](mid + x),
                laplacian[WIDTH](u, up, mid, down, x),
                laplacian[WIDTH](v, up, mid, down, x),
                feed,
                kill,
            )
            u2.unsafe_store(mid + x, r[0])
            v2.unsafe_store(mid + x, r[1])
            x += WIDTH
        while x < w - 1:
            var r = update_cell[1](
                u[unsafe_offset = mid + x],
                v[unsafe_offset = mid + x],
                laplacian[1](u, up, mid, down, x),
                laplacian[1](v, up, mid, down, x),
                feed,
                kill,
            )
            u2[unsafe_offset = mid + x] = r[0][0]
            v2[unsafe_offset = mid + x] = r[1][0]
            x += 1

    parallelize(row, h)


def step_cpu(
    u_addr: PythonObject,
    v_addr: PythonObject,
    u2_addr: PythonObject,
    v2_addr: PythonObject,
    w_obj: PythonObject,
    h_obj: PythonObject,
    steps_obj: PythonObject,
    feed_obj: PythonObject,
    kill_obj: PythonObject,
) raises -> PythonObject:
    """Advance `steps` (rounded up to even) steps; result ends up in u, v."""
    var u = ptr_from(u_addr)
    var v = ptr_from(v_addr)
    var u2 = ptr_from(u2_addr)
    var v2 = ptr_from(v2_addr)
    var w = Int(py=w_obj)
    var h = Int(py=h_obj)
    var steps = Int(py=steps_obj)
    var feed = Float32(Float64(py=feed_obj))
    var kill = Float32(Float64(py=kill_obj))
    for _ in range((steps + 1) // 2):
        cpu_step_once(u, v, u2, v2, w, h, feed, kill)
        cpu_step_once(u2, v2, u, v, w, h, feed, kill)
    return Python.none()


# ---------------------------------------------------------------------------
# Rendering: v in [0, ~0.4] -> one byte per pixel (the browser applies colour)
# ---------------------------------------------------------------------------


def render(
    v_addr: PythonObject, out_addr: PythonObject, n_obj: PythonObject
) raises -> PythonObject:
    var v = ptr_from(v_addr)
    var out = BytePtr(unsafe_from_address=Int(py=out_addr))
    var n = Int(py=n_obj)
    var i = 0
    while i + WIDTH <= n:
        var x = (v.unsafe_load[width=WIDTH](i) * 3.0).clamp(0.0, 1.0) * 255.0
        out.unsafe_store(i, x.cast[DType.uint8]())
        i += WIDTH
    while i < n:
        out[unsafe_offset=i] = UInt8(min(max(v[unsafe_offset=i] * 3.0, 0.0), 1.0) * 255.0)
        i += 1
    return Python.none()
