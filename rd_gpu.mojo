# Gray-Scott reaction-diffusion on the GPU: one thread per cell.
#
# `GpuSim` is a Mojo type exposed to Python. It owns the device buffers, so the
# simulation state stays on the GPU between frames. Per frame only the rendered
# image (one byte per cell) comes back to the host.
#
# This lives in its own module so the CPU version still works on machines where
# the GPU toolchain (e.g. Xcode's Metal toolchain on macOS) is missing.

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from gray_scott import *

comptime BLOCK = 16


@export
def PyInit_rd_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("rd_gpu")
        _ = (
            m.add_type[GpuSim]("GpuSim")
            .def_py_init[GpuSim.py_init]()
            .def_method[GpuSim.upload]("upload")
            .def_method[GpuSim.download]("download")
            .def_method[GpuSim.step]("step")
            .def_method[GpuSim.render]("render")
            .def_method[GpuSim.paint]("paint")
        )
        m.def_function[gpu_name]("gpu_name", docstring="Name of the GPU")
        return m.finalize()
    except e:
        abort(String("failed to create rd_gpu module: ", e))


def gpu_name() raises -> PythonObject:
    return PythonObject(DeviceContext().name())


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def step_kernel(
    u: FloatPtr,
    v: FloatPtr,
    u2: FloatPtr,
    v2: FloatPtr,
    w32: Int32,
    h32: Int32,
    feed: Float32,
    kill: Float32,
):
    var w = Int(w32)
    var h = Int(h32)
    var x = Int(global_idx.x)
    var y = Int(global_idx.y)
    if x >= w or y >= h:
        return
    var rows = wrap_rows(y, w, h)
    var up = rows[0]
    var mid = rows[1]
    var down = rows[2]
    var r = update_cell[1](
        u[unsafe_offset = mid + x],
        v[unsafe_offset = mid + x],
        laplacian_wrapped(u, up, mid, down, x, w),
        laplacian_wrapped(v, up, mid, down, x, w),
        feed,
        kill,
    )
    u2[unsafe_offset = mid + x] = r[0][0]
    v2[unsafe_offset = mid + x] = r[1][0]


def render_kernel(v: FloatPtr, img: BytePtr, n32: Int32):
    var i = Int(global_idx.x)
    if i < Int(n32):
        var x = min(max(v[unsafe_offset=i] * 3.0, 0.0), 1.0) * 255.0
        img[unsafe_offset=i] = UInt8(x)


def paint_kernel(
    u: FloatPtr,
    v: FloatPtr,
    w32: Int32,
    h32: Int32,
    cx: Int32,
    cy: Int32,
    r: Int32,
    erase: Int32,
):
    var x = Int32(global_idx.x)
    var y = Int32(global_idx.y)
    if x >= w32 or y >= h32:
        return
    if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r:
        var i = Int(y * w32 + x)
        u[unsafe_offset=i] = 1.0 if erase else 0.25
        v[unsafe_offset=i] = 0.0 if erase else 0.5


# ---------------------------------------------------------------------------
# The Python-visible type
# ---------------------------------------------------------------------------


struct GpuSim(Movable, Writable):
    var ctx: DeviceContext
    var u: DeviceBuffer[DType.float32]
    var v: DeviceBuffer[DType.float32]
    var u2: DeviceBuffer[DType.float32]
    var v2: DeviceBuffer[DType.float32]
    var img: DeviceBuffer[DType.uint8]
    var w: Int
    var h: Int

    def __init__(out self, w: Int, h: Int) raises:
        self.ctx = DeviceContext()
        self.w = w
        self.h = h
        self.u = self.ctx.enqueue_create_buffer[DType.float32](w * h)
        self.v = self.ctx.enqueue_create_buffer[DType.float32](w * h)
        self.u2 = self.ctx.enqueue_create_buffer[DType.float32](w * h)
        self.v2 = self.ctx.enqueue_create_buffer[DType.float32](w * h)
        self.img = self.ctx.enqueue_create_buffer[DType.uint8](w * h)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GpuSim(", self.w, "x", self.h, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(
        out self: GpuSim, args: PythonObject, kwargs: PythonObject
    ) raises:
        self = GpuSim(Int(py=args[0]), Int(py=args[1]))

    @staticmethod
    def _get(py_self: PythonObject) raises -> Pointer[GpuSim, MutAnyOrigin]:
        return py_self.downcast_value_ptr[GpuSim]()

    @staticmethod
    def upload(
        py_self: PythonObject, u_addr: PythonObject, v_addr: PythonObject
    ) raises -> PythonObject:
        """Copy host (NumPy) state onto the GPU."""
        var s = GpuSim._get(py_self)
        s[].ctx.enqueue_copy(s[].u, ptr_from(u_addr))
        s[].ctx.enqueue_copy(s[].v, ptr_from(v_addr))
        s[].ctx.synchronize()
        return Python.none()

    @staticmethod
    def download(
        py_self: PythonObject, u_addr: PythonObject, v_addr: PythonObject
    ) raises -> PythonObject:
        """Copy the GPU state back into host (NumPy) arrays."""
        var s = GpuSim._get(py_self)
        s[].ctx.enqueue_copy(ptr_from(u_addr), s[].u)
        s[].ctx.enqueue_copy(ptr_from(v_addr), s[].v)
        s[].ctx.synchronize()
        return Python.none()

    @staticmethod
    def step(
        py_self: PythonObject,
        steps_obj: PythonObject,
        feed_obj: PythonObject,
        kill_obj: PythonObject,
    ) raises -> PythonObject:
        """Advance `steps` (rounded up to even) steps on the GPU."""
        var s = GpuSim._get(py_self)
        var feed = Float32(Float64(py=feed_obj))
        var kill = Float32(Float64(py=kill_obj))
        var w32 = Int32(s[].w)
        var h32 = Int32(s[].h)
        var grid = ((s[].w + BLOCK - 1) // BLOCK, (s[].h + BLOCK - 1) // BLOCK)
        for _ in range((Int(py=steps_obj) + 1) // 2):
            s[].ctx.enqueue_function[step_kernel](
                s[].u.unsafe_ptr(),
                s[].v.unsafe_ptr(),
                s[].u2.unsafe_ptr(),
                s[].v2.unsafe_ptr(),
                w32,
                h32,
                feed,
                kill,
                grid_dim=grid,
                block_dim=(BLOCK, BLOCK),
            )
            s[].ctx.enqueue_function[step_kernel](
                s[].u2.unsafe_ptr(),
                s[].v2.unsafe_ptr(),
                s[].u.unsafe_ptr(),
                s[].v.unsafe_ptr(),
                w32,
                h32,
                feed,
                kill,
                grid_dim=grid,
                block_dim=(BLOCK, BLOCK),
            )
        s[].ctx.synchronize()
        return Python.none()

    @staticmethod
    def render(py_self: PythonObject, out_addr: PythonObject) raises -> PythonObject:
        """Render v to bytes on the GPU and copy just those bytes back."""
        var s = GpuSim._get(py_self)
        var n = s[].w * s[].h
        s[].ctx.enqueue_function[render_kernel](
            s[].v.unsafe_ptr(),
            s[].img.unsafe_ptr(),
            Int32(n),
            grid_dim=(n + 255) // 256,
            block_dim=256,
        )
        s[].ctx.enqueue_copy(BytePtr(unsafe_from_address=Int(py=out_addr)), s[].img)
        s[].ctx.synchronize()
        return Python.none()

    @staticmethod
    def paint(
        py_self: PythonObject,
        cx: PythonObject,
        cy: PythonObject,
        r: PythonObject,
        erase: PythonObject,
    ) raises -> PythonObject:
        """Drop (or erase) a disk of chemical V at (cx, cy)."""
        var s = GpuSim._get(py_self)
        s[].ctx.enqueue_function[paint_kernel](
            s[].u.unsafe_ptr(),
            s[].v.unsafe_ptr(),
            Int32(s[].w),
            Int32(s[].h),
            Int32(Int(py=cx)),
            Int32(Int(py=cy)),
            Int32(Int(py=r)),
            Int32(1 if Bool(erase) else 0),
            grid_dim=((s[].w + BLOCK - 1) // BLOCK, (s[].h + BLOCK - 1) // BLOCK),
            block_dim=(BLOCK, BLOCK),
        )
        s[].ctx.synchronize()
        return Python.none()
