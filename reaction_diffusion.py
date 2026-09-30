# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo"]
#
# [tool.pixi.workspace]
# channels = ["https://conda.modular.com/max", "conda-forge"]
#
# [tool.pixi.dependencies]
# marimo = ">=0.25.0"
# anywidget = "*"
# mojo = ">=1.1"
# max = ">=26.6"
# numpy = "*"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import sys
    import time

    import anywidget
    import marimo as mo
    import numpy as np
    import traitlets

    import mojo.importer  # noqa: F401  (lets Python import .mojo files)

    sys.path.insert(0, str(mo.notebook_dir()))

    # Classic Gray-Scott parameter sets (feed rate F, kill rate k).
    PRESETS = {
        "🪸 Coral": (0.0545, 0.062),
        "🦠 Mitosis": (0.0367, 0.0649),
        "🪱 Worms": (0.029, 0.057),
        "🫆 Fingerprints": (0.037, 0.06),
        "🛹 U-Skate": (0.062, 0.0609),
        "🌀 Spirals": (0.018, 0.051),
        "🌊 Waves": (0.014, 0.045),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 🔥 Reaction-diffusion, live, in Mojo

    Two virtual chemicals, **U** and **V**, spread across a grid and react:
    `U + 2V → 3V`. U is fed in at rate **F**, V is removed at rate **k**.
    From those two numbers you get coral, dividing cells, fingerprints and
    gliders ([Gray-Scott model](https://www.karlsims.com/rd.html)).

    Each frame runs dozens of simulation steps over the whole grid in Mojo,
    either on **all CPU cores with SIMD** or **on the GPU**. The pixels go
    straight to the canvas. **Click and drag on the canvas to add chemical V.**
    """)
    return


@app.cell
def _():
    import rd_cpu  # compiles rd_cpu.mojo on first import

    try:
        import rd_gpu

        gpu_name = rd_gpu.gpu_name()
        gpu_problem = None
    except ImportError as e:
        rd_gpu, gpu_name = None, None
        cause = str(e.__cause__ or e)
        gpu_problem = (
            "the Metal toolchain is missing. Run `xcodebuild -downloadComponent MetalToolchain` and restart the kernel."
            if "Metal Toolchain" in cause
            else cause.splitlines()[-1]
        )

    backends = ["CPU (SIMD + all cores)"] + ([f"GPU ({gpu_name})"] if rd_gpu else [])
    mo.callout(
        mo.md(f"GPU backend ready: **{gpu_name}** 🚀")
        if rd_gpu
        else mo.md(f"Running on the CPU only. The GPU backend didn't compile because {gpu_problem}"),
        kind="success" if rd_gpu else "warn",
    )
    return backends, rd_cpu, rd_gpu


@app.cell(hide_code=True)
def _():
    _ESM = r"""
    const PALETTES = {
      magma:  [[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]],
      ocean:  [[2,6,23],[8,47,73],[14,116,144],[34,211,238],[165,243,252],[255,255,255]],
      neon:   [[5,0,20],[60,0,120],[255,0,170],[255,200,0],[240,255,240]],
      forest: [[10,15,10],[20,60,30],[70,130,50],[190,220,110],[255,250,220]],
      mono:   [[0,0,0],[255,255,255]],
    };
    function makeLut(name) {
      const stops = PALETTES[name] || PALETTES.magma;
      const lut = new Uint8ClampedArray(256 * 4);
      for (let i = 0; i < 256; i++) {
        const t = (i / 255) * (stops.length - 1);
        const j = Math.min(Math.floor(t), stops.length - 2), f = t - j;
        for (let c = 0; c < 3; c++) lut[i * 4 + c] = stops[j][c] + (stops[j + 1][c] - stops[j][c]) * f;
        lut[i * 4 + 3] = 255;
      }
      return lut;
    }

    function render({ model, el }) {
      const w = model.get("width"), h = model.get("height");
      const wrap = document.createElement("div");
      wrap.style.cssText = "position:relative;width:100%;max-width:720px;margin:auto";
      const canvas = document.createElement("canvas");
      canvas.width = w; canvas.height = h;
      canvas.style.cssText = "width:100%;aspect-ratio:1;border-radius:14px;cursor:crosshair;touch-action:none;display:block;box-shadow:0 10px 40px rgba(0,0,0,.35)";
      const hud = document.createElement("div");
      hud.style.cssText = "position:absolute;left:12px;top:10px;padding:4px 10px;border-radius:8px;background:rgba(0,0,0,.55);color:#fff;font:12px ui-monospace,monospace;pointer-events:none;white-space:pre";
      wrap.append(canvas, hud);
      el.append(wrap);

      const g = canvas.getContext("2d");
      const img = g.createImageData(w, h);
      let lut = makeLut(model.get("palette"));
      model.on("change:palette", () => { lut = makeLut(model.get("palette")); });

      let alive = true, waiting = false, frames = 0, fps = 0, last = performance.now();
      const tick = () => {
        if (!alive || waiting) return;
        waiting = true;
        model.send({ type: "tick" });
      };

      model.on("msg:custom", (msg, buffers) => {
        if (msg.type !== "frame") return;
        const b = buffers[0];
        const px = b instanceof DataView ? new Uint8Array(b.buffer, b.byteOffset, b.byteLength) : new Uint8Array(b);
        const d = img.data;
        for (let i = 0, j = 0; i < px.length; i++, j += 4) {
          const k = px[i] * 4;
          d[j] = lut[k]; d[j + 1] = lut[k + 1]; d[j + 2] = lut[k + 2]; d[j + 3] = 255;
        }
        g.putImageData(img, 0, 0);
        frames++;
        const now = performance.now();
        if (now - last > 500) { fps = (frames * 1000) / (now - last); frames = 0; last = now; }
        const s = msg.stats;
        hud.textContent = `${s.backend}  ${w}×${h}\n${fps.toFixed(0)} fps · ${s.steps} steps/frame · ${s.gcells.toFixed(2)} Gcell/s`;
        waiting = false;
        requestAnimationFrame(tick);
      });

      const paint = (e) => {
        if (!(e.buttons & 1)) return;
        const r = canvas.getBoundingClientRect();
        model.send({ type: "paint", x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height, erase: e.shiftKey });
      };
      canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); paint(e); });
      canvas.addEventListener("pointermove", paint);

      tick();
      return () => { alive = false; };
    }
    export default { render };
    """

    class ReactionDiffusion(anywidget.AnyWidget):
        """Holds the simulation state; the browser asks for frames, Mojo computes them."""

        _esm = _ESM
        width = traitlets.Int(512).tag(sync=True)
        height = traitlets.Int(512).tag(sync=True)
        palette = traitlets.Unicode("magma").tag(sync=True)

        def __init__(self, size, rd_cpu, rd_gpu=None):
            super().__init__(width=size, height=size)
            self.cpu = rd_cpu
            # The GPU state lives on the device between frames; `on_gpu` says
            # which copy (NumPy or GpuSim) is current.
            self.gpu = rd_gpu.GpuSim(size, size) if rd_gpu else None
            self.feed, self.kill, self.steps, self.backend = 0.0545, 0.062, 16, "CPU"
            self.out = np.zeros(size * size, np.uint8)
            self.reset()
            self.on_msg(self._on_msg)

        def reset(self, seed=0):
            n = self.width
            rng = np.random.default_rng(seed)
            self.u = np.ones((n, n), np.float32)
            self.v = np.zeros((n, n), np.float32)
            for _ in range(24):  # scatter a few seeds of V
                x, y, r = rng.integers(0, n, 2).tolist() + [int(rng.integers(n // 80 + 2, n // 25 + 4))]
                self.v[max(0, y - r) : y + r, max(0, x - r) : x + r] = 0.5
                self.u[max(0, y - r) : y + r, max(0, x - r) : x + r] = 0.25
            self.u2, self.v2 = np.empty_like(self.u), np.empty_like(self.v)
            self.on_gpu = False

        def _move_state(self, to_gpu):
            if to_gpu and not self.on_gpu:
                self.gpu.upload(self.u.ctypes.data, self.v.ctypes.data)
            elif self.on_gpu and not to_gpu:
                self.gpu.download(self.u.ctypes.data, self.v.ctypes.data)
            self.on_gpu = to_gpu

        def _on_msg(self, _widget, content, _buffers):
            n = self.width
            if content.get("type") == "paint":
                cx, cy, r = int(content["x"] * n), int(content["y"] * n), max(3, n // 60)
                erase = bool(content.get("erase"))
                if self.on_gpu:
                    self.gpu.paint(cx, cy, r, erase)
                    return
                yy, xx = np.ogrid[:n, :n]
                disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
                self.v[disk], self.u[disk] = (0.0, 1.0) if erase else (0.5, 0.25)
                return
            if content.get("type") != "tick":
                return
            use_gpu = self.gpu is not None and self.backend.startswith("GPU")
            self._move_state(use_gpu)
            t = time.perf_counter()
            if use_gpu:
                self.gpu.step(self.steps, self.feed, self.kill)
            else:
                self.cpu.step(
                    self.u.ctypes.data, self.v.ctypes.data, self.u2.ctypes.data, self.v2.ctypes.data,
                    n, n, self.steps, self.feed, self.kill,
                )
            dt = time.perf_counter() - t
            if use_gpu:
                self.gpu.render(self.out.ctypes.data)
            else:
                self.cpu.render(self.v.ctypes.data, self.out.ctypes.data, n * n)
            stats = {"backend": self.backend.split(" (")[0], "steps": self.steps, "gcells": n * n * self.steps / max(dt, 1e-9) / 1e9}
            self.send({"type": "frame", "stats": stats}, [self.out.tobytes()])

    return (ReactionDiffusion,)


@app.cell
def _(backends):
    preset = mo.ui.dropdown(list(PRESETS), value="🪸 Coral", label="pattern")
    size = mo.ui.dropdown(["256", "512", "1024", "2048"], value="512", label="grid")
    backend = mo.ui.radio(backends, value=backends[-1], label="backend", inline=True)
    palette = mo.ui.dropdown(["magma", "ocean", "neon", "forest", "mono"], value="magma", label="colours")
    steps = mo.ui.slider(2, 96, step=2, value=24, label="steps / frame", show_value=True)
    reset = mo.ui.button(label="↺ reseed", value=0, on_click=lambda v: v + 1)
    return backend, palette, preset, reset, size, steps


@app.cell
def _(preset):
    # Re-created whenever the preset changes, so the sliders jump to its values.
    _f, _k = PRESETS[preset.value]
    feed = mo.ui.slider(0.005, 0.1, step=0.0005, value=_f, label="feed F", show_value=True)
    kill = mo.ui.slider(0.03, 0.075, step=0.0005, value=_k, label="kill k", show_value=True)
    return feed, kill


@app.cell
def _(ReactionDiffusion, rd_cpu, rd_gpu, size):
    sim = ReactionDiffusion(int(size.value), rd_cpu, rd_gpu)
    return (sim,)


@app.cell
def _(backend, feed, kill, palette, preset, reset, sim, size, steps):
    # Push the controls into the running simulation (without restarting it).
    sim.feed, sim.kill, sim.steps, sim.backend = feed.value, kill.value, steps.value, backend.value
    sim.palette = palette.value
    mo.vstack(
        [
            mo.hstack([preset, size, palette, reset], justify="start", gap=1.5),
            mo.hstack([feed, kill, steps], justify="start", gap=1.5),
            backend,
        ]
    )
    return


@app.cell
def _(reset, sim):
    if reset.value:
        sim.reset(seed=reset.value)
    mo.ui.anywidget(sim)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The kernel

    This is the whole update rule, from [`gray_scott.mojo`](./gray_scott.mojo).
    It's generic over the SIMD width `n`: the CPU backend calls it with `n = 8`,
    so each instruction updates 8 cells, and the GPU backend calls it with
    `n = 1` from one thread per cell. It's the same source in both cases.
    """)
    return


@app.cell(hide_code=True)
def _():
    _src = (mo.notebook_dir() / "gray_scott.mojo").read_text()
    _start = _src.index("@always_inline\ndef update_cell")
    _end = _src.index("@always_inline\ndef laplacian_wrapped")
    mo.md(f"```python\n{_src[_start:_end].strip()}\n```")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How much faster than NumPy?

    Here's the same simulation written in idiomatic NumPy (`np.roll` for the
    neighbours), timed against the Mojo backends on a fresh grid. This runs
    separately from the live animation above.
    """)
    return


@app.cell
def _():
    bench_size = mo.ui.dropdown(["512", "1024", "2048"], value="1024", label="grid")
    bench = mo.ui.run_button(label="🏁 Benchmark")
    mo.hstack([bench_size, bench], justify="start")
    return bench, bench_size


@app.cell
def _(bench, bench_size, rd_cpu, rd_gpu):
    mo.stop(not bench.value, mo.md("_Press **Benchmark** to race NumPy against Mojo._"))

    def numpy_step(u, v, F, k):
        def lap(a):
            up, down = np.roll(a, 1, 0), np.roll(a, -1, 0)
            edges = up + down + np.roll(a, 1, 1) + np.roll(a, -1, 1)
            corners = np.roll(up, 1, 1) + np.roll(up, -1, 1) + np.roll(down, 1, 1) + np.roll(down, -1, 1)
            return 0.2 * edges + 0.05 * corners - a

        uvv = u * v * v
        return u + lap(u) - uvv + F * (1 - u), v + 0.5 * lap(v) + uvv - (F + k) * v

    _n = int(bench_size.value)
    _F, _k = PRESETS["🪸 Coral"]

    def _fresh():
        u, v = np.ones((_n, _n), np.float32), np.zeros((_n, _n), np.float32)
        v[_n // 3 : _n // 2, _n // 3 : _n // 2] = 0.5
        return u, v

    def time_mojo(step_fn, steps=200):
        u, v = _fresh()
        u2, v2 = np.empty_like(u), np.empty_like(v)
        step_fn(u.ctypes.data, v.ctypes.data, u2.ctypes.data, v2.ctypes.data, _n, _n, 2, _F, _k)  # warm up
        t = time.perf_counter()
        step_fn(u.ctypes.data, v.ctypes.data, u2.ctypes.data, v2.ctypes.data, _n, _n, steps, _F, _k)
        return (time.perf_counter() - t) / steps

    def time_gpu(steps=1000):
        gpu = rd_gpu.GpuSim(_n, _n)
        u, v = _fresh()
        gpu.upload(u.ctypes.data, v.ctypes.data)
        gpu.step(2, _F, _k)  # warm up
        t = time.perf_counter()
        gpu.step(steps, _F, _k)  # synchronizes before returning
        return (time.perf_counter() - t) / steps

    _u, _v = _fresh()
    _t = time.perf_counter()
    for _ in range(10):
        _u, _v = numpy_step(_u, _v, np.float32(_F), np.float32(_k))
    _rows = {"NumPy": (time.perf_counter() - _t) / 10, "Mojo CPU": time_mojo(rd_cpu.step)}
    if rd_gpu is not None:
        _rows[f"Mojo GPU ({rd_gpu.gpu_name()})"] = time_gpu()

    _base = _rows["NumPy"]
    mo.ui.table(
        [
            {
                "backend": name,
                "ms / step": round(t * 1e3, 3),
                "cells / s": f"{_n * _n / t / 1e9:.2f} G",
                "vs NumPy": f"{_base / t:.0f}x",
            }
            for name, t in _rows.items()
        ],
        selection=None,
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
