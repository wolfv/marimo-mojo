# Mojo 🔥 JSON parser + marimo + pixi sandbox

- `json_parser.mojo`: a small recursive-descent JSON parser in Mojo, exposed to
  Python as an extension module (`parse`, `validate`, `tokenize`).
- `notebook.py`: a marimo notebook with a live JSON playground, pytest tests
  against `json.loads`, a benchmark, and a cell where you can write and run Mojo.

The notebook's inline script header pulls `mojo` from Modular's conda channel,
so all you need is pixi:

```bash
pixi exec marimo edit --sandbox=pixi notebook.py   # interactive
pixi run --script notebook.py                      # run as a script
```

## Live reaction-diffusion (CPU SIMD + GPU)

- `reaction_diffusion.py`: marimo notebook with a live Gray-Scott simulation
  you can paint on, presets, a CPU/GPU switch, and a NumPy benchmark.
- `gray_scott.mojo`: the update rule, generic over SIMD width (shared).
- `rd_cpu.mojo`: all cores via `parallelize`, 8-wide SIMD per row.
- `rd_gpu.mojo`: one GPU thread per cell (`max.gpu`).

```bash
pixi exec marimo edit --sandbox=pixi reaction_diffusion.py
```

On macOS the GPU backend needs Xcode's Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`); without it the notebook
falls back to the CPU backend.

## Standalone Mojo script (`conda-script` header)

`mandelbrot.mojo` declares its own environment in a `/// conda-script`
comment block (channels, dependencies, and the command that runs it), so
it needs no `pixi.toml`:

```bash
pixi run --experimental --script mandelbrot.mojo
./mandelbrot.mojo   # same thing, via the shebang
```
