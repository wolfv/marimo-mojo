# Mojo + marimo demos

Interactive marimo notebooks with Mojo backends, plus a standalone Mandelbrot
script. Each demo declares its dependencies inline, so you only need
[pixi](https://pixi.sh) to get started. Run the commands below from this directory.

<img width="1589" height="1175" alt="Screenshot 2026-09-30 at 08 29 16" src="https://github.com/user-attachments/assets/a017b321-6c9d-4948-a5d4-ed0ebab3c1d1" />

## Reaction-diffusion (CPU and GPU)

- `reaction_diffusion.py`: marimo notebook with a live Gray-Scott simulation
  you can paint on, presets, a CPU/GPU switch, and a NumPy benchmark.
- `gray_scott.mojo`: the shared update rule, generic over SIMD width.
- `rd_cpu.mojo`: parallel row updates with 8-wide SIMD.
- `rd_gpu.mojo`: one GPU thread per cell using `max.gpu`.

```bash
pixi exec marimo edit --sandbox=pixi reaction_diffusion.py
```

On macOS the GPU backend needs Xcode's Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`); without it the notebook
falls back to the CPU backend.

## Standalone Mandelbrot script

`mandelbrot.mojo` prints an ASCII Mandelbrot set. Its `conda-script` header
declares the channels, dependencies, and entrypoint, so it needs no `pixi.toml`.
This demo uses pixi's experimental script support:

```bash
pixi run --experimental --script mandelbrot.mojo
./mandelbrot.mojo   # same thing, via the shebang
```

## JSON parser

- `json_parser.mojo`: a small recursive-descent JSON parser in Mojo, exposed to
  Python as an extension module (`parse`, `validate`, `tokenize`).
- `notebook.py`: a marimo notebook with a live JSON playground, pytest tests
  against `json.loads`, a benchmark, and a cell where you can write and run Mojo.

The notebook's script header installs Mojo from Modular's conda channel.
Python imports `json_parser.mojo` through `mojo.importer`, which compiles it on
first import and caches the build.

```bash
pixi exec marimo edit --sandbox=pixi notebook.py   # interactive
pixi run --script notebook.py                      # run as a script
```

