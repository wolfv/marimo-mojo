#!/usr/bin/env -S pixi run --experimental --script
# /// conda-script
# channels = ["https://conda.modular.com/max", "conda-forge"]
# entrypoint = "mojo run ${SCRIPT}"
#
# [dependencies]
# mojo = ">=1.1"
# /// end-conda-script

# ASCII Mandelbrot set. Run with:
#   pixi run --experimental --script mandelbrot.mojo
# or make it executable and run ./mandelbrot.mojo

comptime WIDTH = 78
comptime HEIGHT = 32
comptime MAX_ITER = 200
comptime PALETTE = " .:-=+*#%@"


def escape_time(cr: Float64, ci: Float64) -> Int:
    var zr = 0.0
    var zi = 0.0
    for i in range(MAX_ITER):
        var zr2 = zr * zr
        var zi2 = zi * zi
        if zr2 + zi2 > 4.0:
            return i
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr
    return MAX_ITER


def main():
    var palette = String(PALETTE)
    var n = palette.byte_length()
    for y in range(HEIGHT):
        var line = String()
        var ci = -1.2 + 2.4 * Float64(y) / Float64(HEIGHT - 1)
        for x in range(WIDTH):
            var cr = -2.2 + 3.0 * Float64(x) / Float64(WIDTH - 1)
            var it = escape_time(cr, ci)
            var idx = 0 if it == MAX_ITER else 1 + min(it // 2, n - 2)
            line += palette[byte=idx]
        print(line)
