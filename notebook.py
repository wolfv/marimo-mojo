# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.25.0", "pytest"]
#
# [tool.pixi.workspace]
# channels = ["https://conda.modular.com/max", "conda-forge"]
#
# [tool.pixi.dependencies]
# mojo = ">=1.1"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import json
    import random
    import sysconfig

    import pytest

    # Workaround: marimo's reactive test runner iterates sys.modules while
    # lazily importing sysconfig's data modules, which raises "dictionary
    # changed size during iteration". Loading them up front avoids that.
    sysconfig.get_paths()

    # Test data lives in the setup block so pytest.mark.parametrize can see it.
    VALID = [
        "[]",
        "{}",
        '"x"',
        "0",
        "-0",
        "0.1",
        "-12.5e-3",
        "1E10",
        "12345678901234567890123",
        "[[[[]]]]",
        '"héllo 🚀"',
        '"\\u00e9\\ud83d\\ude80\\n\\t\\"\\\\\\/"',
        ' [1 , {"k" :[ ]} , true, false, null] ',
    ]
    INVALID = ["{", "[1,]", '{"a" 1}', "01", '"abc', "nul", "[1] x", '"\\x"', "", "-", "1.", '{"a":1,}', '"\\ud800"']


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥 A JSON parser in Mojo, poked at from marimo

    This notebook runs in a **pixi sandbox**. The header at the top of the file
    pulls `mojo` from Modular's conda channel and `marimo` from PyPI, so you get
    the Mojo compiler without installing anything globally:

    ```bash
    pixi exec marimo edit --sandbox=pixi notebook.py
    ```

    Next to this notebook is [`json_parser.mojo`](./json_parser.mojo): a small
    hand-written recursive-descent JSON parser. It compiles to a native Python
    extension module. `import mojo.importer` teaches Python to `import` `.mojo`
    files directly. The first import compiles the file, and later imports use the
    cached build.
    """)
    return


@app.cell
def _(mo):
    import sys
    import time

    import mojo.importer  # noqa: F401  (registers the .mojo import hook)

    sys.path.insert(0, str(mo.notebook_dir()))

    _t0 = time.perf_counter()
    import json_parser

    compile_seconds = time.perf_counter() - _t0
    mo.md(
        f"Imported `json_parser` in **{compile_seconds:.2f}s** "
        "(slow on the first run while Mojo compiles; fast once it's cached)."
    )
    return json_parser, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Playground

    Type some JSON. It gets tokenized (for the syntax highlighting), validated
    and parsed by Mojo as you type. Try breaking it: the error messages include
    line and column.
    """)
    return


@app.cell
def _(mo):
    json_input = mo.ui.text_area(
        value="""{
      "language": "Mojo",
      "fast": true,
      "version": 1.1,
      "features": ["SIMD", "traits", "ownership", "Python interop"],
      "escapes": "tab\\tnewline\\nrocket \\ud83d\\ude80",
      "nothing": null,
      "big": 123456789012345678901234567890,
      "nested": {"deeper": {"deepest": [[1, 2], [3, [4, [5]]]]}}
    }""",
        rows=14,
        full_width=True,
        label="JSON input",
    )
    json_input
    return (json_input,)


@app.cell(hide_code=True)
def _(json_input, json_parser, mo):
    import html as _html

    _COLORS = {
        "key": "#c678dd",
        "string": "#98c379",
        "number": "#d19a66",
        "bool": "#56b6c2",
        "null": "#e06c75",
        "punct": "#abb2bf",
    }

    def highlight(text: str) -> mo.Html:
        """Color the input using the token stream coming back from Mojo."""
        raw = text.encode()  # Mojo reports byte offsets
        out, last = [], 0
        for kind, start, end in json_parser.tokenize(text):
            out.append(_html.escape(raw[last:start].decode()))
            tok = _html.escape(raw[start:end].decode())
            out.append(f'<span style="color:{_COLORS[kind]}">{tok}</span>')
            last = end
        out.append(_html.escape(raw[last:].decode()))
        return mo.Html(
            '<pre style="background:#282c34;color:#abb2bf;padding:12px;'
            'border-radius:8px;overflow-x:auto;margin:0">' + "".join(out) + "</pre>"
        )

    try:
        _parsed = json_parser.parse(json_input.value)
        _stats = json_parser.validate(json_input.value)
        _view = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(value=str(_stats[k]), label=k.replace("_", " "))
                        for k in ["objects", "arrays", "strings", "numbers", "max_depth"]
                    ],
                    justify="start",
                ),
                mo.hstack(
                    [
                        mo.vstack([mo.md("**Tokens (from Mojo)**"), highlight(json_input.value)]),
                        mo.vstack([mo.md("**Parsed Python object**"), mo.tree(_parsed)]),
                    ],
                    widths=[1, 1],
                    align="start",
                ),
            ]
        )
    except Exception as e:
        _view = mo.callout(mo.md(f"**Parse error:** `{e}`"), kind="danger")
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Does it agree with Python's `json`?

    The cell below holds a few plain `pytest` tests. marimo runs cells that only
    contain `test_*` functions through pytest and shows the results inline. You
    can also run them from the terminal with
    `pixi run --script notebook.py` or `pytest notebook.py`.
    """)
    return


@app.function
def random_value(rng, depth=0):
    """A random JSON-able Python value, used for fuzzing and benchmarking."""
    r = rng.random()
    if depth > 4 or r < 0.4:
        return rng.choice(
            [rng.randint(-(10**9), 10**9), rng.uniform(-1e6, 1e6), "ü" * rng.randint(0, 5), True, False, None]
        )
    if r < 0.7:
        return [random_value(rng, depth + 1) for _ in range(rng.randint(0, 6))]
    return {f"k{i}": random_value(rng, depth + 1) for i in range(rng.randint(0, 6))}


@app.cell
def _(json_parser):
    @pytest.mark.parametrize("doc", VALID)
    def test_matches_stdlib(doc):
        ours, theirs = json_parser.parse(doc), json.loads(doc)
        assert ours == theirs and type(ours) is type(theirs)

    @pytest.mark.parametrize("doc", INVALID)
    def test_rejects_invalid(doc):
        with pytest.raises(Exception):
            json_parser.parse(doc)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_roundtrip(seed):
        doc = json.dumps(random_value(random.Random(seed)))
        assert json_parser.parse(doc) == json.loads(doc)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Race it against `json.loads`

    `json.loads` is CPython's C implementation, so it's a tough opponent. We time
    two Mojo entry points:

    * **`parse`** builds real Python `dict`/`list`/`str` objects, the same as
      `json.loads`. Most of its time goes into allocating Python objects through
      the CPython API.
    * **`validate`** walks the whole document and counts things, but creates no
      Python objects. This shows how fast the Mojo code itself is.
    """)
    return


@app.cell
def _(mo):
    n_items = mo.ui.slider(10_000, 200_000, step=10_000, value=50_000, label="records", show_value=True)
    race = mo.ui.run_button(label="🏁 Race!")
    mo.hstack([n_items, race], justify="start")
    return n_items, race


@app.cell
def _(json_parser, mo, n_items, race, time):
    mo.stop(not race.value, mo.md("_Press **Race!** to generate a document and time the parsers._"))

    _rng = random.Random(42)

    def make_record(i):
        """Something that looks like a typical API response row."""
        return {
            "id": i,
            "name": f"user_{i}",
            "email": f"user_{i}@example.com",
            "active": _rng.random() > 0.3,
            "score": round(_rng.uniform(0, 100), 3),
            "tags": _rng.sample(["mojo", "python", "marimo", "pixi", "simd", "gpu"], k=_rng.randint(0, 4)),
            "address": {"city": _rng.choice(["Berlin", "Paris", "Zürich", "東京"]), "zip": _rng.randint(10000, 99999)},
            "manager": None,
        }

    doc = json.dumps([make_record(i) for i in range(n_items.value)], ensure_ascii=False)

    def best_of(fn, repeat=3):
        times = []
        for _ in range(repeat):
            t = time.perf_counter()
            fn(doc)
            times.append(time.perf_counter() - t)
        return min(times)

    _results = {
        "json.loads (C)": best_of(json.loads),
        "mojo parse": best_of(json_parser.parse),
        "mojo validate": best_of(json_parser.validate),
    }
    _baseline = _results["json.loads (C)"]
    _mb = len(doc.encode()) / 1e6
    mo.vstack(
        [
            mo.md(f"Document size: **{_mb:.1f} MB**"),
            mo.ui.table(
                [
                    {
                        "parser": name,
                        "time (ms)": round(t * 1000, 1),
                        "throughput (MB/s)": round(_mb / t),
                        "vs json.loads": f"{_baseline / t:.2f}x",
                    }
                    for name, t in _results.items()
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write Mojo right here

    The sandbox includes the full Mojo toolchain, so you can also write Mojo
    directly in the notebook. Edit the code below and hit **Run Mojo**. It gets
    compiled and executed with `mojo run`, and whatever it prints shows up
    underneath.
    """)
    return


@app.cell
def _(mo):
    mojo_code = mo.ui.code_editor(
        value='''def fib(n: Int) -> Int:
        var a = 0
        var b = 1
        for _ in range(n):
            a, b = b, a + b
        return a


    def main():
        # SIMD vectors are first-class citizens
        var v = SIMD[DType.float32, 8](1, 2, 3, 4, 5, 6, 7, 8)
        print("v * 2      =", v * 2)
        print("sum(v)     =", v.reduce_add())

        for n in [10, 50, 90]:
            print("fib(", n, ") =", fib(n))
    ''',
        language="python",
        min_height=280,
    )
    run_mojo = mo.ui.run_button(label="▶ Run Mojo")
    mo.vstack([mojo_code, run_mojo])
    return mojo_code, run_mojo


@app.cell
def _(mo, mojo_code, run_mojo, time):
    import tempfile
    from pathlib import Path

    from mojo.run import subprocess_run_mojo

    mo.stop(not run_mojo.value, mo.md("_Press **Run Mojo** to compile and run the code above._"))

    with tempfile.TemporaryDirectory() as _tmp:
        _src = Path(_tmp) / "playground.mojo"
        _src.write_text(mojo_code.value)
        _t = time.perf_counter()
        _proc = subprocess_run_mojo(["run", str(_src)], capture_output=True, text=True)
        _elapsed = time.perf_counter() - _t

    _output = (_proc.stdout + _proc.stderr).replace(str(_src), "playground.mojo")
    mo.vstack(
        [
            mo.md(f"{'✅' if _proc.returncode == 0 else '❌'} exit code {_proc.returncode}, compiled + ran in {_elapsed:.2f}s"),
            mo.plain_text(_output or "(no output)"),
        ]
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
