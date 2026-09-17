# nam_render (native)

Prefer not to build this yourself? `scripts/download_nam_render.sh` (macOS/
Linux) / `scripts/download_nam_render.ps1` (Windows) fetch a prebuilt binary
from this repo's GitHub Releases instead — see the README's "Quick start".
CI builds and attaches these binaries automatically on every
`nam-render-v*` tag (`.github/workflows/build-nam-render.yml`).

A small C++ CLI wrapper that gives `hybrid/render.py` real NAM inference,
without needing torch or the Python `neural-amp-modeler` package installed.
It links directly against [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
(the same inference core used by the official NAM plugin) and builds
the repository's small `nam_render.cpp` wrapper (`nam::get_dsp` loads the
`.nam` file; `dsp->process()` runs it block-by-block over a WAV). The wrapper
adds a test-only `--block-size` option; normal application rendering keeps
the established 64-frame block size.

`CMakeLists.txt` here fetches NAMCore v0.5.4's *source* only (not its own
`tools/CMakeLists.txt`, which sets an MSVC-invalid `-Wno-error` flag) and
compiles the `render.cpp` tool plus the NAM sources and AudioDSPTools' WAV
I/O directly, as target `nam_render`.

## Build

Requires CMake 3.18+ and a C++20 compiler (MSVC, or GCC/Clang on other
platforms). First configure downloads NAMCore + Eigen + AudioDSPTools via
`FetchContent` (a few hundred MB, one-time).

```bash
cmake -B build -S .
cmake --build build --config Release --target nam_render
```

On Windows this produces `build/Release/nam_render.exe`; on other platforms,
`build/nam_render`. `hybrid/render.py`'s `find_nam_render_exe()` looks in
both locations (and on `PATH`) automatically.

## Manual CLI usage

```bash
nam_render <model.nam> <input.wav> [output.wav]
```

Input WAV's sample rate must match the model's expected sample rate (if the
`.nam` file declares one) -- `nam_render` errors out clearly if they don't
match rather than silently resampling. Mono input only.

## Experimental Sequential compatibility gate

The released runtime remains pinned to NAMCore `v0.5.4`. Canonical NAM 0.7
`Sequential`/`Linear` support is evaluated only against the explicitly pinned
upstream commit `2563c0fd4cb1f9ce457d89a761738ea15097e1f3`; it is not a
floating-branch dependency and is not yet used by normal exports.

Build the two renderers and run the gate with:

```bash
cmake -S . -B build
cmake --build build --target nam_render
cmake -S . -B build-sequential -DNAMCORE_GIT_TAG=2563c0fd4cb1f9ce457d89a761738ea15097e1f3
cmake --build build-sequential --target nam_render
NAM_RENDER_BASELINE=build/nam_render NAM_RENDER_SEQUENTIAL=build-sequential/nam_render \
  python -m pytest tests/test_namcore_sequential_gate.py -q
```

The gate proves ordinary WaveNet/Slimmable parity plus exact Linear and
Sequential FIR processing. A Sequential A2+Linear stream has a deterministic
canonical startup difference from separately prewarmed head rendering. The
gate derives the silence prefix from the complete Sequential child histories,
compares only the post-warm-up stream,
and fails for any later or block-boundary discrepancy. Embedded-cab export
must remain experimental until this gate passes on every supported renderer
build. The current Sequential wrapper does not forward the CLI's Full/Lite
selection into a nested SlimmableContainer, so experimental embedded output
must not claim a verified Full or Lite variant.
