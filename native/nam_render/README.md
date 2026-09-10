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
NAMCore's own `tools/render.cpp` CLI (`nam::get_dsp` loads the `.nam` file;
`dsp->process()` runs it block-by-block over a WAV).

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
