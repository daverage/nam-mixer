# Desktop app icons

`icon.icns` (macOS), `icon.ico` (Windows), and `icon.png` (Linux) are all
rendered from `static/nam-mixer-logo.png` (512x512) and wired into
`desktop/build.spec`'s `EXE`/`BUNDLE` `icon=` arguments.

To regenerate after that source logo changes:

```bash
pip install pillow  # not otherwise a runtime dependency
mkdir -p /tmp/nam_icon.iconset
python3 - <<'EOF'
from PIL import Image
src = Image.open("static/nam-mixer-logo.png").convert("RGBA")

sizes = [16, 24, 32, 48, 64, 128, 256]
src.save("desktop/icons/icon.ico", sizes=[(s, s) for s in sizes])
src.resize((256, 256), Image.LANCZOS).save("desktop/icons/icon.png")

for name, size in [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
]:
    src.resize((size, size), Image.LANCZOS).save(f"/tmp/nam_icon.iconset/{name}")
EOF

# macOS only -- iconutil isn't available on Windows/Linux:
iconutil -c icns /tmp/nam_icon.iconset -o desktop/icons/icon.icns
```

If `iconutil` isn't available (building on Windows/Linux), the `.icns` step
can be skipped -- `build.spec` falls back to no icon on macOS builds that
lack it, without failing the build.
