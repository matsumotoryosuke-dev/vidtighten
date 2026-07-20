#!/usr/bin/env bash
# Build VidTighten.app — a native macOS window app using pywebview.
# Usage: bash build_app.sh
# Output: VidTighten.app (current dir) — copy to /Applications to install.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="VidTighten"
APP_DIR="${SCRIPT_DIR}/${APP_NAME}.app"

# Prefer, in order:
#   1. <repo>/.venv           — produced by `scripts/bootstrap.sh` (uv sync --all-extras);
#                               reproducible from the committed lockfile, and what a
#                               freshly-cloned contributor checkout gets. This is now the
#                               canonical path — run scripts/bootstrap.sh to get it.
#   2. ~/.preprod/venv       — legacy hand-built venv from before scripts/bootstrap.sh
#                               existed. Kept as a fallback so existing checkouts (that
#                               haven't run the new bootstrap yet) keep building; not the
#                               documented path for anyone starting fresh.
#   3. Homebrew python3      — last resort. NOT guaranteed to work with whisperx: Homebrew
#                               python3 tracks the latest CPython, which can land outside
#                               whisperx's supported range (verified: requires Python
#                               >=3.10,<3.14 — see the requires-python comment in
#                               pyproject.toml). This fallback is a degraded-mode attempt,
#                               not a supported path — run scripts/bootstrap.sh instead.
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
elif [[ -x "${HOME}/.preprod/venv/bin/python" ]]; then
    PYTHON="${HOME}/.preprod/venv/bin/python"
elif [[ -x "/opt/homebrew/bin/python3" ]]; then
    PYTHON="/opt/homebrew/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

echo "Using Python: ${PYTHON}"
echo "Building ${APP_NAME}.app..."

# ── Run test suite before building ──────────────────────────────
echo "Running tests..."
"${PYTHON}" -m pytest tests/ -q || { echo "Tests failed — aborting build"; exit 1; }

# ── Clean previous build ─────────────────────────────────────────
rm -rf "${APP_DIR}"

# ── Directory structure ──────────────────────────────────────────
MACOS="${APP_DIR}/Contents/MacOS"
RESOURCES="${APP_DIR}/Contents/Resources"
mkdir -p "${MACOS}" "${RESOURCES}"

# ── 1. Launcher shell script (MacOS/VidTighten) ────────────────────
cat > "${MACOS}/${APP_NAME}" << SHELL
#!/usr/bin/env bash
# Inject Homebrew and common tool paths — Finder-launched apps don't inherit shell PATH
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH"
RESOURCES="\$(dirname "\$0")/../Resources"
exec "${PYTHON}" "\${RESOURCES}/app.py"
SHELL
chmod +x "${MACOS}/${APP_NAME}"

# ── 2. Python app launcher (Resources/app.py) ───────────────────
cat > "${RESOURCES}/app.py" << 'PYAPP'
#!/usr/bin/env python3
"""VidTighten native window launcher (embedded in .app bundle)."""
import os
import sys
import socket
import threading

# Add the Resources directory to path so `preprod` package is found
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from preprod.web import app as flask_app

PORT = 9877


def _find_free_port(start: int) -> int:
    for p in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start


class Api:
    """Python API exposed to JavaScript via window.pywebview.api.*"""

    def get_dropped_paths(self):
        """Return filesystem paths from the most recent Finder drag-drop.

        pywebview's performDragOperation_ reads real paths from NSPasteboard
        and stores them in _dnd_state['paths'] as (basename, full_path) tuples.
        We drain that buffer here so each drop is delivered exactly once.
        """
        from webview.dom import _dnd_state
        paths = [full for (_, full) in _dnd_state.get('paths', [])]
        _dnd_state['paths'] = []
        return paths


def main():
    port = _find_free_port(PORT)
    t = threading.Thread(
        target=lambda: flask_app.run(
            host="127.0.0.1", port=port, use_reloader=False, threaded=True
        ),
        daemon=True,
    )
    t.start()

    import webview

    # Tell pywebview's WKWebView subclass to capture Finder-drag paths from
    # NSPasteboard into _dnd_state['paths'].  performDragOperation_ only does
    # this when num_listeners > 0 (normally set by element.on('drop', ...)).
    # We set it directly here so the standard JS addEventListener('drop', ...)
    # in the frontend also benefits from the native path capture.
    from webview.dom import _dnd_state
    _dnd_state['num_listeners'] = 1

    webview.create_window(
        "VidTighten",
        f"http://127.0.0.1:{port}",
        js_api=Api(),
        width=1280,
        height=860,
        min_size=(800, 580),
    )
    webview.start()


if __name__ == "__main__":
    main()
PYAPP

# ── 3. Copy preprod Python package ──────────────────────────────
cp -R "${SCRIPT_DIR}/src/preprod" "${RESOURCES}/preprod"

# ── 4. Info.plist ────────────────────────────────────────────────
cat > "${APP_DIR}/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>            <string>VidTighten</string>
  <key>CFBundleDisplayName</key>     <string>VidTighten</string>
  <key>CFBundleIdentifier</key>      <string>com.vidtighten.app</string>
  <key>CFBundleVersion</key>         <string>0.1.0</string>
  <key>CFBundleShortVersionString</key> <string>0.1.0</string>
  <key>CFBundleExecutable</key>      <string>${APP_NAME}</string>
  <key>CFBundleIconFile</key>        <string>AppIcon</string>
  <key>CFBundlePackageType</key>     <string>APPL</string>
  <key>CFBundleSignature</key>       <string>????</string>
  <key>NSHighResolutionCapable</key> <true/>
  <key>NSHumanReadableCopyright</key> <string>VidTighten 0.1.0</string>
  <key>LSMinimumSystemVersion</key>  <string>12.0</string>
</dict>
</plist>
PLIST

# ── 5. Generate app icon — three overlapping pentagons (rotary rotor) ──
ICONSET_DIR=$(mktemp -d)/AppIcon.iconset
mkdir -p "${ICONSET_DIR}"

"${PYTHON}" - "${ICONSET_DIR}" << 'PYICON'
"""
Three pentagons arranged like Wankel rotary engine rotors:
  • Three different sizes, centers offset in an equilateral triangle
  • Each rotated progressively (24° apart — one-third of a pentagon vertex gap)
  • Black background, golden anti-aliased outlines with subtle glow
  • Rounded-square clip for macOS app icon shape
"""
import sys, struct, zlib, os, math
import numpy as np

iconset = sys.argv[1]
TWO_PI  = 2.0 * math.pi


def pentagon_edges(cx, cy, r, theta):
    """Return 5 edge tuples (ax,ay,bx,by) for a regular pentagon."""
    v = [(cx + r * math.cos(theta + TWO_PI * k / 5),
          cy + r * math.sin(theta + TWO_PI * k / 5)) for k in range(5)]
    return [(v[i][0], v[i][1], v[(i+1)%5][0], v[(i+1)%5][1]) for i in range(5)]


def create_png(w, h):
    cx, cy = w / 2.0, h / 2.0

    # ── Pentagon layout ──────────────────────────────────────────
    # Centers form an equilateral triangle offset from icon center.
    # a0=-90° puts the first center at top; +120° steps around clockwise.
    off   = w * 0.065
    a0    = -math.pi / 2.0
    base  = math.radians(-18)        # flat-top pentagon (vertex pointing up = -18° offset)
    step  = TWO_PI / 3.0

    pentagons = [
        # (cx, cy, radius, rotation)   — large / medium / small
        (cx + off * math.cos(a0),          cy + off * math.sin(a0),
         w * 0.390, base),

        (cx + off * math.cos(a0 + step),   cy + off * math.sin(a0 + step),
         w * 0.295, base + math.radians(24)),

        (cx + off * math.cos(a0 + 2*step), cy + off * math.sin(a0 + 2*step),
         w * 0.210, base + math.radians(48)),
    ]

    # Collect all 15 edge segments
    edges = []
    for (pcx, pcy, r, theta) in pentagons:
        edges.extend(pentagon_edges(pcx, pcy, r, theta))

    # ── Pixel coordinate grids ───────────────────────────────────
    xs = np.linspace(0.5, w - 0.5, w, dtype=np.float32)
    ys = np.linspace(0.5, h - 0.5, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)          # shape (h, w)

    # ── Minimum distance to any edge ─────────────────────────────
    min_d = np.full((h, w), 1e6, dtype=np.float32)
    for (ax, ay, bx, by) in edges:
        dx, dy   = bx - ax, by - ay
        len_sq   = float(dx*dx + dy*dy)
        if len_sq < 1e-10:
            d = np.hypot(xx - ax, yy - ay)
        else:
            t = np.clip(((xx - ax)*dx + (yy - ay)*dy) / len_sq, 0.0, 1.0)
            d = np.hypot(xx - (ax + t*dx), yy - (ay + t*dy))
        np.minimum(min_d, d, out=min_d)

    # ── Line metrics (thin, delicate) ────────────────────────────
    half_w  = max(0.35, w * 0.006)    # very thin solid core
    aa_w    = max(0.60, w * 0.005)    # tight anti-aliasing fade
    glow_hw = half_w * 4.5            # wider-but-faint glow halo
    glow_aw = glow_hw * 1.2           # soft glow falloff

    # Core alpha  [0..1]
    core_a  = np.clip(1.0 - (min_d - half_w) / aa_w,   0.0, 1.0)
    # Glow alpha  (very subtle halo — delicate shimmer only)
    glow_a  = np.clip(1.0 - (min_d - glow_hw) / glow_aw, 0.0, 1.0) * 0.15

    # ── Colors ───────────────────────────────────────────────────
    # Core line: rich gold  #D4AF37  → (212, 175, 55)
    # Glow:      warm amber #C8901A  → (200, 144, 26)  slightly deeper
    GOLD  = np.array([212, 175, 55],  dtype=np.float32)
    GLOW  = np.array([200, 144, 26],  dtype=np.float32)

    # Composite: glow first, then core on top (both over black bg)
    alpha_f = np.clip(core_a + glow_a * (1.0 - core_a), 0.0, 1.0)   # combined opacity

    # Blended colour at each pixel  (bg = 0,0,0)
    r_ch = np.clip(GLOW[0] * glow_a * (1-core_a) + GOLD[0] * core_a, 0, 255).astype(np.uint8)
    g_ch = np.clip(GLOW[1] * glow_a * (1-core_a) + GOLD[1] * core_a, 0, 255).astype(np.uint8)
    b_ch = np.clip(GLOW[2] * glow_a * (1-core_a) + GOLD[2] * core_a, 0, 255).astype(np.uint8)

    # ── Rounded-square clip (22% corner radius — macOS icon shape) ─
    rc    = w * 0.22
    rcx_l, rcx_r = rc, w - rc
    rcy_t, rcy_b = rc, h - rc
    outside = (
        ((xx < rcx_l) & (yy < rcy_t) & (np.hypot(xx - rcx_l, yy - rcy_t) > rc)) |
        ((xx > rcx_r) & (yy < rcy_t) & (np.hypot(xx - rcx_r, yy - rcy_t) > rc)) |
        ((xx < rcx_l) & (yy > rcy_b) & (np.hypot(xx - rcx_l, yy - rcy_b) > rc)) |
        ((xx > rcx_r) & (yy > rcy_b) & (np.hypot(xx - rcx_r, yy - rcy_b) > rc))
    )

    # ── Assemble RGBA ─────────────────────────────────────────────
    rgba        = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:,:,0] = r_ch
    rgba[:,:,1] = g_ch
    rgba[:,:,2] = b_ch
    rgba[:,:,3] = 255                     # fully opaque canvas (black bg)
    rgba[outside] = 0                     # transparent outside rounded corners

    # ── PNG encode ────────────────────────────────────────────────
    rows = b''.join(b'\x00' + rgba[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows, 6))
            + chunk(b'IEND', b''))


sizes = [16, 32, 64, 128, 256, 512, 1024]
for s in sizes:
    print(f"  {s}×{s}…", end=" ", flush=True)
    png = create_png(s, s)
    with open(os.path.join(iconset, f"icon_{s}x{s}.png"), 'wb') as f:
        f.write(png)
    if 16 < s <= 512:
        with open(os.path.join(iconset, f"icon_{s//2}x{s//2}@2x.png"), 'wb') as f:
            f.write(png)
    print("✓")

print("Icon generation complete")
PYICON

if command -v iconutil &> /dev/null; then
    iconutil -c icns "${ICONSET_DIR}" -o "${RESOURCES}/AppIcon.icns" 2>/dev/null \
        && echo "Icon created: AppIcon.icns" \
        || echo "iconutil failed — app will use default icon"
fi
rm -rf "$(dirname "${ICONSET_DIR}")"

# ── Done ─────────────────────────────────────────────────────────
echo ""
echo "✓ Built: ${APP_DIR}"
echo ""
echo "  To install in Applications:"
echo "    cp -R \"${APP_DIR}\" /Applications/"
echo ""
echo "  Or open directly:"
echo "    open \"${APP_DIR}\""
