"""
Tinder-style mask review UI.

Usage:
    python scripts/review_masks.py --pending_dir data/pseudo_labels/pending/

Opens http://localhost:7000 in your browser.
  Approve (→ key or button) → image + mask moved to pending_dir/approved/
  Reject  (← key or button) → image + mask moved to pending_dir/rejected/

After reviewing, run:
    python scripts/pseudo_label.py accept \\
        --pending_dir data/pseudo_labels/pending/approved/ \\
        --data_dir data/ --all
"""

import argparse
import csv
import glob
import json
import mimetypes
import os
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

PENDING_DIR: Path = Path()
PORT = 7000

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_images() -> list[dict]:
    seen: set[str] = set()
    result: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
        for p in glob.glob(str(PENDING_DIR / ext)):
            base = Path(p).stem
            if base.endswith("_mask") or base in seen:
                continue
            if (PENDING_DIR / f"{base}_mask.png").exists():
                seen.add(base)
                result.append(base)
    result.sort()
    return result


def load_confidence() -> dict[str, float]:
    scores: dict[str, float] = {}
    csv_path = PENDING_DIR / "confidence_scores.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                scores[row["basename"]] = float(row["mean_confidence"])
    return scores


def find_image_file(basename: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = PENDING_DIR / (basename + ext)
        if p.exists():
            return p
    return None


def move_pair(basename: str, dest_sub: str) -> None:
    dest = PENDING_DIR / dest_sub
    dest.mkdir(exist_ok=True)
    img = find_image_file(basename)
    mask = PENDING_DIR / f"{basename}_mask.png"
    if img and img.exists():
        shutil.move(str(img), str(dest / img.name))
    if mask.exists():
        shutil.move(str(mask), str(dest / mask.name))


def safe_path(fname: str) -> Path | None:
    try:
        resolved = (PENDING_DIR / fname).resolve()
        if resolved.is_relative_to(PENDING_DIR.resolve()):
            return resolved
    except Exception:
        pass
    return None


# ── Embedded UI ────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Mask Review</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #111;
    color: #e8e8e8;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 24px 32px;
    user-select: none;
  }

  /* ── Header ── */
  .top {
    width: 100%;
    max-width: 1100px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .top-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .counter { font-size: 13px; font-weight: 600; color: #aaa; white-space: nowrap; }
  .progress-track {
    flex: 1; height: 3px;
    background: #252525; border-radius: 2px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: #4ade80;
    transition: width 0.25s ease;
  }
  .conf { font-size: 12px; white-space: nowrap; }
  .conf-high { color: #4ade80; }
  .conf-mid  { color: #fbbf24; }
  .conf-low  { color: #f87171; }

  .basename {
    font-family: monospace; font-size: 11px; color: #444;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }

  /* ── Image row ── */
  .images {
    display: flex;
    gap: 10px;
    width: 100%;
    max-width: 1100px;
  }
  .img-card {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: center;
    min-width: 0;
  }
  .img-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #444;
  }
  .img-card img,
  .img-card canvas {
    width: 100%;
    height: auto;
    max-height: 54vh;
    object-fit: contain;
    border-radius: 8px;
    border: 1px solid #1e1e1e;
    background: #161616;
    display: block;
  }

  /* ── Buttons ── */
  .buttons {
    display: flex;
    gap: 16px;
    justify-content: center;
    width: 100%;
    max-width: 1100px;
  }
  .btn {
    flex: 1;
    max-width: 260px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 14px 0;
    border-radius: 10px;
    border: 1px solid transparent;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    transition: filter 0.1s, transform 0.08s;
    letter-spacing: 0.02em;
  }
  .btn:active { transform: scale(0.97); }
  .btn-reject  { background: #2a1010; color: #f87171; border-color: #4a1c1c; }
  .btn-reject:hover  { filter: brightness(1.25); }
  .btn-approve { background: #0b2415; color: #4ade80; border-color: #1a5430; }
  .btn-approve:hover { filter: brightness(1.25); }
  .key {
    font-size: 10px; font-weight: 400; opacity: 0.4;
    border: 1px solid currentColor; border-radius: 3px;
    padding: 1px 5px;
  }

  /* ── Done screen ── */
  .done {
    display: flex; flex-direction: column;
    align-items: center; gap: 24px; text-align: center;
  }
  .done h2 { font-size: 24px; font-weight: 700; }
  .stats { display: flex; gap: 48px; }
  .stat .num { font-size: 44px; font-weight: 800; line-height: 1; }
  .stat .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #555; margin-top: 5px; }
  .num-green { color: #4ade80; }
  .num-red   { color: #f87171; }

  .cmd-wrap { display: flex; flex-direction: column; gap: 6px; align-items: center; }
  .cmd-label { font-size: 12px; color: #555; }
  .cmd-box {
    background: #181818; border: 1px solid #2a2a2a; border-radius: 8px;
    padding: 12px 18px; font-family: monospace; font-size: 12px; color: #aaa;
    max-width: 640px; word-break: break-all; text-align: left;
    cursor: pointer; transition: border-color 0.2s, color 0.2s;
  }
  .cmd-box:hover { border-color: #444; color: #ccc; }
  .copy-hint { font-size: 10px; color: #3a3a3a; }

  .empty { color: #555; font-size: 14px; text-align: center; }
</style>
</head>
<body>
<div id="app"><div class="empty">Loading…</div></div>
<script>
let images   = [];
let current  = 0;
let nApproved = 0;
let nRejected = 0;

async function init() {
  const res = await fetch('/api/images');
  images = await res.json();
  images.length ? render() : showEmpty();
}

function confInfo(c) {
  if (c == null) return null;
  const pct = (c * 100).toFixed(1);
  if (c >= 0.65) return { cls: 'conf-high', label: `${pct}% confidence` };
  if (c >= 0.35) return { cls: 'conf-mid',  label: `${pct}% confidence` };
  return               { cls: 'conf-low',  label: `${pct}% confidence` };
}

// Build red overlay using canvas pixel ops
async function buildOverlay(origImg, maskSrc) {
  const maskImg = await new Promise((res, rej) => {
    const m = new Image();
    m.onload = () => res(m);
    m.onerror = rej;
    m.src = maskSrc;
  });

  const W = origImg.naturalWidth  || origImg.width;
  const H = origImg.naturalHeight || origImg.height;

  // Draw original
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(origImg, 0, 0, W, H);

  // Draw mask scaled to same size
  const tmp = document.createElement('canvas');
  tmp.width = W; tmp.height = H;
  tmp.getContext('2d').drawImage(maskImg, 0, 0, W, H);
  const maskPx = tmp.getContext('2d').getImageData(0, 0, W, H).data;

  // Tint white mask pixels red on the original
  const frame = ctx.getImageData(0, 0, W, H);
  const d = frame.data;
  for (let i = 0; i < maskPx.length; i += 4) {
    if (maskPx[i] > 128) {         // white = crack
      d[i]   = 220;  // R
      d[i+1] = 38;   // G
      d[i+2] = 38;   // B
      // leave alpha as-is
    }
  }
  ctx.putImageData(frame, 0, 0);
  return canvas;
}

function render() {
  if (current >= images.length) { showDone(); return; }
  const item = images[current];
  const pct  = (current / images.length * 100).toFixed(1);
  const conf = confInfo(item.confidence);

  const maskSrc = `/img/${encodeURIComponent(item.basename)}_mask.png`;
  const exts    = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'];

  document.getElementById('app').innerHTML = `
    <div class="top">
      <div class="top-row">
        <span class="counter">${current + 1} / ${images.length}</span>
        <div class="progress-track">
          <div class="progress-fill" style="width:${pct}%"></div>
        </div>
        ${conf ? `<span class="conf ${conf.cls}">${conf.label}</span>` : ''}
      </div>
      <div class="basename">${item.basename}</div>
    </div>

    <div class="images">
      <div class="img-card">
        <div class="img-label">Original</div>
        <img id="orig-img" alt="original" />
      </div>
      <div class="img-card">
        <div class="img-label">Predicted Mask</div>
        <img src="${maskSrc}" alt="mask" />
      </div>
      <div class="img-card">
        <div class="img-label">Overlay</div>
        <canvas id="overlay-canvas"></canvas>
      </div>
    </div>

    <div class="buttons">
      <button class="btn btn-reject"  onclick="act('reject')">
        <span class="key">←</span> Reject
      </button>
      <button class="btn btn-approve" onclick="act('approve')">
        Approve <span class="key">→</span>
      </button>
    </div>
  `;

  // Load original image, trying extensions in order
  const origEl = document.getElementById('orig-img');
  let ei = 0;
  const tryNext = () => {
    if (ei >= exts.length) return;
    origEl.src = `/img/${encodeURIComponent(item.basename)}${exts[ei++]}`;
  };
  origEl.onerror = tryNext;
  origEl.onload  = async () => {
    try {
      const overlayCanvas = await buildOverlay(origEl, maskSrc);
      const slot = document.getElementById('overlay-canvas');
      if (slot) {
        slot.width  = overlayCanvas.width;
        slot.height = overlayCanvas.height;
        slot.getContext('2d').drawImage(overlayCanvas, 0, 0);
      }
    } catch (_) { /* mask not loaded yet — ignore */ }
  };
  tryNext();
}

async function act(action) {
  const item = images[current];
  await fetch(`/api/${action}/${encodeURIComponent(item.basename)}`, { method: 'POST' });
  action === 'approve' ? nApproved++ : nRejected++;
  current++;
  render();
}

function showDone() {
  const cmd = `python scripts/pseudo_label.py accept --pending_dir data/pseudo_labels/pending/approved/ --data_dir data/ --all`;
  document.getElementById('app').innerHTML = `
    <div class="done">
      <h2>Review complete</h2>
      <div class="stats">
        <div class="stat">
          <div class="num num-green">${nApproved}</div>
          <div class="lbl">Approved</div>
        </div>
        <div class="stat">
          <div class="num num-red">${nRejected}</div>
          <div class="lbl">Rejected</div>
        </div>
      </div>
      ${nApproved > 0 ? `
      <div class="cmd-wrap">
        <div class="cmd-label">Add approved images to your training set:</div>
        <div class="cmd-box" onclick="copyCmd(this)">${cmd}</div>
        <div class="copy-hint">click to copy</div>
      </div>` : '<p style="color:#555;font-size:13px">No images approved.</p>'}
    </div>
  `;
}

function showEmpty() {
  document.getElementById('app').innerHTML = `
    <div class="empty">
      <p>No pending images found in this directory.</p>
      <p style="margin-top:8px;font-size:12px;color:#3a3a3a">Run the generate step first.</p>
    </div>
  `;
}

function copyCmd(el) {
  navigator.clipboard.writeText(el.textContent.trim()).then(() => {
    el.style.borderColor = '#4ade80';
    setTimeout(() => el.style.borderColor = '', 1400);
  });
}

document.addEventListener('keydown', e => {
  if (current >= images.length) return;
  if (e.key === 'ArrowRight') act('approve');
  if (e.key === 'ArrowLeft')  act('reject');
});

init();
</script>
</body>
</html>"""


# ── HTTP handler ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path).split("?")[0]

        if path in ("/", "/index.html"):
            self._respond(200, "text/html", HTML.encode())

        elif path == "/api/images":
            imgs = get_images()
            conf = load_confidence()
            data = [{"basename": b, "confidence": conf.get(b)} for b in imgs]
            self._json(data)

        elif path.startswith("/img/"):
            fpath = safe_path(path[5:])
            if fpath and fpath.is_file():
                mime, _ = mimetypes.guess_type(str(fpath))
                self._respond(200, mime or "application/octet-stream", fpath.read_bytes())
            else:
                self._respond(404, "text/plain", b"Not found")

        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self):
        path = unquote(self.path).split("?")[0]
        if path.startswith("/api/approve/"):
            move_pair(path[13:], "approved")
            self._json({"ok": True})
        elif path.startswith("/api/reject/"):
            move_pair(path[12:], "rejected")
            self._json({"ok": True})
        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data):
        self._respond(200, "application/json", json.dumps(data).encode())

    def log_message(self, *_):
        pass


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global PENDING_DIR

    parser = argparse.ArgumentParser(description="Tinder-style mask review UI.")
    parser.add_argument("--pending_dir", default="data/pseudo_labels/pending/",
                        help="Folder containing generated images + masks.")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    PENDING_DIR = Path(args.pending_dir).resolve()
    if not PENDING_DIR.exists():
        print(f"[error] pending_dir not found: {PENDING_DIR}")
        return

    url = f"http://localhost:{args.port}"
    print(f"[review] Serving {PENDING_DIR}")
    print(f"[review] Open → {url}")
    print(f"[review] Ctrl+C to stop\n")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[review] Stopped.")


if __name__ == "__main__":
    main()
