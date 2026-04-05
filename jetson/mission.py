"""
jetson/mission.py — Autonomous 1m forward flight + live crack detection + web UI

Connects to Pixhawk 6C via USB (/dev/ttyACM0), streams telemetry and live
crack-detection camera feed over WiFi to any browser on the same network.

Usage:
    # Camera + UI only (no flight)
    python3 jetson/mission.py --dry_run

    # Full autonomous mission
    python3 jetson/mission.py

    # Alternate USB port
    python3 jetson/mission.py --serial /dev/ttyACM1

Web UI: http://192.168.1.222:5000

Install deps (run once on Jetson):
    pip install mavsdk flask

PX4 setup (do once in QGroundControl / Mission Planner):
    - Set SERIAL_PX4IO to enable USB MAVLink  (usually already on)
    - Set COM_RC_OVERRIDE = 3  (RC sticks override offboard + auto modes)
    - Set MPC_XY_VEL_MAX, MPC_Z_VEL_MAX to safe values for indoor flight
"""

import argparse
import asyncio
import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify
from torchvision import transforms

from crack_detection.models.efficientcracknet import EfficientCrackNet


# ---------------------------------------------------------------------------
# Shared state (thread-safe)
# ---------------------------------------------------------------------------

class State:
    def __init__(self):
        self._lock = threading.Lock()
        self._overlay = None
        self._crack_ratio = 0.0
        self._crack_log = []        # list of dicts
        self._flight_status = "idle"
        self._telemetry = {
            "armed": False,
            "flight_mode": "N/A",
            "altitude_m": 0.0,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
            "vx_ms": 0.0,
            "vy_ms": 0.0,
            "vz_ms": 0.0,
            "pos_north_m": 0.0,
            "pos_east_m": 0.0,
            "battery_v": 0.0,
            "battery_pct": 0.0,
        }

    # overlay
    def set_overlay(self, frame):
        with self._lock: self._overlay = frame.copy()
    def get_overlay(self):
        with self._lock: return self._overlay

    # crack ratio
    def set_crack_ratio(self, r):
        with self._lock: self._crack_ratio = r
    def get_crack_ratio(self):
        with self._lock: return self._crack_ratio

    # crack log
    def add_crack(self, entry):
        with self._lock: self._crack_log.append(entry)
    def get_crack_log(self):
        with self._lock: return list(self._crack_log)

    # flight status
    def set_status(self, s):
        with self._lock: self._flight_status = s
    def get_status(self):
        with self._lock: return self._flight_status

    # telemetry
    def update_telemetry(self, d):
        with self._lock: self._telemetry.update(d)
    def get_telemetry(self):
        with self._lock: return dict(self._telemetry)


STATE = State()


# ---------------------------------------------------------------------------
# Camera — identical pipeline to live_inference.py
# ---------------------------------------------------------------------------

SENSOR_MODE_DIMS = {
    0: (4032, 3040),
    1: (3840, 2160),
    2: (1920, 1080),
}


def build_gst_pipeline(sensor_mode: int, wbmode: int = 1) -> str:
    w, h = SENSOR_MODE_DIMS[sensor_mode]
    fps = {0: 21, 1: 30, 2: 60}[sensor_mode]
    return (
        f"nvarguscamerasrc sensor-id=0 sensor-mode={sensor_mode} wbmode={wbmode} "
        f"tnr-mode=1 tnr-strength=1.0 ee-mode=1 ee-strength=0.1 "
        f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 "
        f"! nvvidconv left=496 right=3544 top=0 bottom=3040 "
        f"! video/x-raw(memory:NVMM),width=512,height=512 "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink max-buffers=1 drop=true sync=false"
    )


class FrameGrabber:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self._lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock: self.frame = frame
            else:
                time.sleep(0.005)

    def get(self):
        with self._lock: return self.frame

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# Inference — identical to live_inference.py
# ---------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

CRACK_PIXEL_THRESHOLD = 0.5   # sigmoid cutoff (same as live_inference default)
CRACK_LOG_RATIO       = 0.05  # log position if >5% of frame is crack


def inference_loop(model, device: torch.device, fp16: bool, grabber: FrameGrabber):
    """Runs continuously in its own thread."""
    while True:
        frame = grabber.get()
        if frame is None:
            time.sleep(0.01)
            continue

        tensor = TRANSFORM(frame).unsqueeze(0).to(device)
        if fp16:
            tensor = tensor.half()
        with torch.no_grad():
            out = model(tensor)
        mask = (out[0, 0].float().cpu().numpy() > CRACK_PIXEL_THRESHOLD).astype(np.uint8) * 255

        crack_ratio = float((mask > 127).sum()) / mask.size
        STATE.set_crack_ratio(crack_ratio)

        # Log crack location (using NED position from Pixhawk)
        if crack_ratio > CRACK_LOG_RATIO:
            tel = STATE.get_telemetry()
            entry = {
                "time":    time.strftime("%H:%M:%S"),
                "north_m": round(tel["pos_north_m"], 3),
                "east_m":  round(tel["pos_east_m"], 3),
                "alt_m":   round(tel["altitude_m"], 3),
                "pct":     round(crack_ratio * 100, 1),
            }
            log = STATE.get_crack_log()
            if (not log or
                    abs(log[-1]["north_m"] - entry["north_m"]) > 0.05 or
                    abs(log[-1]["east_m"]  - entry["east_m"])  > 0.05):
                STATE.add_crack(entry)

        # Build overlay (same as live_inference.py)
        overlay = frame.copy()
        colored = np.zeros_like(frame)
        colored[mask > 127] = (0, 0, 220)
        overlay = cv2.addWeighted(overlay, 0.55, colored, 0.45, 0)
        label_color = (0, 0, 220) if crack_ratio > CRACK_LOG_RATIO else (0, 255, 0)
        cv2.putText(overlay, f"Cracks: {crack_ratio*100:.1f}%  |  {STATE.get_status()}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        STATE.set_overlay(overlay)


# ---------------------------------------------------------------------------
# MAVSDK flight mission
# ---------------------------------------------------------------------------

async def _flight_mission(serial: str, baud: int):
    from mavsdk import System
    from mavsdk.offboard import OffboardError, PositionNedYaw

    drone = System()
    STATE.set_status("connecting")
    await drone.connect(system_address=f"serial://{serial}:{baud}")

    print("Waiting for Pixhawk connection...")
    async for s in drone.core.connection_state():
        if s.is_connected:
            print("Pixhawk connected.")
            break

    # --- Telemetry streams ---
    async def _pos():
        async for p in drone.telemetry.position():
            STATE.update_telemetry({"altitude_m": round(p.relative_altitude_m, 2)})

    async def _att():
        async for a in drone.telemetry.attitude_euler():
            STATE.update_telemetry({
                "roll_deg":  round(a.roll_deg, 1),
                "pitch_deg": round(a.pitch_deg, 1),
                "yaw_deg":   round(a.yaw_deg, 1),
            })

    async def _vel():
        async for v in drone.telemetry.velocity_ned():
            STATE.update_telemetry({
                "vx_ms": round(v.north_m_s, 2),
                "vy_ms": round(v.east_m_s, 2),
                "vz_ms": round(v.down_m_s, 2),
            })

    async def _bat():
        async for b in drone.telemetry.battery():
            STATE.update_telemetry({
                "battery_v":   round(b.voltage_v, 2),
                "battery_pct": round((b.remaining_percent or 0) * 100, 0),
            })

    async def _armed():
        async for armed in drone.telemetry.armed():
            STATE.update_telemetry({"armed": armed})

    async def _mode():
        async for m in drone.telemetry.flight_mode():
            STATE.update_telemetry({"flight_mode": str(m)})

    async def _ned():
        async for pv in drone.telemetry.position_velocity_ned():
            STATE.update_telemetry({
                "pos_north_m": round(pv.position.north_m, 3),
                "pos_east_m":  round(pv.position.east_m, 3),
            })

    for coro in (_pos, _att, _vel, _bat, _armed, _mode, _ned):
        asyncio.ensure_future(coro())

    await asyncio.sleep(2)   # let telemetry streams settle
    STATE.set_status("ready")
    print("Telemetry live. Mission standing by.")
    print("  -> Call start_mission() or use --dry_run to skip flight.")

    # Mission runs only when explicitly triggered (see run_mission flag)
    while not _START_MISSION.is_set():
        await asyncio.sleep(0.2)

    # --- Arm + offboard ---
    STATE.set_status("arming")
    await drone.action.arm()
    await asyncio.sleep(1)

    # Must send at least one setpoint before starting offboard
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        STATE.set_status("error")
        return

    # Takeoff to 30 cm
    STATE.set_status("takeoff")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -0.3, 0.0))
    await asyncio.sleep(5)

    # Fly 1 m forward (north), hold altitude 30 cm
    STATE.set_status("cruising")
    await drone.offboard.set_position_ned(PositionNedYaw(1.0, 0.0, -0.3, 0.0))
    await asyncio.sleep(7)   # generous for ~0.2 m/s approach

    # Hover at destination
    STATE.set_status("hovering")
    await asyncio.sleep(2)

    # Land
    STATE.set_status("landing")
    await drone.offboard.stop()
    await drone.action.land()
    await asyncio.sleep(6)
    STATE.set_status("done")
    print("Mission complete.")


# Event that triggers the flight sequence (set via /start_mission endpoint)
_START_MISSION = threading.Event()


def _run_mavsdk(serial: str, baud: int):
    asyncio.run(_flight_mission(serial, baud))


# ---------------------------------------------------------------------------
# Flask web UI
# ---------------------------------------------------------------------------

app = Flask(__name__)

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mission Control — Crack Detection</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d0d; color: #ddd; font-family: 'Courier New', monospace; padding: 12px; }
h1 { color: #29b6f6; font-size: 16px; margin-bottom: 10px; letter-spacing: 1px; }
.grid { display: grid; grid-template-columns: 512px 1fr; gap: 10px; }
.panel { background: #161616; border: 1px solid #2a2a2a; border-radius: 6px; padding: 10px; }
.panel-title { color: #29b6f6; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
img#feed { width: 512px; height: 512px; display: block; border-radius: 4px; background: #000; }
.crack-bar-wrap { margin-top: 6px; background: #222; border-radius: 3px; height: 8px; }
.crack-bar { height: 8px; border-radius: 3px; background: #e53935; transition: width 0.3s; }
.telem-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; }
.telem-item { font-size: 12px; padding: 3px 0; border-bottom: 1px solid #1e1e1e; }
.telem-label { color: #666; font-size: 10px; }
.telem-val { color: #fff; font-weight: bold; }
.status-badge {
  display: inline-block; padding: 3px 10px; border-radius: 4px;
  font-size: 13px; font-weight: bold; margin-bottom: 10px; text-transform: uppercase;
}
.s-idle, .s-ready      { background: #1a1a1a; color: #888; }
.s-connecting          { background: #3e2723; color: #ff8f00; }
.s-arming, .s-takeoff  { background: #1b5e20; color: #69f0ae; }
.s-cruising            { background: #0d47a1; color: #82b1ff; }
.s-hovering            { background: #1a237e; color: #b0bec5; }
.s-landing             { background: #e65100; color: #ffe0b2; }
.s-done                { background: #006064; color: #80deea; }
.s-error               { background: #b71c1c; color: #fff; }
.s-dry_run             { background: #1b5e20; color: #b9f6ca; }
table.crack-log { width: 100%; font-size: 11px; border-collapse: collapse; }
table.crack-log th { color: #555; font-weight: normal; text-align: left; padding: 2px 5px; }
table.crack-log td { padding: 3px 5px; border-bottom: 1px solid #1e1e1e; }
table.crack-log tr:hover td { background: #1c1c1c; }
.pct-badge { background: #b71c1c; color: #fff; border-radius: 3px; padding: 1px 5px; font-size: 10px; }
.btn {
  display: inline-block; margin-top: 10px; padding: 7px 18px;
  background: #1565c0; color: #fff; border: none; border-radius: 4px;
  font-family: monospace; font-size: 12px; cursor: pointer; letter-spacing: 0.5px;
}
.btn:hover { background: #1976d2; }
.btn:disabled { background: #333; color: #555; cursor: default; }
.crack-pct-label { font-size: 12px; color: #888; margin-top: 5px; }
.crack-pct-val { color: #e53935; font-weight: bold; }
</style>
</head>
<body>
<h1>&#9679; MISSION CONTROL &mdash; CRACK DETECTION</h1>
<div class="grid">

  <!-- LEFT: Camera feed -->
  <div>
    <div class="panel">
      <div class="panel-title">Live Camera Feed (512×512)</div>
      <img id="feed" src="/video_feed" alt="feed">
      <div class="crack-pct-label">Frame crack coverage: <span class="crack-pct-val" id="crack-pct">0.0%</span></div>
      <div class="crack-bar-wrap"><div class="crack-bar" id="crack-bar" style="width:0%"></div></div>
    </div>
  </div>

  <!-- RIGHT: Status + Telemetry + Log -->
  <div style="display:flex; flex-direction:column; gap:10px;">

    <!-- Flight status + control -->
    <div class="panel">
      <div class="panel-title">Flight</div>
      <div class="status-badge s-idle" id="status-badge">idle</div>
      <div class="telem-grid">
        <div class="telem-item"><div class="telem-label">Armed</div><div class="telem-val" id="t-armed">—</div></div>
        <div class="telem-item"><div class="telem-label">Mode</div><div class="telem-val" id="t-mode">—</div></div>
        <div class="telem-item"><div class="telem-label">Altitude</div><div class="telem-val" id="t-alt">—</div></div>
        <div class="telem-item"><div class="telem-label">Battery</div><div class="telem-val" id="t-bat">—</div></div>
        <div class="telem-item"><div class="telem-label">Roll / Pitch</div><div class="telem-val" id="t-rp">—</div></div>
        <div class="telem-item"><div class="telem-label">Yaw</div><div class="telem-val" id="t-yaw">—</div></div>
        <div class="telem-item"><div class="telem-label">Vel N/E/D (m/s)</div><div class="telem-val" id="t-vel">—</div></div>
        <div class="telem-item"><div class="telem-label">Pos North/East (m)</div><div class="telem-val" id="t-pos">—</div></div>
      </div>
      <button class="btn" id="start-btn" onclick="startMission()">&#9654; START MISSION</button>
    </div>

    <!-- Crack log -->
    <div class="panel" style="flex:1; overflow:hidden;">
      <div class="panel-title">Crack Locations Logged</div>
      <div style="max-height:240px; overflow-y:auto;">
        <table class="crack-log">
          <thead><tr><th>Time</th><th>North (m)</th><th>East (m)</th><th>Alt (m)</th><th>Coverage</th></tr></thead>
          <tbody id="crack-tbody"><tr><td colspan="5" style="color:#333; padding:6px;">None detected yet</td></tr></tbody>
        </table>
      </div>
    </div>

    <!-- Flight plan -->
    <div class="panel">
      <div class="panel-title">Flight Plan</div>
      <div style="font-size:11px; color:#666; line-height:1.8;">
        <div>&#9312; Arm</div>
        <div>&#9313; Takeoff → 0.3 m altitude</div>
        <div>&#9314; Fly 1.0 m forward (north)</div>
        <div>&#9315; Hover 2 s</div>
        <div>&#9316; Land + disarm</div>
      </div>
    </div>

  </div>
</div>

<script>
const STATUS_ACTIVE = new Set(['arming','takeoff','cruising','hovering','landing']);

function startMission() {
  fetch('/start_mission', {method:'POST'})
    .then(r => r.json())
    .then(d => { if (!d.ok) alert(d.msg); });
}

function poll() {
  fetch('/telemetry').then(r => r.json()).then(d => {
    const t = d.telemetry;
    const s = d.flight_status;

    // Status badge
    const badge = document.getElementById('status-badge');
    badge.textContent = s;
    badge.className = 'status-badge s-' + s;

    // Disable button once mission starts
    const btn = document.getElementById('start-btn');
    btn.disabled = STATUS_ACTIVE.has(s) || s === 'done' || s === 'dry_run';

    // Telemetry
    document.getElementById('t-armed').textContent = t.armed ? '✓ ARMED' : 'Disarmed';
    document.getElementById('t-armed').style.color  = t.armed ? '#69f0ae' : '#e57373';
    document.getElementById('t-mode').textContent  = t.flight_mode;
    document.getElementById('t-alt').textContent   = t.altitude_m + ' m';
    document.getElementById('t-bat').textContent   = t.battery_v + ' V  (' + t.battery_pct + '%)';
    document.getElementById('t-rp').textContent    = t.roll_deg + '° / ' + t.pitch_deg + '°';
    document.getElementById('t-yaw').textContent   = t.yaw_deg + '°';
    document.getElementById('t-vel').textContent   = t.vx_ms + ' / ' + t.vy_ms + ' / ' + t.vz_ms;
    document.getElementById('t-pos').textContent   = t.pos_north_m + ' / ' + t.pos_east_m;

    // Crack coverage bar
    const pct = d.crack_ratio_pct;
    document.getElementById('crack-pct').textContent = pct.toFixed(1) + '%';
    document.getElementById('crack-bar').style.width = Math.min(pct * 4, 100) + '%';

    // Crack log (newest first)
    const rows = d.crack_log;
    const tbody = document.getElementById('crack-tbody');
    if (rows.length === 0) return;
    tbody.innerHTML = [...rows].reverse().map(e =>
      `<tr>
        <td>${e.time}</td>
        <td>${e.north_m}</td>
        <td>${e.east_m}</td>
        <td>${e.alt_m}</td>
        <td><span class="pct-badge">${e.pct}%</span></td>
      </tr>`
    ).join('');
  });
}
setInterval(poll, 400);
poll();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return _UI_HTML


def _gen_frames():
    while True:
        frame = STATE.get_overlay()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/telemetry")
def telemetry():
    return jsonify({
        "telemetry":      STATE.get_telemetry(),
        "flight_status":  STATE.get_status(),
        "crack_log":      STATE.get_crack_log(),
        "crack_ratio_pct": STATE.get_crack_ratio() * 100,
    })


@app.route("/start_mission", methods=["POST"])
def start_mission():
    status = STATE.get_status()
    if status not in ("ready", "idle"):
        return jsonify({"ok": False, "msg": f"Cannot start: status is '{status}'"})
    _START_MISSION.set()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Crack detection mission controller")
    p.add_argument("--model_path", default="results/saved_models/EfficientCrackNet/best_model_num_real_4.pt")
    p.add_argument("--sensor_mode", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--wbmode",      type=int, default=1)
    p.add_argument("--fp16",        action="store_true", default=True)
    p.add_argument("--serial",      default="/dev/ttyACM0",
                   help="Pixhawk USB serial port (default: /dev/ttyACM0)")
    p.add_argument("--baud",        type=int, default=921600)
    p.add_argument("--port",        type=int, default=5000,
                   help="Flask web UI port")
    p.add_argument("--dry_run",     action="store_true",
                   help="Camera + UI only — no Pixhawk connection, no flight")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model (same as live_inference.py)
    model = EfficientCrackNet().to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    model.eval()
    if args.fp16:
        model = model.half()
        print("Model in FP16.")

    # Camera
    pipeline = build_gst_pipeline(args.sensor_mode, args.wbmode)
    print(f"Opening camera pipeline...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: Could not open camera via GStreamer.")
        return
    grabber = FrameGrabber(cap)

    print("Waiting for first frame...")
    while grabber.get() is None:
        time.sleep(0.05)
    print("Camera ready.")

    # Inference thread
    threading.Thread(
        target=inference_loop,
        args=(model, device, args.fp16, grabber),
        daemon=True,
    ).start()

    # MAVSDK thread
    if args.dry_run:
        STATE.set_status("dry_run")
        print("Dry run — no Pixhawk connection.")
    else:
        threading.Thread(
            target=_run_mavsdk,
            args=(args.serial, args.baud),
            daemon=True,
        ).start()

    # Flask (blocks main thread)
    print(f"\nWeb UI: http://192.168.1.222:{args.port}")
    print("Open that URL in your browser.\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)

    grabber.stop()
    cap.release()


if __name__ == "__main__":
    main()
