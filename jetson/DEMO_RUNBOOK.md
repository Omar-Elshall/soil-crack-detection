# Senior Design Defense — Demo Runbook

10-minute live demo. Drone is already powered on, services running, UI open in the browser. No setup or pitch — straight into the product.

---

## System State (final demo config)

| Component | State |
|---|---|
| **Model** | `best_model_num_real_4.pt` (real_6 available — Switching Models below) |
| **Detection threshold** | 0.5 (real_4 default) |
| **Sensor mode** | 1 → 3840×2160 @ 30 FPS, center-cropped to 2160×2160 |
| **Stream resolution** | 1440×1440 lossless PNG (~1.9 MB/frame) |
| **Live FPS** | ~5.5 in MAXN_SUPER (PyTorch FP16) |
| **Power profile** | MAXN_SUPER, locked at boot via `maxn-boot.service` |
| Inference (`:8001`) / MAVLink (`:8002`) / Data (`:8003`) / UI (`:5173`) | All up via `soilcrack.service` at boot |
| Camera | Arducam IMX477 via nvargus |
| Telemetry path | SiK radio (laptop USB) primary; Jetson WiFi MAVLink fallback |
| Network | Jetson hosts `soil-crack-demo` AP; laptop joins it; iPhone USB-tether for laptop's internet |

### TRT acceleration — investigated, kept PyTorch
Compiled real_4 to TensorRT FP16 successfully (after raising workspace to 4 GB and stopping concurrent services). Result: **2.6 FPS vs PyTorch's 5.5 FPS** in MAXN. TRT falls back to FP32 on the MobileViT attention layers (build log: `Precision: FP32+FP16`), wiping the FP16 speedup. PyTorch's native FP16 SDPA kernels handle attention faster. **Demo runs PyTorch FP16.**

---

## Demo Flow (10 min)

### A) UI tour (1 min)

Browser is already on `http://soilcrack.local:5173`. Walk through the layout in this order:

- **Camera tile, top-left** — live downward feed with red crack-mask overlay, ~5 FPS
- **Status bar, top-right** — four green dots: Inference, MAVLink·radio, Camera, Data
- **Telemetry panel, right side** — battery V/%, flight mode, armed/disarmed, attitude (roll/pitch/yaw), altitude, GPS fix
- **Crack-ratio chart** under the camera — % of mask pixels per frame, updating live
- **Detection log** under the chart — every frame with ≥0.10 % crack coverage gets a row with timestamp + lat/lon/alt/heading
- **Pages** (left rail): Live · Plan · History · Flight Controls

> "All telemetry numbers are coming over the SiK radio plugged into the laptop, not WiFi. The drone could be hundreds of meters away and these still update."

### B) Manual detection mission over A1 prints (3-4 min)

Two A1 sheets taped to the ground (tape down — outdoors wind will move them) in a clear area.

1. Click **Start Mission** (terracotta button, top-right of Live page) — badge becomes *Mission Active*
2. Pick the drone up by hand, camera facing down
3. Hover ~30-50 cm above sheet #1 — narrate as the UI reacts:
    - Crack ratio jumps to 5-25 %
    - Red mask paints the cracks pixel-perfect
    - Detection log starts filling, one row per frame above threshold + **real lat/lon/alt** from GPS (since we're outdoors)
    - Crack-ratio chart spikes
4. Translate over to sheet #2, then back
5. Hover over bare ground — ratio drops to ~0 %, no log entries (proves the threshold is real)
6. Click **Stop Mission** — mission moves to History

Then **History page** (30 s):
- Click the just-finished mission
- Show stats: total detections, max coverage %, mean coverage %
- Click **Download CSV** → opens a CSV with timestamped rows + GPS coords
- Click **Download GeoJSON** → opens in any GIS viewer
- Click **Download PDF** → auto-generated report opens

### C) Automated 30s up-and-down flight (3 min)

GPS lock means clean GUIDED flight via the UI buttons — no RC transmitter needed.

Pre-flight call-out (out loud):
- Area clear 5 m around / 5 m above
- All personnel ≥ 10 m back
- Battery ≥ 80 %
- Telemetry green
- **GPS fix** in the UI shows ≥ 3D — confirm before arming
- A1 sheets taped down, drone placed near them

Sequence:
1. Drone on the ground (camera down, near a print so the takeoff frame already shows cracks)
2. Press the Pixhawk **safety switch** (LED solid)
3. UI: **Start Mission** so detections log during the flight
4. **Flight Controls** panel: click **Set GUIDED** → confirm
5. Click **ARM** → 6-item checklist confirmation modal — read it out loud as you confirm
6. Click **Takeoff** with target altitude **2 m**
7. Drone climbs and hovers — 5-10 s
8. Narrate the UI: crack ratio, detection log filling with non-zero altitude rows, telemetry showing the climb
9. Click **Land** — drone descends and disarms automatically
10. UI: **Stop Mission**, safety switch off

The `/command/demo-flight` endpoint can do this whole sequence in one POST (`GUIDED → ARM → Takeoff 1 m → 30 s hover → Land → Disarm`). Available if you want a one-button version, but the manual click-through reads better in a defense.

Then **History** to show the mission has detections logged with **non-zero altitude** rows — proof that inference ran during flight.

### D) Stretch goal — full automated mission over the A1 prints

If C ran clean and you have time:

1. **Plan page**: draw a small polygon enclosing both A1 sheets, altitude **3 m**, overlap **60 %**
2. **Generate Path** — shows the lawnmower scanline grid
3. **Upload Mission** — pushes waypoints to the Pixhawk
4. **Start Mission** in the UI
5. **Flight Controls**: Set GUIDED → ARM → **Auto** mode (Pixhawk follows the uploaded waypoints)
6. Drone flies the lawnmower pattern over the prints autonomously
7. After the last waypoint, send **Land** (or click RTL)
8. After landing, History shows the full waypoint trace + detection log keyed to actual GPS positions

This is the "full system" demo — Plan → Upload → Auto-fly → Detect → Report — all closed-loop. Save it for last and only run it if everything earlier worked.

---

## Q&A — Tech Stack Reference

Quick lookup of what's used for what, organized by layer.

### Aircraft
- **Frame** — Holybro X500 V2 quadcopter
- **Flight controller** — Pixhawk 6C (STM32H753) running **ArduCopter** firmware
- **Telemetry radios** — SiK 433/915 MHz (FTDI USB serial @ 57.6 kbps, MAVLink2)
- **GPS** — u-blox M9N (on the Pixhawk's GPS port)
- **Battery / power module** — Holybro PM02 (powers the entire stack)

### Compute
- **Onboard computer** — NVIDIA Jetson Orin Nano 8 GB Super (1024 CUDA cores SM 8.7, MAXN_SUPER profile = 1020 MHz GPU, 1728 MHz CPU)
- **Camera** — Arducam IMX477 (12 MP, 1/2.3" Sony sensor) via Jetson CSI port
- **JetPack** — 6.2.2 (L4T R36.4.7 + CUDA 12.5 + cuDNN 9.3 + TensorRT 10.3)
- **Storage** — NVMe SSD on the Jetson carrier

### Camera capture pipeline
- **Driver** — `nvargus-daemon` (NVIDIA's hardware ISP for IMX477)
- **GStreamer pipeline** — `nvarguscamerasrc → nvvidconv → appsink`, 30 FPS, AE-driven shutter cap 8 ms (kills motion blur), TNR off, EE off
- **Frame format** — 2160×2160 center-crop from 3840×2160 sensor, downsized to 512×512 for the model, prediction upscaled to 1440×1440 for streaming
- **Stream encoding** — lossless PNG (JPEG chroma subsampling smudged the red overlay)

### Model
- **Architecture** — EfficientCrackNet — encoder-decoder with MobileViT transformer blocks for feature mixing + depthwise-separable convolutions
- **Loss** — alpha-scheduled (BCE × α + Dice × (1-α)), starts BCE-heavy, plateau-based α reduction
- **Training** — 351 real images (280 train / 71 test), 4-way rotation augmentation, ~80 epochs to plateau
- **Test metrics** (real_6) — F1=0.83, mIoU=0.86
- **Inference** — PyTorch 2.6 + CUDA 12.4, FP16 weights, native SDPA Flash Attention kernels
- **Runtime FPS** — 5.5 FPS in MAXN_SUPER, ~3.3 FPS in 15 W mode

### Onboard services (Python, FastAPI)
- **`inference` (`:8001`)** — runs the model on each camera frame, paints the red overlay, broadcasts the annotated frame as motion-PNG
- **`mavlink` (`:8002`)** — opens `/dev/ttyACM0` via `pymavlink`, polls telemetry (HEARTBEAT, ATTITUDE, GLOBAL_POSITION_INT, SYS_STATUS, GPS_RAW_INT, STATUSTEXT), exposes `/status` REST + `/ws/telemetry` WebSocket + `/command/{arm,disarm,takeoff,land,goto,upload-mission,demo-flight}`
- **`data` (`:8003`)** — mission CRUD: SQLite for missions/detections, exports CSV + GeoJSON + PDF (ReportLab)
- **UI server (`:5173`)** — built React bundle served by Python `http.server`, no Node runtime in production
- **All three** wrapped by **`soilcrack.service`** (systemd unit), launched by `start.sh`. Audible **`play_ready_tone.sh`** sends a `PLAY_TUNE` MAVLink message to the Pixhawk buzzer when all three are healthy.

### Laptop services
- **`laptop_mavlink_relay.py`** — runs on the laptop, opens the SiK radio's COM port (auto-detected), exposes the same WebSocket + REST contract as the Jetson MAVLink service on `localhost:18002`. UI prefers this over the Jetson WiFi path. Self-reconnecting watchdog handles SiK USB unplug/replug.
- **`laptop_autoconnect.ps1`** — elevated PowerShell scheduled task at logon. Detects the Jetson on the local network; if not there, joins the Jetson's `soil-crack-demo` AP, SSH-pushes current laptop SSID + password to the Jetson via `nmcli`, swaps back. Then opens the browser to `http://soilcrack.local:5173`.

### UI
- **Framework** — React 19 + TypeScript + Vite 8
- **Routing** — `react-router-dom` v6
- **State** — local hooks (`useState`, `useEffect`, custom hooks like `useTelemetry`, `useServiceHealth`, `useMissionLogger`)
- **Charts** — Recharts (crack-ratio time series)
- **Map (Plan page)** — Leaflet + `react-leaflet`, OpenStreetMap tiles
- **Camera streaming** — motion-PNG (browser auto-refreshes the `<img src=>`)
- **Telemetry / detection streaming** — native WebSocket
- **MAVLink source switching** — runtime probe of `localhost:18002/status` (relay) vs `soilcrack.local:8002/status` (Jetson WiFi); flips on WS disconnect
- **Build** — `npm run build` → static `dist/` served by Python `http.server` on the Jetson

### Networking / OS
- **Jetson WiFi** — `NetworkManager` (`nmcli`); `Hotspot` connection profile is the configured fallback (`soil-crack-demo`, WPA2-PSK, 10.42.0.1)
- **mDNS** — `avahi-daemon` advertises `soilcrack.local` so the laptop reaches the Jetson without knowing the IP
- **Jetson autostart** — `soilcrack.service` (services), `maxn-boot.service` (clocks), both `enabled` at multi-user.target
- **Laptop OS** — Windows 11 + WSL2 (Ubuntu 22.04). WSL kept warm via `WSLBoot` task (`wsl.exe --exec sleep infinity`) + `vmIdleTimeout=-1` in `.wslconfig`
- **Watcher autostart** — Scheduled Task `SoilCrackAutoConnect`, `ONLOGON`, `RunLevel=HighestAvailable`, `DisallowStartIfOnBatteries=false`
- **Laptop ↔ Jetson SSH** — Windows-native OpenSSH, ed25519 key at `%USERPROFILE%\.ssh\jetson_nano`
- **Internet at the venue** — iPhone USB-tether to the laptop (`Apple Mobile Device Ethernet`); laptop's WiFi is then free to join the Jetson AP without losing internet

### Tooling / dev
- **Repo layout** — Python package `crack_detection/` (model + training), shell scripts under `jetson/`, React app under `jetson/ui/`
- **Training** — PyTorch + torchvision, AMP enabled (`grad_accum_steps=8`)
- **Cross-machine dev access** — Tailscale on the dev machine + demo laptop (not used during the demo itself, just for prep)
- **Version control** — git, `jetson-integration` branch on GitHub
- **Image label tool** — semi-supervised labelling pipeline + manual review UI (custom React tool inside `jetson/ui` under the Plan/Mask Review pages)

---

## What NOT to do

- **Do not arm without checking GPS fix is ≥ 3D** in the telemetry panel — GUIDED needs it.
- **Do not unplug the Jetson mid-demo.**
- **Do not let the IMX477 lens contact any surface** — fragile.
- **Watch for sun glare** on the camera over the A1 prints — auto-exposure handles most cases but direct sun reflection can wash out cracks. Reposition prints if needed.
- **Tape the A1 sheets down** — outdoor wind will move them mid-flight otherwise.

---

## Recovery (live, mid-demo)

| Symptom | Fix |
|---|---|
| All service dots red, no camera feed | Hard-refresh browser (Ctrl+Shift+R). If still red → restart services (below). |
| Crack ratio stuck at 0 % over a clearly cracked paper | Move 10-20 cm closer/farther — IMX477 has fixed focus. Best ~30-50 cm. Avoid glare. |
| Camera feed frozen | `ssh jetson 'pkill -f "inference.main"'` then restart inference (below). |
| Telemetry panel zeros and `MAVLink·wifi` chip | Pixhawk USB cable seated? `ssh jetson 'ls /dev/ttyACM*'` should show ACM0. |
| `MAVLink·wifi` instead of `MAVLink·radio` | Radio USB unplugged or relay died. Re-plug; relay watchdog respawns within 30 s. |
| UI white screen | Hard refresh; if still broken, rebuild UI (below). |
| "PreArm: Hardware safety switch" | **Expected.** Safety switch unpressed. Press when ready to arm. |
| "Arm: Yaw (RC4) is not neutral" | RC transmitter yaw stick not centered, or trim off, or transmitter off. |
| Mission won't start | `curl http://soilcrack.local:8003/missions/start -X POST` — if this fails, restart data service. |

### Restart all services
```bash
ssh jetson 'sudo systemctl restart soilcrack.service'
sleep 12 && curl http://soilcrack.local:8001/status
```

### Restart only inference (camera frozen)
```bash
ssh jetson 'pkill -f "inference.main"; sleep 4; cd ~/soil-crack-detection && setsid python3 -m uvicorn jetson.services.inference.main:app --host 0.0.0.0 --port 8001 --log-level warning > /tmp/inf.log 2>&1 < /dev/null & disown'
```

### Switching models live
```bash
# real_6 (higher F1 but sparser on the live feed) with lowered threshold
ssh jetson 'pkill -f "inference.main"; sleep 4; cd ~/soil-crack-detection && \
  MODEL_PATH=results/saved_models/EfficientCrackNet/best_model_num_real_6.pt \
  CRACK_THRESHOLD=0.30 \
  setsid python3 -m uvicorn jetson.services.inference.main:app --host 0.0.0.0 --port 8001 --log-level warning > /tmp/inf.log 2>&1 < /dev/null & disown'

# back to real_4 default
ssh jetson 'sudo systemctl restart soilcrack.service'
```

---

## File reference (where things live)

| Path | Purpose |
|---|---|
| `~/soil-crack-detection` | Repo root (Jetson + laptop) |
| `jetson/start.sh` | Launches all 3 services + UI |
| `jetson/services/inference/` | Camera + model + PNG stream |
| `jetson/services/mavlink/` | Pixhawk telemetry + flight commands |
| `jetson/services/data/` | Mission CRUD + CSV/GeoJSON/PDF export |
| `jetson/laptop_mavlink_relay.py` | Laptop-side SiK radio relay |
| `jetson/laptop_autoconnect.ps1` | Watcher (Windows) |
| `jetson/ui/dist/` | Built React app served on `:5173` |
| `jetson/data/missions/` | Recorded mission folders (CSV + masks + meta) |
| `results/saved_models/EfficientCrackNet/best_model_num_real_4.pt` | Active model checkpoint |
| `/etc/systemd/system/maxn-boot.service` | MAXN_SUPER lock at boot |
| `/etc/systemd/system/soilcrack.service` | Pipeline autostart at boot |
| `/tmp/soilcrack.log` / `/tmp/ready-tone.log` | Service stdout |
