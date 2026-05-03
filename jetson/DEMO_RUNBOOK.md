# Senior Design Defense — Demo Runbook

Two-part demo: **(1)** indoor non-flight crack detection over printed paper, **(2)** short tethered up-and-down flight to prove flight + detection run together.

---

## System State (final demo config — verified 2026-05-02)

| Component | State |
|---|---|
| Jetson git | `bd86cb6` + local fixes (see below) |
| **Model** | `best_model_num_real_4.pt` (real_6 available — Switching Models below) |
| **Detection threshold** | 0.5 (real_4 default) |
| **Sensor mode** | 1 → 3840×2160 @ 30 FPS, center-cropped to 2160×2160 |
| **Stream resolution** | 1440×1440 lossless PNG (~1.9 MB/frame) |
| **Shutter cap** | 8 ms — AE max; floor stays at sensor min (13 µs) so AE can fully dim under bright light |
| **WB** | auto |
| **TNR / EE / Python sharpening** | all off |
| **Live FPS** | ~3.3 |
| Inference (`:8001`) | `/status` returns running |
| MAVLink (`:8002`) | Connected to Pixhawk on `/dev/ttyACM0` @ 921600 |
| Data (`:8003`) | Mission CRUD verified end-to-end |
| UI (`:5173`) | Host-relative API URLs |
| Camera | Arducam IMX477 via nvargus |

A clean `bash jetson/start.sh` reboot uses these settings — defaults in `main.py` were updated to `SENSOR_MODE=1` and `EXPOSURE_MAX_MS=8`. No env vars needed.

### TRT acceleration attempt — failed, do not retry pre-demo

We tried compiling real_6 to a TensorRT FP16 engine to get >3 FPS. Build failed with `Error Code 10: Could not find any implementation for node /MobileViTBlock3/transformer/.../Slice_3_output_0`. The MobileViT attention block has a constant-slice pattern that TRT 10 can't legalize, even after disabling flash-attention paths during ONNX export. Fixing this would mean rewriting MobileViT or running an ONNX simplification pass — out of scope before demo. **System runs PyTorch FP16; ~3.3 FPS is the ceiling for now.**

### Fixes baked into source (so a clean `start.sh` reboot Just Works)
- `jetson/ui/src/api/config.ts` — host-relative service URLs (was hard-coded `localhost`, broke from any laptop)
- `jetson/services/data/main.py` — added `/status` endpoint (was missing → red dot in UI)
- `jetson/services/inference/streamer.py` — (1) fixed overlay-blend bug: `cv2.addWeighted(frame, 0.55, zeros_with_mask, 0.45, 0)` multiplied *every* pixel by 0.55, darkening the whole frame by 45% — read as "blue/desaturated". Now blends only masked pixels. (2) Switched MJPEG → motion-PNG: at 512×512 the JPEG chroma subsampling smudged the red overlay edges; PNG is lossless, ~470 kB/frame at 3.7 FPS.
- `jetson/services/inference/model.py` — (1) `DEFAULT_PT_PATH` is now real_4 (real_6 has higher held-out F1 but under-predicts on the live feed at demo distance — sparse on-screen detections; real_4 visually matches what a human marks). (2) `CRACK_THRESHOLD` is now an env var (default 0.5) so a conservative checkpoint can be made more aggressive without retraining.
- `jetson/services/inference/camera.py` + `main.py` — default sensor mode switched **0 → 2** (1920×1080 @ 60 FPS instead of 4032×3040 @ 21 FPS). 60 FPS lets the auto-exposure pick a much shorter shutter time (~16 ms cap vs ~47 ms cap), removing motion blur on hand-held demo work. Crop bounds are now computed dynamically as `min(w,h)` center-square. **Pipeline outputs full-res cropped frames (1080×1080)** instead of resizing to 512×512 inside GStreamer.
- `jetson/services/inference/main.py` — inference loop now: (1) grabs full-res 1080×1080 frame, (2) applies a mild unsharp mask (`SHARP_AMOUNT=0.30` env var; 0 disables) to preserve thin crack edges through the resize, (3) downsizes to 512×512 for the model, (4) upscales the predicted mask back to 1080×1080 (nearest-neighbour for sharp pixel edges), (5) overlays on the unsharpened full-res frame. The MJPEG-style stream is now 1080×1080 lossless PNG (~1.9 MB/frame at 3.3 FPS ≈ 6 MB/s, comfortably within WiFi).
- `jetson/services/inference/camera.py` — `tnr-mode=0 ee-mode=0`, `interpolation-method=5` (TNR at full strength was smearing detail; wbmode left at 1/auto — verified against gst-launch raw captures, auto WB is correct under our lighting once the overlay bug is fixed)
- `jetson/ui/src/hooks/useMissionLogger.ts` — detection-log threshold **0.1%** (was 4%; see Detection Threshold below)
- `jetson/ui/dist/` — rebuilt with all UI fixes

### Detection threshold (evidence-based)

We previously logged detections only when crack-coverage ≥ **4%**. The held-out test set (N=65 real images) has:

| Distribution | min | p05 | p25 | median | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| GT crack ratio % | 0.03 | 0.09 | 0.14 | 0.24 | 0.34 | 0.65 | 0.96 | 0.28 |
| real_6 predicted % | 0.04 | 0.09 | 0.15 | 0.24 | 0.35 | 0.57 | 0.65 | 0.26 |

**At 4% threshold: 0/64 true-cracked frames retained (0% recall).** Cracks are thin hairline structures — they cover well under 1% of pixels even when visually obvious.

Recall at lower thresholds (true-cracked = GT ≥ 0.05%):
- T=0.05% → 100% recall
- **T=0.10% → 94% recall, 0 false positives** ← new default
- T=0.15% → 77% recall
- T=0.30% → 36% recall

Threshold = **0.10%** chosen as the highest value that still retains ≥ 90% of true-cracked frames in the test set. It's the answer to "the minimum predicted crack-coverage that real cracks reliably exceed".

---

## Prep (do this 30 min before)

### Print
- 3–6 A4 pages, **color, photo-quality**, of the highest-contrast test images:
  - `data/test/images/IMG007.png` (best, F1=0.900)
  - `data/test/images/IMG062.png` (sparse cracks — edge case demo)
  - 3–4 more from `data/test/images/` showing varied crack patterns
- Dim, even lighting helps. Avoid glare from overheads.

### Hardware
- LiPo charged > 90%
- Spare LiPo if you have one
- USB-C wall adapter for the Jetson (so the LiPo isn't drained by services during the talk)
- For the flight: Pixhawk **safety switch held until ready to arm** (LED solid red)
- 2× spare propellers, just in case
- Long table or floor space ~3 m × 3 m for the paper grid

### Network
- Laptop on the same WiFi as the Jetson (`192.168.1.x`)
- Jetson IP today: **`192.168.1.233`** — verify with `ssh jetson 'hostname -I'`

---

## Stable URL on any network: `http://soilcrack.local:5173`

Jetson hostname is `soilcrack` and avahi/mDNS broadcasts it. Whichever WiFi network the Jetson and the laptop share, the URL never changes. Browser handles `.local` natively (Bonjour on macOS/Windows, libnss-mdns on Linux).

## Auto-start on boot

`soilcrack.service` is enabled — power on the Jetson and all 4 services come up by themselves. Once everything is healthy, `play_ready_tone.sh` POSTs to mavlink `/command/play-tone` and the **Pixhawk buzzer plays a short rising arpeggio** as the audible "ready, browser will work now" signal.

Service status / logs:
```bash
ssh jetson 'systemctl status soilcrack.service'
ssh jetson 'tail -50 /tmp/soilcrack.log'      # captured stdout/stderr
ssh jetson 'tail /tmp/ready-tone.log'         # tone-script output
```

## Connecting Jetson to a new WiFi (e.g. demo location)

Hotspot is configured as a low-priority fallback. So when you arrive at a new location:

1. Power on Jetson — it tries known WiFi profiles first, fails, falls back to its own hotspot
2. Laptop joins SSID `soil-crack-demo` (password `cracksoil2026`)
3. Add the demo location's WiFi to the Jetson:
   ```bash
   SSH_HOST=10.42.0.1 bash jetson/connect_wifi.sh "DemoSSID" "demopassword"
   ```
4. Jetson reconnects to that WiFi; laptop joins the same WiFi
5. Browse `http://soilcrack.local:5173`

Once added, that network is saved on the Jetson and auto-connects on every future boot.

## Network: no external WiFi needed (Jetson is its own AP)

The Jetson can broadcast its own WiFi network so the laptop reaches it without depending on any infrastructure. This makes the demo portable and removes the "what's the WiFi here?" failure mode.

**One-time setup, on the Jetson:**

```bash
ssh jetson 'cd ~/soil-crack-detection && bash jetson/setup_hotspot.sh'
```

After that, the Jetson:
- Broadcasts SSID `soil-crack-demo` (password `cracksoil2026`)
- Has IP `10.42.0.1` on that network
- Auto-starts the hotspot on every boot
- Also reachable via `<hostname>.local` (mDNS / avahi)

**On demo day:**

1. Power the Jetson on (it auto-starts the hotspot)
2. On the laptop, connect to WiFi `soil-crack-demo`
3. Run the startup script:

```bash
SSH_HOST=10.42.0.1 bash jetson/demo_start.sh
```

The script prints the URL — open it in your browser. Works in any room, with or without infrastructure WiFi.

**Reverting to a regular WiFi** (e.g. for development with internet access):

```bash
ssh jetson '
  sudo nmcli connection down Hotspot
  sudo nmcli connection modify Hotspot connection.autoconnect no
  sudo nmcli connection up "your-home-wifi-name"
'
```

To put it back into hotspot mode:
```bash
ssh jetson '
  sudo nmcli connection modify Hotspot connection.autoconnect yes
  sudo nmcli connection up Hotspot
'
```

**Trade-off:** in hotspot mode the Jetson has no internet. `git pull` on the Jetson won't work. So pull any last-minute fixes *before* switching to hotspot mode.

## One-shot startup from a fresh laptop session

If you open WSL on your laptop fresh and just want everything running:

```bash
cd ~/path/to/soil-crack-detection      # wherever your local clone lives
bash jetson/demo_start.sh
```

The script:
1. Verifies `ssh jetson` works
2. Kills any leftover services
3. Restarts nvargus-daemon (clears any stuck camera state)
4. Launches all 4 services via `start.sh`
5. Polls until `/status` is healthy on all of them
6. Prints the Jetson IP and the URL to open in your browser

If any step fails, the script tells you what's wrong + dumps the last 25 lines of the services log. Should take ~30 seconds total when everything's healthy.

Prerequisites on the laptop (one-time setup):
- WSL or any Linux shell
- SSH key to `jetson` set up so `ssh jetson 'echo up'` works without a password prompt
- This repo cloned (any path)

## Pre-Demo Checks (5 min before walking on stage)

```bash
# From your laptop
ssh jetson 'pgrep -af uvicorn | wc -l'             # expect 3
curl http://192.168.1.233:8001/status               # running:true, fps>3
curl http://192.168.1.233:8002/status               # connected:true
curl http://192.168.1.233:8003/missions | head -c 100
```

Open the UI: **`http://192.168.1.233:5173`**

You should see:
- **Camera feed live** (whatever the drone is pointing at)
- **3 green dots** in the status bar (inference / mavlink / data)
- Telemetry: battery %, mode, armed=false
- Crack ratio: ~0% (idle, not over a cracked surface)

If all 3 service dots are red and there's no feed → see **Recovery** below.

---

## Demo Part 1 — Non-Flight (5 min)

### 1) Open with architecture (30 s)
Point at the drone:
> "Pixhawk on the bottom. Jetson Orin Nano on top. IMX477 camera nadir. Three FastAPI services on the Jetson; a React UI on this laptop. Same WiFi, no cloud."

### 2) Show idle Live page (30 s)
- Camera feed updating ~3.5 FPS
- Telemetry panel
- Detection log empty
- Status bar: 3 green dots

### 3) Plan a mission (60 s) — go to **Plan** page
- Draw a polygon on the map
- Set altitude (10 m), overlap (60%)
- Show the auto-generated lawnmower scanline grid
- > "Outdoors this would upload to the Pixhawk. We'll do that in part two."

### 4) Run the indoor detection demo (3 min) — back to **Live** page
- Click **Start Mission** (terracotta button, top right)
  - Badge becomes "Mission Active" with the mission ID
- Pick up the drone with two hands, **camera facing down**
- Hover ~30–50 cm above **crack paper #1**
  - Crack ratio jumps to 5–25%
  - Mask overlay shows red crack pixels
  - Detection log row appears
  - Crack ratio chart updates
- Move to paper #2, #3 — narrate
- Hover over a blank surface — ratio drops to ~0%, no log entry (≥4% threshold)
- Click **Stop Mission**
  - Badge clears, mission moves to History

### 5) Show outputs (60 s) — go to **History** page
- Click your just-finished mission
- Show: total detections, max coverage %, mean coverage %
- Download **CSV** — show the row(s)
- Download **GeoJSON** — show it parses
- Download **PDF report** — show the auto-generated report

---

## Demo Part 2 — Tethered Up-and-Down Flight (3 min)

**Goal: prove flight + detection happen at the same time.**

### Pre-flight checklist (out loud, before arming)
1. Area clear of obstacles 3 m in every direction and 3 m overhead
2. All personnel ≥ 5 m back
3. Battery ≥ 80% (check telemetry panel)
4. Telemetry connected (green MAVLink dot)
5. GPS — **for indoor demo this will fail; that's expected.** State this: "Indoors GPS won't acquire, so we'll fly in ALT_HOLD/STABILIZE rather than GUIDED for the up-and-down."

### Flight sequence

If indoors with no GPS, do a **manual** flight:
1. **Place drone on the ground**, camera facing down at one of the printed crack papers
2. **Press the safety switch** on the Pixhawk (LED goes solid)
3. In the UI: leave mode as `ALT_HOLD` (or `STABILIZE`)
4. **Click Start Mission** in the UI (so detections log during the flight)
5. **Arm via RC transmitter** (lower-left stick to bottom-right corner) — *not* via UI
6. **Throttle up gently** to ~1.5 m altitude — hover for 5–10 seconds
7. **Hold position** — narrate the live UI: "Crack ratio updating, detections logging with timestamps, telemetry showing altitude rising/holding"
8. **Throttle down gently** — land
9. **Disarm via RC** (throttle low + yaw left held for 3 s)
10. **Click Stop Mission** in the UI
11. **Press safety switch off**

If outdoors with GPS lock:
- Same as above, OR use the **Set GUIDED → ARM → Takeoff 2.0 m → Land** flow from the UI's Flight Controls panel (each button has a confirmation modal)
- ARM modal has a 6-item checklist — read the checklist out loud as you confirm

### What you're showing
- Crack ratio updating live during flight
- Detection log filling with **non-zero altitude** rows (telemetry from the actual flight, not stationary)
- Status bar still all-green after landing
- After landing: open the mission in History, show that detections were captured *while the drone was in the air*

### Talking points
> "Same pipeline as the indoor demo — Start Mission, fly, Stop Mission. The Jetson is doing inference at 3–5 FPS the entire time. Each detection row carries the live altitude and heading from the Pixhawk."

---

## What NOT to do

- **Do not arm via the UI indoors unless GPS has a lock** — the UI's ARM defaults to GUIDED which needs GPS.
- **Do not use Takeoff/Land buttons indoors** — those need GUIDED + GPS.
- **Do not leave services running on LiPo for 30+ min idle** — drains the battery while you're talking.
- **Do not unplug the Jetson mid-demo.**
- **Do not let the IMX477 lens contact any surface** — fragile.

---

## Recovery (live, mid-demo)

| Symptom | Fix |
|---|---|
| All 3 service dots red, no camera feed | Hard-refresh browser (Ctrl+Shift+R). If still red, restart services (below). |
| Crack ratio stuck at 0% over a clearly cracked paper | Move 10–20 cm closer/farther — IMX477 has fixed focus. Best ~30–50 cm. Adjust lighting (no glare). |
| Camera feed frozen | `ssh jetson 'pkill -f "inference.main"'` then restart inference (below). |
| Telemetry panel all zeros | Pixhawk USB cable seated? `ssh jetson 'ls /dev/ttyACM*'` should show ACM0. |
| UI white screen | Hard refresh Ctrl+Shift+R; if still broken, rebuild UI (below). |
| "PreArm: Hardware safety switch" warning | **Expected.** It means the Pixhawk safety switch is unpressed, blocking arm. Press it when you're ready to arm. |
| Mission won't start | `curl http://192.168.1.233:8003/missions/start -X POST` — if this fails, restart data service. |

### Restart all services
```bash
ssh jetson 'pkill -f uvicorn; pkill -f "http.server"; sleep 2; cd ~/soil-crack-detection && nohup bash jetson/start.sh > /tmp/services.log 2>&1 & disown'
sleep 12
curl http://192.168.1.233:8001/status
```

### Restart only the inference service (camera frozen)
```bash
ssh jetson 'pkill -f "inference.main"; sleep 4; cd ~/soil-crack-detection && setsid python3 -m uvicorn jetson.services.inference.main:app --host 0.0.0.0 --port 8001 --log-level warning > /tmp/inf.log 2>&1 < /dev/null & disown'
```

### Switching models live

```bash
# real_6 (more conservative, higher test F1) with lowered threshold so it
# still produces visible detections — the env vars take effect on restart.
ssh jetson 'pkill -f "inference.main"; sleep 4; cd ~/soil-crack-detection && \
  MODEL_PATH=results/saved_models/EfficientCrackNet/best_model_num_real_6.pt \
  CRACK_THRESHOLD=0.30 \
  setsid python3 -m uvicorn jetson.services.inference.main:app --host 0.0.0.0 --port 8001 --log-level warning > /tmp/inf.log 2>&1 < /dev/null & disown'

# back to real_4 default
ssh jetson 'pkill -f "inference.main"; sleep 4; cd ~/soil-crack-detection && setsid python3 -m uvicorn jetson.services.inference.main:app --host 0.0.0.0 --port 8001 --log-level warning > /tmp/inf.log 2>&1 < /dev/null & disown'
```

### Rebuild UI (only if a code change was made)
```bash
ssh jetson 'export PATH=$HOME/.nvm/versions/node/v20.20.2/bin:$PATH && cd ~/soil-crack-detection/jetson/ui && npm run build'
```
Then hard-refresh browser.

### Force the Jetson IP (if WiFi reassigned)
```bash
ssh jetson 'hostname -I'
# Then update the URL you opened in the browser
```

---

## After the demo

Stop services to save battery:
```bash
ssh jetson 'pkill -f uvicorn; pkill -f "http.server"'
```

Pull the mission CSVs/PDFs back to your laptop if you want them for the final report:
```bash
scp -r jetson:soil-crack-detection/jetson/data/missions/ ~/Downloads/demo_missions/
```

---

## File reference (where things live on the Jetson)

| Path | Purpose |
|---|---|
| `~/soil-crack-detection` | Repo root |
| `jetson/start.sh` | Launches all 3 services + UI |
| `jetson/services/inference/` | Camera + model + MJPEG/WebSocket |
| `jetson/services/mavlink/` | Pixhawk telemetry + flight commands |
| `jetson/services/data/` | Mission CRUD + CSV/GeoJSON/PDF export |
| `jetson/ui/dist/` | Built React app served on `:5173` |
| `jetson/data/missions/` | Recorded mission folders (CSV + masks + meta) |
| `results/saved_models/EfficientCrackNet/best_model_num_real_4.pt` | Active model checkpoint |
| `/tmp/services.log` | Most recent service stdout/stderr |
