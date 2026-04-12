# Implementation Plan — SDP Semester 2
**Deadline:** April 12 | **Today:** April 8 | **Time left:** 4 days

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         JETSON ORIN NANO                           │
│                                                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ inference/      │  │ mavlink/        │  │ data/            │   │
│  │ port 8001       │  │ port 8002       │  │ port 8003        │   │
│  │                 │  │                 │  │                  │   │
│  │ camera.py       │  │ connection.py   │  │ recorder.py      │   │
│  │ model.py        │  │ telemetry.py    │  │ exporter.py      │   │
│  │ streamer.py     │  │ flight.py       │  │ models.py        │   │
│  │ routes.py       │  │ routes.py       │  │ routes.py        │   │
│  │ main.py         │  │ main.py         │  │ main.py          │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘   │
│           │                    │                     │             │
│           └────────────────────┴─────────────────────┘             │
│                                │ HTTP + WebSocket                  │
│                      ┌─────────▼──────────┐                        │
│                      │ ui/ (React)        │                        │
│                      │ port 5173 (dev)    │                        │
│                      │ or static /dist    │                        │
│                      └────────────────────┘                        │
└────────────────────────────────────────────────────────────────────┘
                                │ WiFi (192.168.1.233)
                      ┌─────────▼──────────┐
                      │  Windows Browser   │
                      └────────────────────┘
```

**Communication:**
- REST (HTTP) for commands, mission data, file downloads
- WebSocket for real-time telemetry and live detections
- MJPEG for camera stream (img src, no JS overhead)
- Services are fully independent — each can be restarted without affecting others

---

## Complete File Structure

```
jetson/
├── services/
│   │
│   ├── inference/
│   │   ├── main.py          # uvicorn entry: app = FastAPI(), CORS, startup event
│   │   ├── camera.py        # GStreamer pipeline builder + FrameGrabber thread class
│   │   ├── model.py         # load_model(), InferenceEngine.run(frame) → (mask, ratio)
│   │   ├── streamer.py      # generate_mjpeg() generator — overlays mask on frame
│   │   └── routes.py        # GET /stream, GET /status, WS /ws/detections
│   │
│   ├── mavlink/
│   │   ├── main.py          # uvicorn entry, startup: open serial, start telemetry loop
│   │   ├── connection.py    # MAVLinkConnection: open/close /dev/ttyACM0:921600
│   │   │                    #   heartbeat sender (1 Hz), reconnect logic
│   │   ├── telemetry.py     # TelemetryPoller: async loop reads MAVLink messages
│   │   │                    #   → updates SharedTelemetry dataclass (GPS, att, bat, mode)
│   │   │                    #   → broadcasts to all WS clients at 5 Hz
│   │   ├── flight.py        # FlightController: set_guided_mode(), arm(), takeoff(alt),
│   │   │                    #   goto_ned(n,e,d), land(), rtl(), disarm()
│   │   │                    #   all methods are async, use pymavlink command_long_send
│   │   └── routes.py        # WS /ws/telemetry, GET /status,
│   │                        # POST /command/{action} (arm/takeoff/goto/land/rtl/disarm)
│   │
│   └── data/
│       ├── main.py          # uvicorn entry, mount /missions static dir
│       ├── recorder.py      # MissionRecorder: start_mission() → mission_id (timestamp)
│       │                    #   log_detection(mission_id, lat, lon, alt, ratio, mask_b64)
│       │                    #   stop_mission(mission_id) → triggers export
│       │                    #   writes detections.csv row-by-row during flight
│       ├── exporter.py      # post_process(mission_dir):
│       │                    #   1. csv → GeoJSON FeatureCollection (Point per detection)
│       │                    #   2. generate PDF report via reportlab
│       │                    #   3. write mission_meta.json (stats summary)
│       ├── models.py        # Pydantic: Mission, Detection, TelemetrySnapshot, MissionMeta
│       └── routes.py        # POST /missions/start → {mission_id}
│                            # POST /missions/{id}/stop
│                            # POST /missions/{id}/detect → log one detection
│                            # GET  /missions → list [{id, date, detections, status}]
│                            # GET  /missions/{id} → full mission detail + stats
│                            # GET  /missions/{id}/geojson → detections.geojson download
│                            # GET  /missions/{id}/csv → detections.csv download
│                            # GET  /missions/{id}/report → report.pdf download
│
├── ui/
│   ├── src/
│   │   ├── api/
│   │   │   ├── inference.ts      # getStatus(), streamUrl (const)
│   │   │   ├── mavlink.ts        # getStatus(), postCommand(action, body?)
│   │   │   └── data.ts           # getMissions(), getMission(id), getGeojson(id), etc.
│   │   │
│   │   ├── hooks/
│   │   │   ├── useTelemetry.ts   # WS hook → mavlink :8002/ws/telemetry
│   │   │   │                     #   returns: {gps, altitude, roll, pitch, yaw,
│   │   │   │                     #            battery, mode, armed, connected}
│   │   │   ├── useDetections.ts  # WS hook → inference :8001/ws/detections
│   │   │   │                     #   returns: {crack_ratio, timestamp, history[]}
│   │   │   └── useMissions.ts    # REST polling hook → data :8003/missions
│   │   │                         #   returns: {missions[], loading, refetch}
│   │   │
│   │   ├── components/
│   │   │   ├── CameraFeed.tsx        # <img src="http://jetson:8001/stream" />
│   │   │   │                         #   with crack ratio badge overlay on top-left
│   │   │   │                         #   aspect-ratio locked, rounded corners
│   │   │   │
│   │   │   ├── TelemetryPanel.tsx    # Live stat grid: altitude, roll/pitch, battery bar,
│   │   │   │                         #   GPS coords, flight mode badge, armed indicator
│   │   │   │
│   │   │   ├── CrackLog.tsx          # Scrolling table of real-time detections
│   │   │   │                         #   columns: time, N/E position, coverage %
│   │   │   │                         #   row highlight animates in on new entry
│   │   │   │
│   │   │   ├── MissionMap.tsx        # react-leaflet MapContainer
│   │   │   │                         #   OSM tiles, crack markers as CircleMarker
│   │   │   │                         #   radius ∝ crack_ratio, color: terracotta gradient
│   │   │   │                         #   click marker → Popup with stats
│   │   │   │                         #   flight path as Polyline
│   │   │   │
│   │   │   ├── MissionControl.tsx    # Button group: Start Mission / Land / RTL
│   │   │   │                         #   buttons disabled based on armed/mode state
│   │   │   │                         #   confirmation modal before Start
│   │   │   │
│   │   │   ├── CrackRatioChart.tsx   # recharts AreaChart: crack_ratio over time
│   │   │   │                         #   x: flight elapsed seconds, y: coverage %
│   │   │   │                         #   terracotta fill, animated
│   │   │   │
│   │   │   ├── SummaryCards.tsx      # 4-up stat cards:
│   │   │   │                         #   Total Detections / Max Coverage /
│   │   │   │                         #   Flight Duration / Frames Processed
│   │   │   │
│   │   │   ├── ExportButtons.tsx     # GeoJSON / CSV / PDF download buttons
│   │   │   │                         #   → GET /missions/{id}/{format}
│   │   │   │
│   │   │   └── StatusBar.tsx         # Top bar: service health indicators
│   │   │                             #   green/red dots for inference/mavlink/data
│   │   │
│   │   ├── pages/
│   │   │   ├── LiveMission.tsx       # Route: /live
│   │   │   │   Layout:
│   │   │   │   ┌──────────────────────────────────┐
│   │   │   │   │ StatusBar                        │
│   │   │   │   ├──────────────┬───────────────────┤
│   │   │   │   │ CameraFeed   │ TelemetryPanel    │
│   │   │   │   │ (live stream)│ (WS telemetry)    │
│   │   │   │   │              ├───────────────────┤
│   │   │   │   │              │ MissionControl    │
│   │   │   │   ├──────────────┴───────────────────┤
│   │   │   │   │ CrackLog (scrolling live table)  │
│   │   │   │   └──────────────────────────────────┘
│   │   │   │
│   │   │   ├── PostFlight.tsx        # Route: /missions/:id
│   │   │   │   Layout:
│   │   │   │   ┌──────────────────────────────────┐
│   │   │   │   │ SummaryCards (4 stats)           │
│   │   │   │   ├──────────────────────────────────┤
│   │   │   │   │ Tabs: [Map] [Analysis] [Raw Data]│
│   │   │   │   │                                  │
│   │   │   │   │ Map tab: MissionMap (Leaflet)     │
│   │   │   │   │   + ExportButtons (GeoJSON/CSV/PDF│
│   │   │   │   │                                  │
│   │   │   │   │ Analysis tab: CrackRatioChart    │
│   │   │   │   │   + detection stats table        │
│   │   │   │   │                                  │
│   │   │   │   │ Raw Data tab: full detections    │
│   │   │   │   │   table with all columns         │
│   │   │   │   └──────────────────────────────────┘
│   │   │   │
│   │   │   └── History.tsx           # Route: /missions
│   │   │       Grid of mission cards:
│   │   │       date / detection count / max coverage / duration
│   │   │       click → /missions/:id
│   │   │
│   │   ├── App.tsx              # Router setup (react-router-dom)
│   │   │                        # Routes: / → /live, /missions, /missions/:id
│   │   └── main.tsx             # ReactDOM.createRoot, import index.css
│   │
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts           # proxy: /api/inference → :8001, etc.
│   ├── tailwind.config.ts
│   └── tsconfig.json
│
├── scripts/
│   ├── start.sh                 # starts all 3 services via uvicorn in background
│   │                            # serves built UI dist/ via data service static mount
│   └── stop.sh                  # pkill -f uvicorn
│
└── results/
    └── missions/
        └── mission_20260408_143200/
            ├── mission_meta.json
            ├── detections.csv
            ├── detections.geojson
            ├── report.pdf
            └── masks/
                └── frame_042_mask.png
```

---

## Design Direction (frontend-design plugin)

**Aesthetic:** *Field Survey Precision* — warm editorial meets technical instrument.
Think a well-funded agritech startup that hired a real designer.
Not dark. Not terminal. Not purple gradients.

**Concept:** The interface should feel like you're looking at a premium field survey tool —
the kind of thing a UAE precision agriculture company would actually use in the field.
Clean, confident, purposeful. The cracked earth *is* the design language.

**Typography:**
- Display/headings: `Instrument Serif` (Google Fonts) — editorial authority
- Body/data: `DM Sans` — clean, legible at small sizes
- Monospaced data (coords, timestamps): `JetBrains Mono` — crisp precision

**Color palette (CSS variables):**
```css
--bg-base:        #F9F6F1   /* warm parchment — like field survey paper */
--bg-card:        #FFFFFF
--bg-subtle:      #F2EDE6
--border:         #E4DDD4
--text-primary:   #1A1612   /* warm near-black */
--text-secondary: #7A6E65
--accent:         #C4622D   /* terracotta — literally cracked dry soil */
--accent-light:   #F5E6DC
--accent-dark:    #9E4A1F
--status-ok:      #3D7A5A   /* field green */
--status-warn:    #D4932A   /* amber */
--status-error:   #C43D2D   /* red clay */
--map-crack:      #C4622D   /* crack markers: terracotta */
```

**Layout philosophy:**
- Generous white space, not cramped
- Cards with very subtle warm shadow (`0 1px 4px rgba(26,22,18,0.08)`)
- Rounded corners: 8px cards, 6px buttons, 4px badges
- The map gets the most screen real estate on post-flight view
- Status indicators are dot + text, never just color alone
- Animations: subtle fade-in on data load, pulse on live indicators only

**Memorable detail:** The crack ratio badge on the camera feed animates its border
from green → amber → terracotta as coverage increases, using a CSS gradient ring.
The map markers use a radial gradient from deep terracotta center to transparent edge.

---

## Backend Services — Detailed Specs

### Service 1: Inference (port 8001)
```
startup:
  - load model from MODEL_PATH env var (default: results/saved_models/.../best_model_num_real_5.pt)
  - open GStreamer pipeline (sensor_mode=0, identical to live_inference.py)
  - start FrameGrabber thread
  - start inference loop thread

inference loop:
  - get frame from grabber
  - run model.run(frame) → (mask_np, crack_ratio)
  - overlay mask on frame → overlay_frame
  - update SharedState(frame=overlay_frame, ratio=crack_ratio, fps=...)
  - broadcast {crack_ratio, timestamp, fps} to all WS clients
  - sleep to target ~5 Hz broadcast rate (inference runs as fast as GPU allows)

MJPEG stream:
  - encode SharedState.frame as JPEG (quality 80)
  - yield multipart boundary
  - runs in its own generator, one per client connection
```

### Service 2: MAVLink (port 8002)
```
startup:
  - open pymavlink connection: master = mavutil.mavlink_connection('/dev/ttyACM0', baud=921600)
  - wait_heartbeat()
  - start telemetry polling loop (asyncio task)

telemetry polling loop (10 Hz):
  - master.recv_match(blocking=False) in tight loop
  - parse: GLOBAL_POSITION_INT → lat/lon/alt/heading
  - parse: ATTITUDE → roll/pitch/yaw
  - parse: SYS_STATUS → battery voltage
  - parse: HEARTBEAT → mode/armed
  - update SharedTelemetry dataclass
  - every 200ms: broadcast full telemetry snapshot to WS clients

flight commands (all via pymavlink command_long_send):
  set_guided_mode():   MAV_CMD_DO_SET_MODE, mode=GUIDED
  arm():               MAV_CMD_COMPONENT_ARM_DISARM, param1=1
  disarm():            MAV_CMD_COMPONENT_ARM_DISARM, param1=0
  takeoff(alt):        MAV_CMD_NAV_TAKEOFF, param7=alt
  goto_ned(n,e,d):     SET_POSITION_TARGET_LOCAL_NED
  land():              MAV_CMD_NAV_LAND
  rtl():               MAV_CMD_NAV_RETURN_TO_LAUNCH

each command: send → wait for ACK (timeout 5s) → return {ok, message}
```

### Service 3: Data (port 8003)
```
mission lifecycle:
  POST /missions/start:
    - generate mission_id = "mission_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    - mkdir results/missions/{mission_id}/masks/
    - write mission_meta.json stub: {id, start_time, model, status: "active"}
    - open detections.csv for append writing
    - return {mission_id}

  POST /missions/{id}/detect:
    - body: {lat, lon, alt_m, heading_deg, crack_ratio_pct, mask_png_b64 (optional)}
    - append row to detections.csv
    - if mask_png_b64 provided: decode and save to masks/frame_{n}.png

  POST /missions/{id}/stop:
    - close CSV file handle
    - call exporter.post_process(mission_dir) in background task
    - update mission_meta.json: {status: "complete", end_time, ...stats}

exporter.post_process(mission_dir):
  1. read detections.csv
  2. write detections.geojson:
       {type: FeatureCollection, features: [
         {type: Feature,
          geometry: {type: Point, coordinates: [lon, lat]},
          properties: {timestamp, alt_m, crack_ratio_pct, heading_deg}}
       ]}
  3. compute stats: total_detections, max_coverage, mean_coverage,
                    flight_duration_s, bbox (min/max lat/lon)
  4. update mission_meta.json with stats
  5. generate report.pdf via reportlab:
       - cover: mission date, model used, summary stats
       - map placeholder (bbox + detection count — no live tiles)
       - detection table: top 10 highest crack_ratio frames
       - methodology blurb (boilerplate from project description)
```

---

## Frontend Hooks — Detailed

```typescript
// useTelemetry.ts
// Connects to ws://jetson:8002/ws/telemetry
// Returns Telemetry | null, connected: boolean
// Auto-reconnects every 3s on disconnect
// Parses: {lat, lon, alt_m, roll_deg, pitch_deg, yaw_deg,
//          battery_v, battery_pct, mode, armed, gps_fix}

// useDetections.ts
// Connects to ws://jetson:8001/ws/detections
// Returns: {current: DetectionEvent, history: DetectionEvent[]}
// history capped at 200 entries (ring buffer)
// DetectionEvent: {crack_ratio_pct, timestamp_ms, fps}

// useMissions.ts
// Polls GET /missions every 10s
// Returns: {missions: Mission[], loading, error, refetch}
// Provides startMission(), stopMission(id) that POST + refetch
```

---

## Execution Order

```
Day 1 (April 8):
  [x] Plan finalized
  [ ] Step 1: Setup — install Python deps on Jetson, init React project
  [ ] Step 2: inference/ service — full implementation + test MJPEG stream

Day 2 (April 9):
  [ ] Step 3: mavlink/ service — connection, telemetry, flight commands
  [ ] Step 4: data/ service — recorder, exporter, all routes

Day 3 (April 10):
  [ ] Step 5: React UI — hooks + all components
  [ ] Step 6: React UI — all 3 pages wired up
  [ ] Step 7: start.sh + integration test (dry run, no flight)

Day 4 (April 11-12):
  [ ] Step 8: Report sections 5 & 6 rewrite
  [ ] Step 9: Buffer / polish / fix issues
```

---

## Dependencies

**Python (Jetson — install once):**
```bash
pip install pymavlink fastapi uvicorn websockets reportlab pydantic python-multipart
# flask and mavsdk already installed (not used going forward)
```

**Node (dev machine — build UI here, SCP dist/ to Jetson):**
```bash
npm create vite@latest ui -- --template react-ts
cd ui
npm install tailwindcss @tailwindcss/vite
npx shadcn@latest init
npm install react-router-dom react-leaflet leaflet recharts axios
npm install -D @types/leaflet
# fonts: loaded via Google Fonts CDN in index.html
# Instrument Serif, DM Sans, JetBrains Mono
```

**Vite proxy config (avoids CORS in dev):**
```typescript
// vite.config.ts
server: {
  proxy: {
    '/api/inference': { target: 'http://192.168.1.233:8001', rewrite: p => p.replace(/^\/api\/inference/, '') },
    '/api/mavlink':   { target: 'http://192.168.1.233:8002', rewrite: p => p.replace(/^\/api\/mavlink/, '') },
    '/api/data':      { target: 'http://192.168.1.233:8003', rewrite: p => p.replace(/^\/api\/data/, '') },
  }
}
```

---

## Open Questions

1. **GPS indoors?** If no GPS fix, lat/lon = 0.0. Data service falls back to storing
   NED position (north_m, east_m) as coordinates. Map shows relative path, not real map.
2. **Model_6 timing** — inference/model.py reads MODEL_PATH from env var at startup.
   Swap checkpoint by restarting the service. No code change needed.
3. **Jetson internet?** If no internet during demo, swap Leaflet OSM tiles for
   `L.tileLayer('')` with a plain grey background. Map still works, just no satellite imagery.
4. **Build on dev machine** — React built on WSL, `dist/` SCPed to Jetson.
   Data service mounts it as `StaticFiles(directory="ui/dist", html=True)`.
