# Laptop Claude — Demo Day Context

You're the Claude Code instance running on the demo laptop. This file is your briefing. Read it and the linked files before doing anything risky.

## What this project is

UAV-based AI system for detecting cracked soil in UAE farms. AUS senior design, demo'd from a laptop browser. The drone (Holybro X500 V2 + Pixhawk 6C) runs a Jetson Orin Nano onboard with an Arducam IMX477 camera. Three FastAPI microservices (inference / mavlink / data) + a React UI live on the Jetson. The laptop is just a browser pointing at the Jetson.

## Current state (set up by the dev-machine Claude session)

- Jetson is at HEAD of `jetson-integration`
- **Hostname is `soilcrack`** so the stable URL is `http://soilcrack.local:5173` (works from any laptop on the same network as the Jetson, regardless of which WiFi)
- Both model checkpoints on Jetson: `real_4` (deployment default) and `real_6`
- TRT engines built (or being built) per-checkpoint at `results/efficientcracknet_real{4,6}_fp16.trt`. `model.py` auto-loads the matching engine when `MODEL_PATH` is set.
- WiFi hotspot connection profile `Hotspot` exists with **autoconnect-priority 1 (fallback)**. It only activates when no known regular WiFi is reachable.
- Avahi (mDNS) is broadcasting `soilcrack.local`
- **systemd unit `soilcrack.service` is enabled** — services auto-start on boot
- On startup, `play_ready_tone.sh` polls all 4 endpoints; once green, it POSTs to mavlink `/command/play-tone` and the Pixhawk buzzer plays a short rising arpeggio. **That beep means "everything is up, browser will work now."**

## Stable URL (works on any WiFi)

`http://soilcrack.local:5173` — once both Jetson and laptop are on the same network, the browser resolves the `.local` name via mDNS. No IP-tracking needed.

If the user ever reports `soilcrack.local` won't load:
- Both devices on same network? `ssh soilcrack.local 'hostname -I'`
- mDNS still up? `ssh jetson 'systemctl is-active avahi-daemon'`
- Some corporate networks block multicast — fall back to the IP (`ssh jetson 'hostname -I'`).

## Connecting the Jetson to a new WiFi (e.g. demo location)

```bash
bash jetson/connect_wifi.sh "DemoNetworkSSID" "demopassword"
```

If the Jetson isn't reachable from the laptop yet (because no shared network), the workflow is:

1. Power on Jetson at the new location
2. Jetson's known WiFi profiles all fail → Hotspot fallback activates (priority 1)
3. Laptop joins SSID `soil-crack-demo` (password `cracksoil2026`)
4. Run `SSH_HOST=10.42.0.1 bash jetson/connect_wifi.sh "DemoNet" "demopassword"`
5. Jetson reconnects to demo WiFi; laptop joins the same WiFi
6. Browse `http://soilcrack.local:5173`

## The scripts

### `bash jetson/demo_start.sh`
The full startup. Auto-detects whether the Jetson is reachable as `jetson` (regular WiFi), `10.42.0.1` (hotspot), or `ubuntu.local` (mDNS). Then:
1. Verifies model files exist
2. Kills leftover services
3. Restarts `nvargus-daemon` (camera)
4. Launches `start.sh`
5. Polls `/status` on inference/mavlink/data/UI
6. Prints the URL for the browser

Should take ~30 s when healthy. Exit code 0 = ready. Non-zero = it tells you what failed.

### `bash jetson/enable_hotspot.sh`
One-shot WiFi switch. The Jetson stops connecting to the regular WiFi and starts broadcasting `soil-crack-demo` (password `cracksoil2026`). The laptop loses its SSH connection — you have to switch the laptop's WiFi to the new SSID and then everything is at `10.42.0.1`. There's a matching `disable_hotspot.sh` to revert (`WIFI=YourHomeNetwork bash jetson/disable_hotspot.sh`).

## When the user says "set up the demo"

If the Jetson is on regular WiFi and they want to demo from anywhere (no infrastructure WiFi):
1. `bash jetson/enable_hotspot.sh`
2. Tell user: "switch laptop WiFi to soil-crack-demo (password cracksoil2026)"
3. After they confirm, run `SSH_HOST=10.42.0.1 bash jetson/demo_start.sh`
4. Tell them the URL it prints (will be `http://10.42.0.1:5173`)

If they want to demo on the regular WiFi (e.g. at home):
1. `bash jetson/demo_start.sh`
2. Tell them the URL it prints

## Common things they'll ask + how to handle

| Ask | Action |
|---|---|
| "set everything up" | Run `demo_start.sh`. If it fails, look at the script's diagnostic output. |
| "switch to hotspot" / "make it portable" | `enable_hotspot.sh` then walk them through laptop WiFi switch. |
| "switch back to normal WiFi" | `WIFI=<name> bash jetson/disable_hotspot.sh` then walk laptop back. |
| "the camera's frozen" | `ssh jetson 'sudo fuser -k 8001/tcp; sleep 3; sudo systemctl restart nvargus-daemon'` then re-launch inference (see DEMO_RUNBOOK.md → Recovery section). |
| "I want a different model" | Edit env on next launch: `MODEL_PATH=results/saved_models/EfficientCrackNet/best_model_num_real_6.pt CRACK_THRESHOLD=0.001 ...` See model-switch examples in DEMO_RUNBOOK.md. |
| "stop everything" | `ssh jetson 'pkill -f uvicorn; pkill -f "http.server"'` |
| "Pixhawk shows battery 0%" | Expected if LiPo is disconnected. Pixhawk is being USB-powered from Jetson during bench tests. |
| "data services dot is red" | Should be fixed in source (`/status` endpoint added). If still red after `demo_start.sh`, hard-refresh browser (Ctrl+Shift+R). |

## Things that WILL bite you

- **`pkill -f "inference.main"` orphans the camera.** Always restart `nvargus-daemon` after killing inference, or the next launch will fail with "Failed to create CaptureSession". `demo_start.sh` already does this.
- **Hotspot mode has no internet on the Jetson.** Don't try to `git pull` on the Jetson while in hotspot mode — switch back to regular WiFi first.
- **Model files are gitignored** (`results/` is in `.gitignore`). They're already on the Jetson; if anyone wipes the Jetson disk, you must SCP them back from the dev machine.
- **Background SSH commands silently die** when the parent `ssh` exits. Use `setsid ... < /dev/null & disown` and the `nohup`-with-sleep pattern in `enable_hotspot.sh` for delayed actions (like switching WiFi while the SSH session is still active).
- **port 8001 stays held** for ~30 s after a kill. `sudo fuser -k 8001/tcp` clears it. `demo_start.sh` already does this.
- **The user does NOT want you spamming pgrep/curl status verification rituals after every change.** Launch, one quick check, stop. (See memory `feedback_no_post_launch_verification.md` if available.)
- **The user does NOT want polling background shells** (`until ssh ...; do sleep N; done`). One foreground sleep+check per action.

## File reference

- `jetson/DEMO_RUNBOOK.md` — full runbook with detailed startup/recovery procedures
- `jetson/demo_start.sh` — main startup script
- `jetson/enable_hotspot.sh` / `disable_hotspot.sh` — WiFi mode switch
- `jetson/setup_hotspot.sh` — one-time hotspot configuration (already run; idempotent)
- `jetson/start.sh` — launches all 4 services on the Jetson (called by demo_start.sh)
- `jetson/services/` — three FastAPI microservices (inference / mavlink / data)
- `jetson/ui/` — React UI; pre-built dist served on port 5173

## Key commands cheat-sheet

```bash
# Reach the Jetson
ssh jetson 'hostname -I'                           # find current IP

# Start everything
bash jetson/demo_start.sh

# Switch to portable hotspot mode
bash jetson/enable_hotspot.sh
# (then switch laptop WiFi to "soil-crack-demo")
SSH_HOST=10.42.0.1 bash jetson/demo_start.sh

# Switch back to regular WiFi (replace network name)
WIFI=YourHomeNetworkName bash jetson/disable_hotspot.sh

# Stop everything
ssh jetson 'pkill -f uvicorn; pkill -f "http.server"'

# Recovery: full nuke and restart
ssh jetson 'sudo fuser -k 8001/tcp 8002/tcp 8003/tcp 5173/tcp
            pkill -9 -f uvicorn; pkill -9 -f "http.server"
            sudo systemctl restart nvargus-daemon
            cd ~/soil-crack-detection
            setsid bash jetson/start.sh > /tmp/services.log 2>&1 < /dev/null & disown'
```

## What NOT to do

- Don't push to GitHub from the laptop. The user pushes themselves.
- Don't add Co-Authored-By to commits.
- Don't commit anything to `archive/`, `temp_pseudo/`, or `.mcp.json` if they exist.
- Don't suggest editing CLAUDE.md, model architecture, or training code mid-demo. Stick to operational concerns.
- Don't run TRT compilation — it fails on this model's MobileViT block (Error Code 10). Documented in DEMO_RUNBOOK.md.

## If something is genuinely broken and you can't figure it out

Tell the user. Don't loop on diagnostics. Their dev-machine Claude has the full conversation history and can be re-engaged from the dev WSL.
