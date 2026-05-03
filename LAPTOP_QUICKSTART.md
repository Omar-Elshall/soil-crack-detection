# Laptop Quickstart — One-Time Setup

Run these once on the demo laptop. After that, the demo is **plug in LiPo, plug in USB-C cable, watch the browser open**. Nothing else.

## Prereqs (5 min)

- Windows 10/11 with **WSL2** installed (Ubuntu)
- The Jetson's SSH private key + `~/.ssh/config` entry copied into WSL (see Section A below if not done)
- This repo cloned in WSL at `~/projects/soil-crack-detection`

## A. SSH key — set up on BOTH the WSL side and the Windows side

The watcher uses **Windows-native OpenSSH** so it doesn't depend on WSL being started. You also want WSL ssh working for ad-hoc commands.

**WSL side (`~/.ssh/jetson_nano`):** drop the key into WSL `~/.ssh/`:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
# After transferring jetson_nano + jetson_nano.pub + config into ~/.ssh:
chmod 600 ~/.ssh/jetson_nano ~/.ssh/config
chmod 644 ~/.ssh/jetson_nano.pub
ssh jetson 'echo up'      # should succeed
```

**Windows side (`C:\Users\<you>\.ssh\jetson_nano`):** copy the same files into the Windows-side .ssh:
```cmd
:: From CMD
mkdir "%USERPROFILE%\.ssh"
copy \\wsl.localhost\Ubuntu\home\<you>\.ssh\jetson_nano "%USERPROFILE%\.ssh\"
copy \\wsl.localhost\Ubuntu\home\<you>\.ssh\jetson_nano.pub "%USERPROFILE%\.ssh\"
:: Then edit %USERPROFILE%\.ssh\config to add (or use existing config):
::   Host jetson
::     HostName 192.168.1.233
::     User sdp-w-nano
::     IdentityFile %USERPROFILE%\.ssh\jetson_nano
ssh jetson echo up        :: from CMD or PowerShell — should succeed
```

The watcher updates Windows-side `config` HostName line whenever the Jetson's IP changes.

## B. Clone the repo

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/Omar-Elshall/soil-crack-detection.git
cd soil-crack-detection
git checkout jetson-integration
```

## C. Install Windows autostart entry (the one-button trick)

In Windows File Explorer, navigate to:

```
\\wsl.localhost\Ubuntu\home\<yourwsluser>\projects\soil-crack-detection\jetson\install_laptop_autostart.bat
```

Double-click it. The script:

- Self-elevates (one UAC prompt) and registers a Scheduled Task that runs the watcher elevated at every Windows logon
- Starts the watcher immediately (no need to log out)

From now on, every time you log into Windows, the watcher runs invisibly in WSL. When you plug the USB-C cable into the Jetson, the watcher:

1. Pulls latest scripts from the repo (so the laptop is always current)
2. Detects the Jetson at `192.168.55.1`
3. Best-effort shares your laptop's current WiFi creds with the Jetson via SSH + nmcli
4. Opens your default browser to `http://soilcrack.local:5173`

To check what it's doing: `wsl tail -f /tmp/laptop_autoconnect.log`.

To uninstall: double-click `uninstall_laptop_autostart.bat`.

## D. The actual demo flow

```
1. Plug LiPo onto drone
2. Wait ~45 s, listen for the rising-arpeggio beep from the drone
3. Watcher detects Jetson, opens browser
4. Demo
```

## E. Outdoor demo network strategy

The laptop sustains useful WiFi to one network at a time. Recommended:

**Best (simplest):** Both Jetson and laptop on your **phone hotspot**.
- Phone runs WiFi hotspot
- Watcher bootstraps Jetson onto it (one-time per venue)
- Laptop joins same hotspot
- Zoom + website both work over the same connection
- Bandwidth: Zoom ~3 Mbps + website ~3-5 Mbps; modern phone hotspot 50+ Mbps

**Alternate (isolated bandwidth):** USB-tether the phone for laptop internet.
- Plug phone USB into laptop, enable USB tethering
- Laptop's WiFi adapter is now free
- Watcher puts Jetson on its own hotspot ("soil-crack-demo")
- Laptop joins Jetson's hotspot for the website
- Zoom flows through the phone USB; website through the WiFi

**Backup (no usable phone hotspot):** stick with Jetson hotspot, run Zoom on a teammate's device.

## F. Telemetry radio (SiK) — primary MAVLink path

When the drone flies past the laptop's WiFi range, the camera/UI feed dies. The SiK telemetry radio (USB-A on the laptop, JST-GH on Pixhawk TELEM1) keeps telemetry + flight commands flowing — it reaches hundreds of meters and is bidirectional.

After running `install_laptop_autostart.bat`, the watcher automatically:
1. Detects the Jetson, opens the browser
2. Spawns `laptop_mavlink_relay.py` against the radio's COM port (auto-detected: SiLabs / FTDI USB serial)
3. Hosts the same WS + REST contract as the Jetson MAVLink service on `localhost:18002`
4. The UI prefers the local relay over the Jetson over WiFi — so commands and live telemetry keep working even when the drone is far from the laptop

Pixhawk firmware needs MAVLink on TELEM1 (one-time, in Mission Planner / QGC):
- ArduCopter: `SERIAL1_PROTOCOL=2`, `SERIAL1_BAUD=57`
- PX4: `MAV_1_CONFIG=TELEM1`, `MAV_1_MODE=Normal`, `MAV_1_RATE=auto`

## If anything goes wrong

Open Claude Code in the repo dir, paste the error. Read `jetson/LAPTOP_CLAUDE.md` for context.

Manual fallback to bring the demo up:

```bash
bash jetson/demo_start.sh
```

Full runbook: `jetson/DEMO_RUNBOOK.md`
