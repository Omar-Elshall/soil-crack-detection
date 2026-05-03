# Laptop Quickstart — One-Time Setup

Run these once on the demo laptop. After that, the demo is **plug in LiPo, plug in USB-C cable, watch the browser open**. Nothing else.

## Prereqs (5 min)

- Windows 10/11 with **WSL2** installed (Ubuntu)
- The Jetson's SSH private key + `~/.ssh/config` entry copied into WSL (see Section A below if not done)
- This repo cloned in WSL at `~/projects/soil-crack-detection`

## A. SSH key for `ssh jetson` (skip if `ssh jetson 'echo up'` already works)

The Jetson accepts the `jetson_nano` key — copy it from the dev WSL via Windows Explorer / OneDrive / USB:

```bash
# In WSL on the demo laptop:
mkdir -p ~/.ssh && chmod 700 ~/.ssh

# Drop these 3 files into ~/.ssh from wherever you transferred them:
#   jetson_nano        (private key)
#   jetson_nano.pub    (public key)
#   config             (with the 'Host jetson' entry)

chmod 600 ~/.ssh/jetson_nano ~/.ssh/config
chmod 644 ~/.ssh/jetson_nano.pub
ssh jetson 'echo up'   # should succeed without a password
```

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
3. Plug USB-C cable to laptop
4. Browser opens automatically
5. Demo
```

## If anything goes wrong

Open Claude Code in the repo dir, paste the error. Read `jetson/LAPTOP_CLAUDE.md` for context.

Manual fallback to bring the demo up:

```bash
bash jetson/demo_start.sh
```

Full runbook: `jetson/DEMO_RUNBOOK.md`
