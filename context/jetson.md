# Jetson Context

## Hardware

- **Device:** NVIDIA Jetson Orin Nano 8GB (dev kit module, extracted and placed in Holybro Pixhawk Jetson Baseboard)
- **SSH:** `ssh jetson` — key auth configured, no password needed from WSL
- **IP:** 192.168.1.222, user: `sdp-w-nano`
- **OS:** JetPack R36.5.0, Ubuntu 22.04, aarch64
- **CUDA:** 12.6

## Flight Controller

- **Pixhawk 6C** — cannot plug directly into the baseboard's integrated Pixhawk connector (6C form factor incompatible). Connected externally via UART port-to-port cabling.
- Baseboard serial ports: `/dev/ttyTHS1`, `/dev/ttyTHS2` — these are used for Pixhawk MAVLink communication.

## Camera

- **Arducam IMX477 Pi HQ** — connected via CSI port on the Holybro baseboard. Does NOT show up as `/dev/video*` — requires GStreamer pipeline to access frames.

## Drone Frame

- **Holybro X500 V2** — fully assembled, flies with RC transmitter but needs PID tuning for stable flight.

---

## Software Environment

### Python / PyTorch Stack

| Package | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.11.0 (Jetson AI Lab build) |
| torchvision | 0.26.0 (Jetson AI Lab build) |
| einops | 0.8.2 |

### Key env vars (set in `~/.profile` and `~/.bashrc`)

```bash
PYTHONPATH=$PYTHONPATH:/home/sdp-w-nano/soil-crack-detection
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libcudss/12:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

These are sourced automatically in login shells. For non-interactive SSH commands use `bash -l -c '...'`.

### Repo location

```
~/soil-crack-detection/
  results/saved_models/EfficientCrackNet/best_model_num_real_4.pt   # checkpoint
  data/test/images/   # IMG0002.png, IMG0011.png, IMG0015.png
  data/test/masks/    # MASK0002.png, MASK0011.png, MASK0015.png
```

### Package install

`crack_detection` package is on PYTHONPATH directly (editable install failed due to old pip). Do NOT run `pip install -e .` on Jetson — just ensure PYTHONPATH is set.

---

## Working Commands on Jetson

```bash
# Run predict.py (sample mode)
bash -l -c 'cd ~/soil-crack-detection && python3 scripts/predict.py --mode sample --data_dir data/ --model_path results/saved_models/EfficientCrackNet/best_model_num_real_4.pt --num_images 3'

# Run from WSL via SSH
ssh jetson "bash -l -c 'cd ~/soil-crack-detection && python3 scripts/predict.py --mode sample --data_dir data/ --model_path results/saved_models/EfficientCrackNet/best_model_num_real_4.pt --num_images 3'"
```

---

## Installation Notes (for reference)

### How torchvision was installed

Standard `pip install torchvision` installs a manylinux wheel incompatible with the Jetson PyTorch C++ extensions (`operator torchvision::nms does not exist` error). The fix:

```bash
# Install from Jetson AI Lab (NVIDIA's official Jetson PyPI index)
pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision --no-deps --upgrade

# Install missing CUDA library required by torch 2.10
sudo apt-get install -y libcudss0-cuda-12
```

### Why libcudss was needed

torch 2.10 (Jetson AI Lab build) requires `libcudss.so.0` which is not installed by default in JetPack 6.5. It is available via apt from NVIDIA's CUDA repo (`libcudss0-cuda-12`). Installed to `/usr/lib/aarch64-linux-gnu/libcudss/12/`.

---

## Working Commands on Jetson (Phase 2 — Live Inference)

```bash
# Run live inference on Jetson (must have a display connected)
bash -l -c 'cd ~/soil-crack-detection && python3 jetson/live_inference.py'

# Full sensor res (sharpest, slower)
bash -l -c 'cd ~/soil-crack-detection && python3 jetson/live_inference.py --sensor_mode 0'

# Controls: q=quit, s=save frame+mask+overlay to results/live_captures/
```

---

## Next Steps

1. **Phase 2** — Live camera inference from Arducam IMX477 via CSI (GStreamer pipeline)
2. **Phase 3** — Jetson ↔ Pixhawk 6C MAVLink communication over `/dev/ttyTHS1` or `/dev/ttyTHS2`
3. **Phase 4** — PID tuning for stable flight (done in QGroundControl, not code)
4. **Phase 5** — Autonomous offboard control from Jetson via MAVSDK-Python
