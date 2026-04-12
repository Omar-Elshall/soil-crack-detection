"""
recorder.py — Mission recorder. Manages active mission lifecycle and per-frame logging.
"""

import base64
import csv
import json
import os
import threading
import time
from datetime import datetime
from typing import Optional

MISSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "missions")


class MissionRecorder:
    def __init__(self):
        os.makedirs(MISSIONS_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._active: dict = {}    # mission_id → {csv_file, writer, start_time, count, max_ratio, ratios}

    def start_mission(self, model: str = "EfficientCrackNet") -> str:
        mission_id = "mission_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        mission_dir = os.path.join(MISSIONS_DIR, mission_id)
        masks_dir = os.path.join(mission_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        # Write stub meta
        meta = {
            "id": mission_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "active",
            "model": model,
            "total_detections": 0,
            "max_coverage_pct": 0.0,
            "mean_coverage_pct": 0.0,
            "flight_duration_s": 0.0,
            "bbox": None,
        }
        with open(os.path.join(mission_dir, "mission_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Open CSV for writing
        csv_path = os.path.join(mission_dir, "detections.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "lat", "lon", "alt_m", "north_m", "east_m",
                         "heading_deg", "crack_ratio_pct", "mask_filename"])

        with self._lock:
            self._active[mission_id] = {
                "dir": mission_dir,
                "csv_file": csv_file,
                "writer": writer,
                "start_time": time.time(),
                "count": 0,
                "max_ratio": 0.0,
                "ratios": [],
                "lats": [],
                "lons": [],
            }

        print(f"Mission started: {mission_id}")
        return mission_id

    def log_detection(self, mission_id: str, lat: float, lon: float, alt_m: float,
                      north_m: float, east_m: float, heading_deg: float,
                      crack_ratio_pct: float, mask_png_b64: Optional[str] = None):
        with self._lock:
            if mission_id not in self._active:
                return
            ctx = self._active[mission_id]

        ctx["count"] += 1

        # Save mask PNG if provided
        mask_filename = ""
        if mask_png_b64:
            mask_filename = f"frame_{ctx['count']:04d}_mask.png"
            mask_path = os.path.join(ctx["dir"], "masks", mask_filename)
            try:
                with open(mask_path, "wb") as f:
                    f.write(base64.b64decode(mask_png_b64))
            except Exception:
                mask_filename = ""

        timestamp = datetime.now().isoformat()
        ctx["writer"].writerow([
            timestamp, lat, lon, alt_m, north_m, east_m,
            heading_deg, crack_ratio_pct, mask_filename
        ])
        ctx["csv_file"].flush()
        ctx["max_ratio"] = max(ctx["max_ratio"], crack_ratio_pct)
        ctx["ratios"].append(crack_ratio_pct)
        if lat != 0.0: ctx["lats"].append(lat)
        if lon != 0.0: ctx["lons"].append(lon)

    def stop_mission(self, mission_id: str) -> dict:
        with self._lock:
            if mission_id not in self._active:
                return {"ok": False, "message": "Mission not found"}
            ctx = self._active.pop(mission_id)

        ctx["csv_file"].close()

        duration = time.time() - ctx["start_time"]
        ratios = ctx["ratios"]
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0

        bbox = None
        if ctx["lats"] and ctx["lons"]:
            bbox = {
                "min_lat": min(ctx["lats"]), "max_lat": max(ctx["lats"]),
                "min_lon": min(ctx["lons"]), "max_lon": max(ctx["lons"]),
            }

        meta_path = os.path.join(ctx["dir"], "mission_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta.update({
            "end_time": datetime.now().isoformat(),
            "status": "complete",
            "total_detections": ctx["count"],
            "max_coverage_pct": round(ctx["max_ratio"], 2),
            "mean_coverage_pct": round(mean_ratio, 2),
            "flight_duration_s": round(duration, 1),
            "bbox": bbox,
        })
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Mission stopped: {mission_id} | {ctx['count']} detections | {duration:.0f}s")
        return {"ok": True, "mission_id": mission_id, "meta": meta}

    def list_missions(self) -> list:
        missions = []
        if not os.path.exists(MISSIONS_DIR):
            return missions
        for name in sorted(os.listdir(MISSIONS_DIR), reverse=True):
            meta_path = os.path.join(MISSIONS_DIR, name, "mission_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    missions.append(json.load(f))
        return missions

    def get_mission(self, mission_id: str) -> Optional[dict]:
        meta_path = os.path.join(MISSIONS_DIR, mission_id, "mission_meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            return json.load(f)

    def mission_dir(self, mission_id: str) -> str:
        return os.path.join(MISSIONS_DIR, mission_id)
