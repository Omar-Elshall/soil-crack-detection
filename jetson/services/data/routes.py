"""
routes.py — FastAPI routes for Data service (port 8003).
  POST /missions/start                  — start a new mission
  POST /missions/{id}/stop              — stop mission, finalize meta
  POST /missions/{id}/detect            — log a detection event
  GET  /missions                        — list all missions
  GET  /missions/{id}                   — get mission meta
  GET  /missions/{id}/detections        — get detections as JSON array
  GET  /missions/{id}/export/csv        — download detections.csv
  GET  /missions/{id}/export/geojson    — download GeoJSON
  GET  /missions/{id}/export/pdf        — download PDF report
  GET  /missions/{id}/masks/{filename} — serve mask PNG
"""

import csv
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

from .exporter import mission_to_geojson, mission_to_pdf
from .models import DetectionEvent, MissionMeta, MissionStartResponse
from .recorder import MissionRecorder

router = APIRouter()
recorder = MissionRecorder()


# ── Mission lifecycle ────────────────────────────────────────────────────────

@router.post("/missions/start", response_model=MissionStartResponse)
async def start_mission(model: str = "EfficientCrackNet"):
    mission_id = recorder.start_mission(model=model)
    return MissionStartResponse(mission_id=mission_id)


@router.post("/missions/{mission_id}/stop")
async def stop_mission(mission_id: str):
    result = recorder.stop_mission(mission_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("message", "Not found"))
    return result


@router.post("/missions/{mission_id}/detect")
async def log_detection(mission_id: str, event: DetectionEvent):
    recorder.log_detection(
        mission_id=mission_id,
        lat=event.lat,
        lon=event.lon,
        alt_m=event.alt_m,
        north_m=event.north_m,
        east_m=event.east_m,
        heading_deg=event.heading_deg,
        crack_ratio_pct=event.crack_ratio_pct,
        mask_png_b64=event.mask_png_b64,
    )
    return {"ok": True}


# ── Mission queries ──────────────────────────────────────────────────────────

@router.get("/missions")
async def list_missions():
    return recorder.list_missions()


@router.get("/missions/{mission_id}", response_model=MissionMeta)
async def get_mission(mission_id: str):
    meta = recorder.get_mission(mission_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return meta


# ── Detections JSON ──────────────────────────────────────────────────────────

@router.get("/missions/{mission_id}/detections")
async def get_detections(mission_id: str):
    mission_dir = recorder.mission_dir(mission_id)
    csv_path = os.path.join(mission_dir, "detections.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="No detections found")
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "timestamp":       row.get("timestamp", ""),
                "lat":             float(row.get("lat", 0)),
                "lon":             float(row.get("lon", 0)),
                "alt_m":           float(row.get("alt_m", 0)),
                "north_m":         float(row.get("north_m", 0)),
                "east_m":          float(row.get("east_m", 0)),
                "heading_deg":     float(row.get("heading_deg", 0)),
                "crack_ratio_pct": float(row.get("crack_ratio_pct", 0)),
                "mask_filename":   row.get("mask_filename", ""),
            })
    return rows


# ── Exports ──────────────────────────────────────────────────────────────────

@router.get("/missions/{mission_id}/export/csv")
async def export_csv(mission_id: str):
    mission_dir = recorder.mission_dir(mission_id)
    csv_path = os.path.join(mission_dir, "detections.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"{mission_id}_detections.csv",
    )


@router.get("/missions/{mission_id}/export/geojson")
async def export_geojson(mission_id: str):
    meta = recorder.get_mission(mission_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    mission_dir = recorder.mission_dir(mission_id)
    geojson = mission_to_geojson(mission_dir, meta)
    return JSONResponse(
        content=geojson,
        headers={"Content-Disposition": f'attachment; filename="{mission_id}.geojson"'},
    )


@router.get("/missions/{mission_id}/export/pdf")
async def export_pdf(mission_id: str):
    meta = recorder.get_mission(mission_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    mission_dir = recorder.mission_dir(mission_id)
    pdf_bytes = mission_to_pdf(mission_dir, meta)
    if pdf_bytes is None:
        raise HTTPException(status_code=503, detail="reportlab not installed")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{mission_id}_report.pdf"'},
    )


# ── Mask images ──────────────────────────────────────────────────────────────

@router.get("/missions/{mission_id}/masks/{filename}")
async def get_mask(mission_id: str, filename: str):
    mission_dir = recorder.mission_dir(mission_id)
    mask_path = os.path.join(mission_dir, "masks", filename)
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Mask not found")
    return FileResponse(mask_path, media_type="image/png")
