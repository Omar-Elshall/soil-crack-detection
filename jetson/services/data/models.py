"""
models.py — Pydantic schemas for the data service.
"""

from typing import List, Optional
from pydantic import BaseModel


class DetectionEvent(BaseModel):
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    north_m: float = 0.0      # NED fallback when no GPS
    east_m: float = 0.0
    heading_deg: float = 0.0
    crack_ratio_pct: float
    mask_png_b64: Optional[str] = None   # base64 encoded PNG, optional


class MissionStartResponse(BaseModel):
    mission_id: str


class MissionMeta(BaseModel):
    id: str
    start_time: str
    end_time: Optional[str] = None
    status: str                          # "active" | "complete"
    model: str = "EfficientCrackNet"
    total_detections: int = 0
    max_coverage_pct: float = 0.0
    mean_coverage_pct: float = 0.0
    flight_duration_s: float = 0.0
    bbox: Optional[dict] = None          # {min_lat, max_lat, min_lon, max_lon}


class MissionListItem(BaseModel):
    id: str
    start_time: str
    status: str
    total_detections: int
    max_coverage_pct: float
    flight_duration_s: float
