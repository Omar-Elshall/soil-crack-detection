"""
routes.py — FastAPI routes for MAVLink service.
  WS   /ws/telemetry         — 5 Hz telemetry broadcast
  GET  /status               — telemetry snapshot
  POST /command/{action}     — flight commands
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter()

# Injected by main.py at startup
poller = None
flight = None


class GotoBody(BaseModel):
    north_m: float = 0.0
    east_m: float = 0.0
    alt_m: float = 0.3


class TakeoffBody(BaseModel):
    altitude_m: float = 0.3


class WaypointItem(BaseModel):
    lat: float
    lon: float
    alt: float = 4.0


class MissionBody(BaseModel):
    waypoints: list[WaypointItem]
    takeoff_alt: float = 4.0


@router.get("/status")
async def status():
    if poller is None:
        return {"connected": False}
    return poller.get()


@router.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    if poller:
        poller.add_ws_client(websocket)
    try:
        while True:
            await asyncio.sleep(0.2)   # 5 Hz
            data = poller.get() if poller else {"connected": False}
            await websocket.send_json(data)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if poller:
            poller.remove_ws_client(websocket)


@router.post("/command/arm")
async def cmd_arm():
    return flight.arm() if flight else {"ok": False, "message": "Service not ready"}


@router.post("/command/disarm")
async def cmd_disarm():
    return flight.disarm() if flight else {"ok": False, "message": "Service not ready"}


@router.post("/command/guided")
async def cmd_guided():
    return flight.set_guided_mode() if flight else {"ok": False, "message": "Service not ready"}


@router.post("/command/takeoff")
async def cmd_takeoff(body: TakeoffBody):
    return flight.takeoff(body.altitude_m) if flight else {"ok": False, "message": "Service not ready"}


@router.post("/command/goto")
async def cmd_goto(body: GotoBody):
    return flight.goto_ned(body.north_m, body.east_m, body.alt_m) if flight else {"ok": False, "message": "Service not ready"}


@router.post("/command/land")
async def cmd_land():
    return flight.land() if flight else {"ok": False, "message": "Service not ready"}


@router.post("/command/rtl")
async def cmd_rtl():
    return flight.rtl() if flight else {"ok": False, "message": "Service not ready"}


@router.post("/command/upload-mission")
async def cmd_upload_mission(body: MissionBody):
    if not flight:
        return {"ok": False, "message": "Service not ready"}
    wps = [{"lat": w.lat, "lon": w.lon, "alt": w.alt} for w in body.waypoints]
    return flight.upload_mission(wps, body.takeoff_alt)


@router.post("/command/start-mission")
async def cmd_start_mission():
    """Set AUTO mode to execute the uploaded mission."""
    if not flight:
        return {"ok": False, "message": "Service not ready"}
    ok = flight.conn.send_command_long(
        __import__("pymavlink").mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        param1=__import__("pymavlink").mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        param2=3,  # AUTO mode
    )
    return {"ok": ok, "message": "AUTO mode" if ok else "Failed"}
