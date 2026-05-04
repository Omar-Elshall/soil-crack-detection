"""
routes.py — FastAPI routes for MAVLink service.
  WS   /ws/telemetry         — 5 Hz telemetry broadcast
  GET  /status               — telemetry snapshot
  POST /command/{action}     — flight commands
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

try:
    from pymavlink import mavutil as _mavutil
    _MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = _mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
except Exception:
    _MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1

router = APIRouter()

# Injected by main.py at startup
poller = None
flight = None
conn = None


@router.post("/command/demo-flight")
async def demo_flight(
    altitude_m: float = 2.0,
    forward_m: float = 3.0,
    speed_m_s: float = 0.2,
):
    """Sequenced detection-pass demo: GUIDED -> ARM -> takeoff to
    `altitude_m` -> set ground speed `speed_m_s` -> goto N=`forward_m`
    (north of arming point) at the same altitude -> LAND -> disarm.

    Designed for an outdoor demo over A1 crack-print sheets: drone hovers
    up, crawls forward at walking-pace so the camera covers the prints,
    then lands. Defaults: 2 m altitude, 3 m forward, 0.2 m/s.
    """
    import asyncio
    if flight is None:
        return {"ok": False, "error": "Service not ready"}

    steps = []

    def step(name, result):
        ok = bool(result.get("ok"))
        steps.append({"step": name, "ok": ok, "message": result.get("message", "")})
        return ok

    if not step("guided", flight.set_guided_mode()):
        return {"ok": False, "steps": steps}
    await asyncio.sleep(1.0)

    if not step("arm", flight.arm()):
        return {"ok": False, "steps": steps}
    await asyncio.sleep(1.0)

    if not step("takeoff", flight.takeoff(altitude_m)):
        flight.disarm()
        return {"ok": False, "steps": steps}

    # Let it climb close to target altitude before issuing the forward command.
    # rough rule of thumb: ~1 s per meter of climb at ArduCopter's default WPNAV_SPEED_UP.
    await asyncio.sleep(max(4.0, altitude_m * 1.5))

    if not step("set_speed", flight.set_speed(speed_m_s)):
        flight.land(); flight.disarm()
        return {"ok": False, "steps": steps}
    await asyncio.sleep(0.5)

    if not step("forward", flight.goto_ned(forward_m, 0.0, altitude_m)):
        flight.land(); flight.disarm()
        return {"ok": False, "steps": steps}

    # Wait for the leg + a small buffer so the drone actually arrives.
    travel_s = forward_m / max(speed_m_s, 0.05)
    await asyncio.sleep(travel_s + 4.0)
    steps.append({"step": "traveled", "ok": True, "message": f"{forward_m} m at {speed_m_s} m/s ({travel_s:.1f}s)"})

    if not step("land", flight.land()):
        return {"ok": False, "steps": steps}

    await asyncio.sleep(8.0)
    step("disarm", flight.disarm())

    return {"ok": True, "steps": steps}


@router.post("/command/hover-flight")
async def hover_flight(altitude_m: float = 1.0, hover_seconds: float = 30.0):
    """Stationary hover test (no horizontal movement):
    GUIDED -> ARM -> takeoff to `altitude_m` -> hover for `hover_seconds`
    -> LAND -> disarm. Same shape as the original demo-flight before the
    forward-pass version replaced it."""
    import asyncio
    if flight is None:
        return {"ok": False, "error": "Service not ready"}
    steps = []
    def step(name, result):
        ok = bool(result.get("ok"))
        steps.append({"step": name, "ok": ok, "message": result.get("message", "")})
        return ok
    if not step("guided", flight.set_guided_mode()): return {"ok": False, "steps": steps}
    await asyncio.sleep(1.0)
    if not step("arm", flight.arm()): return {"ok": False, "steps": steps}
    await asyncio.sleep(1.0)
    if not step("takeoff", flight.takeoff(altitude_m)):
        flight.disarm(); return {"ok": False, "steps": steps}
    await asyncio.sleep(hover_seconds)
    steps.append({"step": "hovered", "ok": True, "message": f"{hover_seconds}s"})
    if not step("land", flight.land()): return {"ok": False, "steps": steps}
    await asyncio.sleep(8.0)
    step("disarm", flight.disarm())
    return {"ok": True, "steps": steps}


@router.post("/command/play-tone")
async def play_tone(tune: str = "MFT200L8>cdefg"):
    """Send a PLAY_TUNE message to the Pixhawk's onboard buzzer.
    Default is a quick rising arpeggio (~0.4 s). Used as the audible
    'all services up' signal at the end of demo_start.sh and on boot.
    """
    if conn is None or not getattr(conn, "connected", False) or conn.master is None:
        return {"ok": False, "error": "Pixhawk not connected"}
    try:
        conn.master.mav.play_tune_send(
            conn.master.target_system,
            conn.master.target_component,
            tune.encode("ascii", errors="ignore"),
        )
        return {"ok": True, "tune": tune}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
    # Pause telemetry poller so it doesn't consume MISSION_ACK / MISSION_REQUEST messages
    if poller:
        poller.pause()
    try:
        result = await asyncio.to_thread(flight.upload_mission, wps, body.takeoff_alt)
    finally:
        if poller:
            poller.resume()
    return result


@router.post("/command/start-mission")
async def cmd_start_mission():
    """Set AUTO mode to execute the uploaded mission."""
    if not flight:
        return {"ok": False, "message": "Service not ready"}
    from pymavlink import mavutil as mu
    ok = flight.conn.send_command_long(
        mu.mavlink.MAV_CMD_DO_SET_MODE,
        param1=mu.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        param2=3,  # AUTO mode
    )
    return {"ok": ok, "message": "AUTO mode engaged" if ok else "Failed to set AUTO mode"}


@router.post("/command/test-flight")
async def cmd_test_flight():
    """Motor test: STABILIZE → force-arm → 5 s → disarm. No GPS required."""
    if not flight:
        return {"ok": False, "message": "Service not ready"}
    if poller:
        poller.pause()
    try:
        result = await asyncio.to_thread(flight.test_flight)
    finally:
        if poller:
            poller.resume()
    return result
