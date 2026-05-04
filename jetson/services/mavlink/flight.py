"""
flight.py — Flight commands for ArduPilot GUIDED mode via pymavlink.
All commands return {"ok": bool, "message": str}.
"""

import time

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False


class FlightController:
    def __init__(self, connection):
        self.conn = connection

    def _check(self) -> tuple[bool, str]:
        if not MAVLINK_AVAILABLE:
            return False, "pymavlink not available"
        if not self.conn.connected:
            return False, "Pixhawk not connected"
        return True, ""

    def set_guided_mode(self) -> dict:
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        # ArduCopter GUIDED mode = 4
        ok = self.conn.send_command_long(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            param1=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            param2=4,  # GUIDED
        )
        return {"ok": ok, "message": "Set GUIDED mode" if ok else "Failed"}

    def arm(self) -> dict:
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        ok = self.conn.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1,  # arm
            param2=0,
        )
        return {"ok": ok, "message": "Armed" if ok else "Arm failed"}

    def disarm(self) -> dict:
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        ok = self.conn.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=0,  # disarm
        )
        return {"ok": ok, "message": "Disarmed" if ok else "Disarm failed"}

    def takeoff(self, altitude_m: float = 0.3) -> dict:
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        # Must be in GUIDED and armed first
        ok = self.conn.send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            param7=altitude_m,
        )
        return {"ok": ok, "message": f"Takeoff to {altitude_m}m" if ok else "Takeoff failed"}

    def goto_ned(self, north_m: float, east_m: float, alt_m: float) -> dict:
        """Fly to position in NED frame relative to arming point."""
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        master = self.conn.master
        master.mav.set_position_target_local_ned_send(
            0,                      # time_boot_ms (ignored)
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,     # position only (ignore vel/accel/yaw)
            north_m, east_m, -alt_m,  # NED: down is positive, so negate alt
            0, 0, 0,                # velocity (ignored)
            0, 0, 0,                # acceleration (ignored)
            0, 0,                   # yaw, yaw_rate (ignored)
        )
        return {"ok": True, "message": f"Goto N={north_m} E={east_m} alt={alt_m}m"}

    def set_speed(self, speed_m_s: float) -> dict:
        """Set ground speed via MAV_CMD_DO_CHANGE_SPEED.
        Stays in effect until changed or mode reset."""
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        ok = self.conn.send_command_long(
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            param1=1,           # 1 = ground speed
            param2=speed_m_s,
            param3=-1,          # throttle: -1 = no change
        )
        return {"ok": ok, "message": f"Speed -> {speed_m_s} m/s" if ok else "Set-speed failed"}

    def land(self) -> dict:
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        ok = self.conn.send_command_long(mavutil.mavlink.MAV_CMD_NAV_LAND)
        return {"ok": ok, "message": "Landing" if ok else "Land command failed"}

    def rtl(self) -> dict:
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        ok = self.conn.send_command_long(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)
        return {"ok": ok, "message": "RTL" if ok else "RTL failed"}

    def set_stabilize_mode(self) -> dict:
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        ok = self.conn.send_command_long(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            param1=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            param2=0,  # STABILIZE = 0
        )
        return {"ok": ok, "message": "Set STABILIZE mode" if ok else "Failed"}

    def arm_force(self) -> dict:
        """Arm bypassing pre-arm GPS checks (for ground/indoor tests)."""
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}
        ok = self.conn.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1,    # arm
            param2=21196,  # ArduPilot force-arm magic number — bypasses pre-arm checks
        )
        return {"ok": ok, "message": "Force armed" if ok else "Force arm failed"}

    def test_flight(self) -> dict:
        """Motor spin test — no GPS required.
        STABILIZE mode + force-arm → spin 5 s → disarm.
        Runs synchronously — call via asyncio.to_thread from FastAPI.
        """
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}

        step = self.set_stabilize_mode()
        if not step["ok"]:
            return {"ok": False, "message": f"STABILIZE failed: {step['message']}"}
        time.sleep(1.0)

        step = self.arm_force()
        if not step["ok"]:
            return {"ok": False, "message": f"Arm failed: {step['message']}"}
        time.sleep(5.0)   # motors spin for 5 s

        self.disarm()
        return {"ok": True, "message": "Motor test complete — armed 5 s in STABILIZE, disarmed"}

    def upload_mission(self, waypoints: list[dict], takeoff_alt: float = 4.0) -> dict:
        """Upload a list of {lat, lon, alt} waypoints as an AUTO mission.
        Automatically prepends a takeoff waypoint and appends RTL.
        waypoints: [{"lat": float, "lon": float, "alt": float}, ...]
        """
        ok, msg = self._check()
        if not ok:
            return {"ok": False, "message": msg}

        master = self.conn.master

        # Build full mission item list:
        # 0 = home (current position, dummy)
        # 1 = takeoff
        # 2..N = survey waypoints
        # N+1 = RTL
        items = []

        # Item 0: home (set to current lat/lon/0, ArduPilot fills it)
        items.append({
            "seq": 0,
            "frame": mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            "command": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            "current": 0, "autocontinue": 1,
            "param1": 0, "param2": 0, "param3": 0, "param4": 0,
            "x": 0, "y": 0, "z": 0,
        })

        # Item 1: takeoff
        items.append({
            "seq": 1,
            "frame": mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            "command": mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            "current": 1, "autocontinue": 1,
            "param1": 0, "param2": 0, "param3": 0, "param4": 0,
            "x": 0, "y": 0, "z": takeoff_alt,
        })

        # Survey waypoints
        for i, wp in enumerate(waypoints):
            items.append({
                "seq": i + 2,
                "frame": mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                "command": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                "current": 0, "autocontinue": 1,
                "param1": 0, "param2": 0, "param3": 0, "param4": float("nan"),
                "x": float(wp["lat"]), "y": float(wp["lon"]), "z": float(wp.get("alt", takeoff_alt)),
            })

        # RTL at end
        items.append({
            "seq": len(items),
            "frame": mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            "command": mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            "current": 0, "autocontinue": 1,
            "param1": 0, "param2": 0, "param3": 0, "param4": 0,
            "x": 0, "y": 0, "z": 0,
        })

        n = len(items)

        try:
            # Clear existing mission
            master.mav.mission_clear_all_send(master.target_system, master.target_component)
            ack = master.recv_match(type="MISSION_ACK", blocking=True, timeout=5)
            if not ack:
                return {"ok": False, "message": "No ACK after MISSION_CLEAR_ALL"}

            # Send count
            master.mav.mission_count_send(master.target_system, master.target_component, n)

            # Send each item as ArduPilot requests it
            for _ in range(n):
                req = master.recv_match(type=["MISSION_REQUEST", "MISSION_REQUEST_INT"], blocking=True, timeout=5)
                if not req:
                    return {"ok": False, "message": f"Timeout waiting for MISSION_REQUEST (sent {_}/{n})"}
                idx = req.seq
                it = items[idx]
                master.mav.mission_item_int_send(
                    master.target_system,
                    master.target_component,
                    it["seq"],
                    it["frame"],
                    it["command"],
                    it["current"],
                    it["autocontinue"],
                    it["param1"], it["param2"], it["param3"], it["param4"],
                    int(it["x"] * 1e7),   # lat in degE7
                    int(it["y"] * 1e7),   # lon in degE7
                    it["z"],
                )

            # Wait for final ACK
            ack = master.recv_match(type="MISSION_ACK", blocking=True, timeout=5)
            if not ack or ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                return {"ok": False, "message": f"Mission not accepted (ack={ack.type if ack else 'none'})"}

            return {"ok": True, "message": f"Mission uploaded: {n} items ({len(waypoints)} waypoints)"}

        except Exception as e:
            return {"ok": False, "message": str(e)}
