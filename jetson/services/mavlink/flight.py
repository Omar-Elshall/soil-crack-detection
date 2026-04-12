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
