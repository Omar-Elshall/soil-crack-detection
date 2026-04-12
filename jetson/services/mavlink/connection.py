"""
connection.py — pymavlink connection manager for Pixhawk 6C over USB.
Handles open/close, heartbeat sending, and reconnect logic.
"""

import os
import threading
import time

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 921600


class MAVLinkConnection:
    def __init__(self, port: str = None, baud: int = None):
        self.port = port or os.environ.get("MAVLINK_PORT", DEFAULT_PORT)
        self.baud = baud or int(os.environ.get("MAVLINK_BAUD", DEFAULT_BAUD))
        self.master = None
        self.connected = False
        self._lock = threading.Lock()
        self._heartbeat_thread = None
        self._running = False

    def connect(self, timeout: int = 30) -> bool:
        if not MAVLINK_AVAILABLE:
            print("WARNING: pymavlink not installed. Running in simulation mode.")
            self.connected = False
            return False
        try:
            print(f"Connecting to Pixhawk on {self.port} @ {self.baud} baud...")
            self.master = mavutil.mavlink_connection(
                f"{self.port}", baud=self.baud, source_system=255
            )
            self.master.wait_heartbeat(timeout=timeout)
            self.connected = True
            print(f"Pixhawk connected. sysid={self.master.target_system} compid={self.master.target_component}")

            # Request all telemetry streams — ArduPilot won't send them by default
            streams = [
                mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,      # IMU
                mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,   # SYS_STATUS, battery
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,          # GPS, local NED
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,            # ATTITUDE
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,            # VFR_HUD
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,            # AHRS etc.
            ]
            for stream_id in streams:
                self.master.mav.request_data_stream_send(
                    self.master.target_system,
                    self.master.target_component,
                    stream_id,
                    10,  # 10 Hz
                    1,   # start
                )
            time.sleep(0.1)
            print("Telemetry streams requested at 10 Hz.")

            self._running = True
            self._heartbeat_thread = threading.Thread(target=self._send_heartbeat, daemon=True)
            self._heartbeat_thread.start()
            return True
        except Exception as e:
            print(f"MAVLink connection failed: {e}")
            self.connected = False
            return False

    def _send_heartbeat(self):
        """Send GCS heartbeat every 1s so Pixhawk doesn't timeout."""
        while self._running and self.connected:
            try:
                with self._lock:
                    self.master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
            except Exception:
                pass
            time.sleep(1.0)

    def recv(self, msg_type: str = None, blocking: bool = False, timeout: float = 0.1):
        if not self.connected or not self.master:
            return None
        with self._lock:
            return self.master.recv_match(type=msg_type, blocking=blocking, timeout=timeout)

    def send_command_long(self, command, param1=0, param2=0, param3=0,
                          param4=0, param5=0, param6=0, param7=0) -> bool:
        if not self.connected or not self.master:
            return False
        with self._lock:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                command, 0,
                param1, param2, param3, param4, param5, param6, param7
            )
        return True

    def disconnect(self):
        self._running = False
        self.connected = False
        if self.master:
            self.master.close()
