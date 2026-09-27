"""
flight_physics.py

Simulates a simple but physically-consistent quadcopter flight.
This is the SOURCE OF TRUTH: GPS, IMU and telemetry are all derived from
the same underlying kinematic state, so an attack that tampers with only
ONE data stream (e.g. GPS) becomes inconsistent with the others (e.g. IMU).
That cross-layer inconsistency is exactly what our detectors will exploit.

State per timestep:
  t           - seconds since start
  lat, lon    - degrees (simulated local-flat approximation)
  alt         - meters AGL
  vx, vy, vz  - m/s, NED frame
  ax, ay, az  - m/s^2, IMU-measured acceleration (NED + gravity removed)
  roll,pitch,yaw - radians
  heading     - degrees compass
"""

import numpy as np
from dataclasses import dataclass, field

EARTH_RADIUS_M = 6371000.0
HOME_LAT = 19.1334   # IIT Bombay approx, just for realism
HOME_LON = 72.9133

@dataclass
class DroneState:
    t: float = 0.0
    lat: float = HOME_LAT
    lon: float = HOME_LON
    alt: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    heading: float = 0.0


def meters_to_latlon_delta(dx, dy, lat0):
    dlat = dy / EARTH_RADIUS_M * (180 / np.pi)
    dlon = dx / (EARTH_RADIUS_M * np.cos(np.radians(lat0))) * (180 / np.pi)
    return dlat, dlon


class NormalFlightSimulator:
    """
    Generates a repeatable 'mission': takeoff, square-pattern cruise,
    loiter, RTL. Adds small realistic sensor noise so the baseline
    isn't a perfectly straight line (which would make detection trivially
    easy and unconvincing to judges).
    """

    def __init__(self, dt=0.1, seed=42):
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.state = DroneState()
        self._waypoints = self._build_mission()
        self._wp_idx = 0
        self.cruise_speed = 8.0  # m/s
        self.cruise_alt = 30.0   # m

    def _build_mission(self):
        # simple square pattern (meters, relative to home)
        return [
            (0, 0, 30),
            (100, 0, 30),
            (100, 100, 30),
            (0, 100, 30),
            (0, 0, 30),
        ]

    def _target_pos_m(self):
        return self._waypoints[self._wp_idx]

    def step(self):
        s = self.state
        tx, ty, tz = self._target_pos_m()

        # current position in local meters relative to home
        cx = (s.lon - HOME_LON) * (EARTH_RADIUS_M * np.cos(np.radians(HOME_LAT))) * (np.pi / 180)
        cy = (s.lat - HOME_LAT) * EARTH_RADIUS_M * (np.pi / 180)
        cz = s.alt

        dx, dy, dz = tx - cx, ty - cy, tz - cz
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist < 3.0:
            self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)

        # simple proportional velocity controller toward waypoint
        if dist > 1e-6:
            dirx, diry, dirz = dx / dist, dy / dist, dz / dist
        else:
            dirx = diry = dirz = 0.0

        target_vx = dirx * self.cruise_speed
        target_vy = diry * self.cruise_speed
        target_vz = dirz * self.cruise_speed

        # smooth acceleration toward target velocity (simple 1st-order lag)
        tau = 1.2
        s.ax = (target_vx - s.vx) / tau
        s.ay = (target_vy - s.vy) / tau
        s.az = (target_vz - s.vz) / tau

        # small process noise (wind gusts, motor jitter)
        s.ax += self.rng.normal(0, 0.05)
        s.ay += self.rng.normal(0, 0.05)
        s.az += self.rng.normal(0, 0.05)

        s.vx += s.ax * self.dt
        s.vy += s.ay * self.dt
        s.vz += s.az * self.dt

        new_cx = cx + s.vx * self.dt
        new_cy = cy + s.vy * self.dt
        s.alt = max(0.0, cz + s.vz * self.dt)

        dlat, dlon = meters_to_latlon_delta(new_cx - cx, new_cy - cy, s.lat)
        s.lat += dlat
        s.lon += dlon

        s.heading = (np.degrees(np.arctan2(s.vx, s.vy)) + 360) % 360
        s.yaw = np.radians(s.heading)
        speed_xy = np.sqrt(s.vx**2 + s.vy**2)
        s.pitch = np.clip(np.arctan2(-s.az, 9.81), -0.5, 0.5) * 0.3
        s.roll = np.clip(np.arctan2(speed_xy, 9.81), -0.5, 0.5) * 0.1

        s.t += self.dt
        return s

    def generate(self, duration_s):
        n = int(duration_s / self.dt)
        records = []
        for _ in range(n):
            s = self.step()
            records.append({
                "t": round(s.t, 2),
                "lat": s.lat, "lon": s.lon, "alt": s.alt,
                "vx": s.vx, "vy": s.vy, "vz": s.vz,
                "ax": s.ax, "ay": s.ay, "az": s.az,
                "roll": s.roll, "pitch": s.pitch, "yaw": s.yaw,
                "heading": s.heading,
                "label": "normal", "attack_type": "none",
            })
        return records


if __name__ == "__main__":
    sim = NormalFlightSimulator()
    data = sim.generate(30)
    print(f"Generated {len(data)} records")
    print(data[0])
    print(data[-1])
