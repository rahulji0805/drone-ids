"""
gps_spoofing_attack.py  (TC-01)

Simulates GPS spoofing: the GPS-reported position drifts/jumps away from
the drone's TRUE position, while IMU-derived velocity/attitude stay
correct (because a GPS spoofer only controls the RF signal into the GPS
receiver, not the drone's actual inertial sensors).

This is exactly the cross-layer inconsistency our detector exploits:
GPS says "moved 40m in 1s" but IMU-integrated velocity says "barely moved".

Two spoofing profiles are supported:
  - "jump"  : sudden large position discontinuity (easy case)
  - "drift" : slow, steadily-growing offset (harder, more realistic)

IMPORTANT: this injector adds two NEW fields (spoofed_lat/spoofed_lon)
rather than overwriting lat/lon. True position is preserved as ground
truth for IMU-consistency checks; the MAVLink encoder is later told to
transmit the SPOOFED position on the GPS_RAW_INT message only (real GPS
spoofing attacks the raw receiver, and a well-designed autopilot's fused
GLOBAL_POSITION_INT may partially resist it — that nuance is itself worth
a line in the proposal).
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from flight_physics import meters_to_latlon_delta

from base_attack import BaseAttack


class GPSSpoofingAttack(BaseAttack):
    name = "gps_spoofing"

    def __init__(self, start_t, duration, rng, profile="jump", magnitude_m=40.0):
        super().__init__(start_t, duration, rng)
        self.profile = profile
        self.magnitude_m = magnitude_m

    def apply(self, records):
        out = []
        RAMP_S = 0.4  # smooth in/out over 0.4s — avoids a hard discontinuity
                       # at the attack boundary that would otherwise show up
                       # as a large single-frame delta on the POST-attack
                       # 'normal' side (a false-positive source we found
                       # empirically, not just a cosmetic concern)
        for rec in records:
            r = dict(rec)
            r["spoofed_lat"] = r["lat"]
            r["spoofed_lon"] = r["lon"]

            if self.is_active(r["t"]):
                elapsed = r["t"] - self.start_t
                remaining = (self.start_t + self.duration) - r["t"]

                if self.profile == "jump":
                    ramp_in = min(1.0, elapsed / RAMP_S)
                    ramp_out = min(1.0, remaining / RAMP_S)
                    ramp = min(ramp_in, ramp_out)
                    offset_m = self.magnitude_m * ramp
                else:  # drift — ramps in across the whole window, no ramp-out
                       # needed (already near start-of-window offset by design,
                       # and drift profile isn't used in current test scenarios)
                    offset_m = self.magnitude_m * min(1.0, elapsed / max(self.duration, 1e-6))

                dx = offset_m + self.rng.normal(0, 1.5)
                dy = self.rng.normal(0, 1.5)
                dlat, dlon = meters_to_latlon_delta(dx, dy, r["lat"])
                r["spoofed_lat"] = r["lat"] + dlat
                r["spoofed_lon"] = r["lon"] + dlon

                r["label"] = "attack"
                r["attack_type"] = self.name
            out.append(r)
        return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from flight_physics import NormalFlightSimulator

    sim = NormalFlightSimulator()
    records = sim.generate(20)
    rng = np.random.default_rng(1)
    atk = GPSSpoofingAttack(start_t=8.0, duration=4.0, rng=rng, profile="jump", magnitude_m=40.0)
    out = atk.apply(records)

    attacked = [r for r in out if r["label"] == "attack"]
    print(f"Attack-labeled records: {len(attacked)} / {len(out)}")
    sample = attacked[len(attacked) // 2]
    print("True lat/lon: ", sample["lat"], sample["lon"])
    print("Spoofed lat/lon:", sample["spoofed_lat"], sample["spoofed_lon"])
