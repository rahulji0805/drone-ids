"""
telemetry_manipulation_attack.py  (TC-04)

Simulates telemetry manipulation: unlike GPS spoofing (which corrupts
position), this attack tampers with ALTITUDE / VELOCITY / ATTITUDE fields
reported over MAVLink — e.g. an attacker on the link modifies
GLOBAL_POSITION_INT's altitude/velocity or ATTITUDE's roll/pitch/yaw,
while the GPS position itself stays correct.

This is the mirror-image inconsistency of GPS spoofing: position says
"still here" but reported vertical speed / attitude imply violent motion
that never actually happened (or vice versa — reported telemetry is
suspiciously smooth/static while position is changing).

We manipulate the FUSED telemetry fields (gpos_alt, gpos_vz, att_roll/pitch)
rather than the raw GPS, distinguishing this attack class from TC-01 at
the detector level: TC-01 shows raw-GPS vs IMU mismatch, TC-04 shows
fused-telemetry vs raw-GPS mismatch.
"""

import numpy as np
from base_attack import BaseAttack


class TelemetryManipulationAttack(BaseAttack):
    name = "telemetry_manipulation"

    def __init__(self, start_t, duration, rng, alt_offset_m=15.0, vz_fake=-3.0):
        """
        alt_offset_m: how much the reported altitude is falsified
        vz_fake     : a fake vertical-speed value reported (m/s) that is
                      inconsistent with the (unaltered) real position trend
        """
        super().__init__(start_t, duration, rng)
        self.alt_offset_m = alt_offset_m
        self.vz_fake = vz_fake

    def apply(self, records):
        out = []
        for rec in records:
            r = dict(rec)
            r["telemetry_alt"] = r["alt"]     # what GLOBAL_POSITION_INT will report
            r["telemetry_vz"] = r["vz"]
            r["telemetry_roll"] = r["roll"]

            if self.is_active(r["t"]):
                elapsed = r["t"] - self.start_t
                ramp = min(1.0, elapsed / 1.0)  # 1s ramp-in, avoids an instant discontinuity
                r["telemetry_alt"] = r["alt"] + self.alt_offset_m * ramp \
                    + self.rng.normal(0, 0.3)
                r["telemetry_vz"] = self.vz_fake * ramp + self.rng.normal(0, 0.1)
                r["telemetry_roll"] = r["roll"] + self.rng.normal(0, 0.15)

                r["label"] = "attack"
                r["attack_type"] = self.name
            out.append(r)
        return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from flight_physics import NormalFlightSimulator

    sim = NormalFlightSimulator()
    records = sim.generate(20)
    rng = np.random.default_rng(4)
    atk = TelemetryManipulationAttack(start_t=8.0, duration=4.0, rng=rng)
    out = atk.apply(records)

    attacked = [r for r in out if r["label"] == "attack"]
    print(f"Attack-labeled records: {len(attacked)} / {len(out)}")
    sample = attacked[-1]
    print(f"True alt: {sample['alt']:.2f}  Reported (fake) alt: {sample['telemetry_alt']:.2f}")
    print(f"True vz : {sample['vz']:.2f}  Reported (fake) vz : {sample['telemetry_vz']:.2f}")
