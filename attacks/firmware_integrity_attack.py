"""
firmware_integrity_attack.py  (TC-06)

Simulates a firmware-integrity violation: the drone's firmware/parameter
set is tampered with, detected by comparing a cryptographic hash of the
firmware/config against a trusted baseline hash.

This is architecturally DIFFERENT from the other 5 attacks — it's not a
telemetry/communication-stream anomaly, it's a periodic INTEGRITY CHECK
(the blueprint calls this out as "a dedicated integrity mechanism rather
than merely another MAVLink anomaly"). So instead of per-message
injection, this module simulates periodic firmware-hash attestation
reports, most of which match the trusted baseline, with a tampering
event injected during the attack window.

Real-world analogue: PX4/ArduPilot expose parameter checksums and a
build-time firmware hash; a production implementation would read these
via MAVLink PARAM/FILE_TRANSFER or a companion-computer attestation
service. Here we simulate the attestation REPORT stream directly, since
that is the signal our detector actually consumes.
"""

import hashlib
import numpy as np
from base_attack import BaseAttack


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


TRUSTED_FIRMWARE_ID = "px4-v1.14.0-stable-build-a1b2c3"
TRUSTED_HASH = _hash(TRUSTED_FIRMWARE_ID)

TAMPERED_FIRMWARE_ID = "px4-v1.14.0-stable-build-a1b2c3-PATCHED"
TAMPERED_HASH = _hash(TAMPERED_FIRMWARE_ID)


class FirmwareIntegrityAttack(BaseAttack):
    name = "firmware_integrity"

    def __init__(self, start_t, duration, rng, attestation_interval_s=1.0):
        super().__init__(start_t, duration, rng)
        self.attestation_interval_s = attestation_interval_s

    def apply(self, records):
        return records  # firmware attestation is a separate event stream

    def generate_attestation_stream(self, total_duration_s):
        """
        Yields periodic attestation events:
          {t, reported_hash, trusted_hash, match, label, attack_type}
        Independent of the flight-physics stream — firmware doesn't care
        where the drone is — but shares the same timeline for correlation
        in the fusion layer.
        """
        t = 0.0
        events = []
        while t < total_duration_s:
            attacked_now = self.is_active(t)
            reported_hash = TAMPERED_HASH if attacked_now else TRUSTED_HASH
            events.append({
                "t": round(t, 2),
                "reported_hash": reported_hash,
                "trusted_hash": TRUSTED_HASH,
                "match": reported_hash == TRUSTED_HASH,
                "label": "attack" if attacked_now else "normal",
                "attack_type": self.name if attacked_now else "none",
            })
            t += self.attestation_interval_s
        return events


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    atk = FirmwareIntegrityAttack(start_t=8.0, duration=4.0, rng=rng, attestation_interval_s=1.0)
    events = atk.generate_attestation_stream(total_duration_s=20.0)

    for e in events:
        status = "OK" if e["match"] else "MISMATCH (ALERT)"
        print(f"t={e['t']:>5.1f}s  hash={e['reported_hash'][:12]}...  {status}")

    mismatches = sum(1 for e in events if not e["match"])
    print(f"\nTotal attestations: {len(events)}, mismatches detected: {mismatches}")
