"""
dos_attack.py  (TC-03)

Simulates a Denial-of-Service attack against the telemetry link: during
the attack window, HEARTBEAT (and optionally other) messages are dropped
or the effective packet rate is drastically reduced, mimicking RF jamming
or a flooding attack that starves the link.

Unlike GPS spoofing / command injection (which add/alter content), DoS is
a STREAM-LEVEL attack: it operates on the already-encoded MAVLink message
list, dropping messages rather than modifying flight records.

Detection signal: heartbeat gap duration, overall message-rate drop.
"""

import numpy as np
from base_attack import BaseAttack


class DoSAttack(BaseAttack):
    name = "dos"

    def __init__(self, start_t, duration, rng, drop_probability=0.9):
        """
        drop_probability: fraction of messages dropped during the attack
                           window (0.9 = severe DoS, link mostly dead)
        """
        super().__init__(start_t, duration, rng)
        self.drop_probability = drop_probability

    def apply(self, records):
        """DoS operates on the MAVLink stream, not raw records — see apply_to_stream()."""
        return records

    def apply_to_stream(self, mavlink_stream):
        """
        mavlink_stream: iterable of (t_us, msg_type, msg, gt_rec)
        Yields the same tuples, but drops messages during the attack
        window with probability `drop_probability`, and relabels the
        surviving gt_rec so the ground truth reflects the attack.
        """
        boot_t_us = None
        for t_us, msg_type, msg, gt_rec in mavlink_stream:
            if boot_t_us is None:
                boot_t_us = t_us
            t_s = (t_us - boot_t_us) / 1e6

            gt_rec = dict(gt_rec)
            if self.is_active(t_s):
                gt_rec["label"] = "attack"
                gt_rec["attack_type"] = self.name
                if self.rng.random() < self.drop_probability:
                    continue  # message dropped — DoS in action
            yield t_us, msg_type, msg, gt_rec


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from flight_physics import NormalFlightSimulator
    from mavlink_encoder import encode_stream

    sim = NormalFlightSimulator()
    records = sim.generate(20)
    rng = np.random.default_rng(3)
    atk = DoSAttack(start_t=8.0, duration=4.0, rng=rng, drop_probability=0.9)

    stream = encode_stream(records)
    dosed_stream = atk.apply_to_stream(stream)

    total, during_attack, survived_during_attack = 0, 0, 0
    for t_us, msg_type, msg, gt_rec in dosed_stream:
        total += 1
        if gt_rec["attack_type"] == "dos":
            during_attack += 1
            survived_during_attack += 1

    print(f"Total messages delivered: {total}")
    print(f"Messages delivered during attack window: {survived_during_attack} "
          f"(heavy drop expected)")
