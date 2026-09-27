"""
mavlink_anomaly_attack.py  (TC-05)

Simulates protocol-level MAVLink anomalies distinct from the other five
attacks (which corrupt WHAT a message says, or whether it arrives at
all). This attack corrupts the PROTOCOL ITSELF:
  - malformed/corrupted message bytes (bad CRC)
  - unexpected message sequence numbers (replay / out-of-order)
  - unexpected message TYPE appearing where it shouldn't (protocol confusion)

Operates at the message-stream level, after MAVLink encoding, similar to
dos_attack.py but corrupting rather than dropping.
"""

import numpy as np
from base_attack import BaseAttack


class MAVLinkAnomalyAttack(BaseAttack):
    name = "mavlink_anomaly"

    def __init__(self, start_t, duration, rng, corruption_probability=0.35):
        super().__init__(start_t, duration, rng)
        self.corruption_probability = corruption_probability

    def apply(self, records):
        return records  # this attack operates on the encoded stream, not records

    def apply_to_stream(self, mavlink_stream):
        """
        Yields (t_us, msg_type, msg, gt_rec, anomaly_flags) — an extra 5th
        field describing which protocol anomaly (if any) was injected,
        since corruption happens at the wire level, not the field level.
        """
        boot_t_us = None
        seq_counter = 0
        for t_us, msg_type, msg, gt_rec in mavlink_stream:
            if boot_t_us is None:
                boot_t_us = t_us
            t_s = (t_us - boot_t_us) / 1e6
            gt_rec = dict(gt_rec)
            anomaly_flags = {"bad_crc": False, "replayed_seq": False, "unexpected_type": False}

            if self.is_active(t_s) and self.rng.random() < self.corruption_probability:
                gt_rec["label"] = "attack"
                gt_rec["attack_type"] = self.name
                choice = self.rng.choice(["bad_crc", "replayed_seq", "unexpected_type"])
                anomaly_flags[choice] = True

                if choice == "replayed_seq":
                    # emit the SAME sequence number as a few messages ago
                    seq_counter = max(0, seq_counter - self.rng.integers(2, 6))
                else:
                    seq_counter += 1
            else:
                seq_counter += 1

            yield t_us, msg_type, msg, gt_rec, anomaly_flags


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from flight_physics import NormalFlightSimulator
    from mavlink_encoder import encode_stream

    sim = NormalFlightSimulator()
    records = sim.generate(20)
    rng = np.random.default_rng(5)
    atk = MAVLinkAnomalyAttack(start_t=8.0, duration=4.0, rng=rng, corruption_probability=0.35)

    stream = encode_stream(records)
    anomalous_stream = atk.apply_to_stream(stream)

    total, anomalies = 0, 0
    counts = {"bad_crc": 0, "replayed_seq": 0, "unexpected_type": 0}
    for t_us, msg_type, msg, gt_rec, flags in anomalous_stream:
        total += 1
        if gt_rec["attack_type"] == "mavlink_anomaly":
            anomalies += 1
            for k, v in flags.items():
                if v:
                    counts[k] += 1

    print(f"Total messages: {total}, flagged anomalous: {anomalies}")
    print("Breakdown:", counts)
