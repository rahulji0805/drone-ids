"""
capture.py

Consumes the raw MAVLink message stream (from sim/mavlink_encoder.py, or
later a real PX4/ArduPilot SITL mavlink connection) and normalizes it into
ONE unified per-timestep schema, regardless of which messages arrived.

This is the boundary between "protocol layer" and "analysis layer" —
everything downstream (features, detectors) reads this clean schema,
so swapping synthetic sim -> real SITL later requires NO changes to
the detection engine. That reusability is itself worth mentioning in
the proposal's architecture section.
"""

import pandas as pd


UNIFIED_COLUMNS = [
    "t_us", "seq",
    "hb_present", "hb_system_status",
    "gpos_lat", "gpos_lon", "gpos_alt", "gpos_vx", "gpos_vy", "gpos_vz", "gpos_hdg",
    "gps_fix_type", "gps_lat", "gps_lon", "gps_alt", "gps_vel", "gps_sats",
    "att_roll", "att_pitch", "att_yaw",
    "sys_load", "sys_batt_v", "sys_batt_remaining",
    "anomaly_bad_crc", "anomaly_replayed_seq", "anomaly_unexpected_type",
    "gt_label", "gt_attack_type",   # ground truth, for training/eval only —
                                      # NEVER fed to the detector as a feature
]


class StreamCapture:
    """
    Buffers incoming MAVLink messages and, whenever a full 'set' for a
    timestep has arrived, flushes a unified row. Real telemetry doesn't
    arrive in lockstep, so we group by nearest timestamp bucket.
    """

    def __init__(self, bucket_us=50_000):
        self.bucket_us = bucket_us
        self._buffers = {}   # bucket_ts -> dict of fields
        self.rows = []
        self._seq = 0

    def _bucket(self, t_us):
        return t_us - (t_us % self.bucket_us)

    def ingest(self, t_us, msg_type, msg, gt_rec, anomaly_flags=None):
        b = self._bucket(t_us)
        row = self._buffers.setdefault(b, {"t_us": b, "gt_label": gt_rec["label"],
                                            "gt_attack_type": gt_rec["attack_type"]})
        # a later message in the same bucket can carry a worse ground-truth
        # label (e.g. an attack window starting mid-bucket) — keep 'attack'
        # if any message in this bucket was attack-labeled
        if gt_rec["label"] == "attack":
            row["gt_label"] = "attack"
            row["gt_attack_type"] = gt_rec["attack_type"]

        if anomaly_flags:
            if anomaly_flags.get("bad_crc"):
                row["anomaly_bad_crc"] = row.get("anomaly_bad_crc", 0) + 1
            if anomaly_flags.get("replayed_seq"):
                row["anomaly_replayed_seq"] = row.get("anomaly_replayed_seq", 0) + 1
            if anomaly_flags.get("unexpected_type"):
                row["anomaly_unexpected_type"] = row.get("anomaly_unexpected_type", 0) + 1

        d = msg.to_dict()

        if msg_type == "HEARTBEAT":
            row["hb_present"] = 1
            row["hb_system_status"] = d.get("system_status")
        elif msg_type == "GLOBAL_POSITION_INT":
            row["gpos_lat"] = d["lat"] / 1e7
            row["gpos_lon"] = d["lon"] / 1e7
            row["gpos_alt"] = d["alt"] / 1000.0
            row["gpos_vx"] = d["vx"] / 100.0
            row["gpos_vy"] = d["vy"] / 100.0
            row["gpos_vz"] = d["vz"] / 100.0
            row["gpos_hdg"] = d["hdg"] / 100.0
        elif msg_type == "GPS_RAW_INT":
            row["gps_fix_type"] = d["fix_type"]
            row["gps_lat"] = d["lat"] / 1e7
            row["gps_lon"] = d["lon"] / 1e7
            row["gps_alt"] = d["alt"] / 1000.0
            row["gps_vel"] = d["vel"] / 100.0
            row["gps_sats"] = d["satellites_visible"]
        elif msg_type == "ATTITUDE":
            row["att_roll"] = d["roll"]
            row["att_pitch"] = d["pitch"]
            row["att_yaw"] = d["yaw"]
        elif msg_type == "SYS_STATUS":
            row["sys_load"] = d["load"] / 10.0  # percent
            row["sys_batt_v"] = d["voltage_battery"] / 1000.0
            row["sys_batt_remaining"] = d["battery_remaining"]

    def finalize(self):
        """Flush all buckets into a sorted, gap-filled DataFrame."""
        for b in sorted(self._buffers):
            r = self._buffers[b]
            r["seq"] = self._seq
            self._seq += 1
            r.setdefault("hb_present", 0)
            r.setdefault("anomaly_bad_crc", 0)
            r.setdefault("anomaly_replayed_seq", 0)
            r.setdefault("anomaly_unexpected_type", 0)
            self.rows.append(r)

        df = pd.DataFrame(self.rows)
        for col in UNIFIED_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[UNIFIED_COLUMNS].sort_values("t_us").reset_index(drop=True)

        # forward-fill genuinely missing fields (a message not arriving in
        # a given bucket isn't itself an anomaly at this layer — DoS
        # detection happens explicitly downstream via heartbeat-gap timing)
        no_fill = ("t_us", "seq", "gt_label", "gt_attack_type", "hb_present",
                   "anomaly_bad_crc", "anomaly_replayed_seq", "anomaly_unexpected_type")
        fill_cols = [c for c in UNIFIED_COLUMNS if c not in no_fill]
        df[fill_cols] = df[fill_cols].ffill().bfill()
        return df


def capture_to_dataframe(mavlink_stream_iter):
    cap = StreamCapture()
    for item in mavlink_stream_iter:
        if len(item) == 5:
            t_us, msg_type, msg, gt_rec, anomaly_flags = item
        else:
            t_us, msg_type, msg, gt_rec = item
            anomaly_flags = None
        cap.ingest(t_us, msg_type, msg, gt_rec, anomaly_flags)
    return cap.finalize()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from flight_physics import NormalFlightSimulator
    from mavlink_encoder import encode_stream

    sim = NormalFlightSimulator()
    records = sim.generate(30)
    df = capture_to_dataframe(encode_stream(records))

    print(df.shape)
    print(df.head())
    print("\nColumns with any missing values:")
    print(df.isna().sum()[df.isna().sum() > 0])

    df.to_csv("../logs/normal_flight_capture.csv", index=False)
    print("\nSaved -> logs/normal_flight_capture.csv")
