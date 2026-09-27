"""
build_dataset.py

Orchestrates the full Stage-1 pipeline end to end:

  NormalFlightSimulator
        |
        v
  [GPS Spoofing] -> [Command Injection] -> [Telemetry Manipulation]   (record-level)
        |
        v
  MAVLink encode_stream()
        |
        v
  [DoS] -> [MAVLink Anomaly]                                          (stream-level)
        |
        v
  StreamCapture.finalize()  -> unified DataFrame
        |
        v
  build_feature_matrix()    -> feature-engineered DataFrame
        |
        v
  + Firmware Integrity attestation log (separate table, joined by time)

All 6 attacks fire in non-overlapping windows across one long flight, so
each attack's ground-truth label is unambiguous. Injected COMMAND_LONG
events (from command injection) are logged separately since they aren't
part of the per-timestep telemetry schema.

Output: logs/labeled_dataset.csv, logs/injected_commands.csv,
        logs/firmware_attestation.csv
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sim"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "capture"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "features"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "attacks"))

from flight_physics import NormalFlightSimulator
from mavlink_encoder import encode_stream
from capture import capture_to_dataframe
from feature_engine import build_feature_matrix

from gps_spoofing_attack import GPSSpoofingAttack
from command_injection_attack import CommandInjectionAttack
from telemetry_manipulation_attack import TelemetryManipulationAttack
from dos_attack import DoSAttack
from mavlink_anomaly_attack import MAVLinkAnomalyAttack
from firmware_integrity_attack import FirmwareIntegrityAttack

FLIGHT_DURATION_S = 90
SEED = 123

# non-overlapping attack windows: (start_t, duration)
WINDOWS = {
    "gps_spoofing":            (10.0, 5.0),
    "command_injection":       (20.0, 5.0),
    "telemetry_manipulation":  (30.0, 5.0),
    "dos":                     (45.0, 5.0),
    "mavlink_anomaly":         (55.0, 5.0),
    "firmware_integrity":      (65.0, 5.0),
}


def build():
    rng = np.random.default_rng(SEED)

    sim = NormalFlightSimulator(seed=SEED)
    records = sim.generate(FLIGHT_DURATION_S)

    # --- record-level attacks ---
    gps_atk = GPSSpoofingAttack(*WINDOWS["gps_spoofing"], rng=rng, profile="jump", magnitude_m=40.0)
    records = gps_atk.apply(records)

    cmd_atk = CommandInjectionAttack(*WINDOWS["command_injection"], rng=rng, injection_rate_hz=2.0)
    records, injected_commands = cmd_atk.apply(records)

    tel_atk = TelemetryManipulationAttack(*WINDOWS["telemetry_manipulation"], rng=rng)
    records = tel_atk.apply(records)

    # --- encode to MAVLink ---
    stream = encode_stream(records)

    # --- stream-level attacks ---
    dos_atk = DoSAttack(*WINDOWS["dos"], rng=rng, drop_probability=0.9)
    stream = dos_atk.apply_to_stream(stream)

    mav_atk = MAVLinkAnomalyAttack(*WINDOWS["mavlink_anomaly"], rng=rng, corruption_probability=0.35)
    stream = mav_atk.apply_to_stream(stream)

    # --- capture + normalize ---
    df = capture_to_dataframe(stream)

    # --- features ---
    feat_df, feature_cols = build_feature_matrix(df)

    # --- firmware integrity (separate attestation stream) ---
    fw_atk = FirmwareIntegrityAttack(*WINDOWS["firmware_integrity"], rng=rng, attestation_interval_s=1.0)
    attestations = fw_atk.generate_attestation_stream(total_duration_s=FLIGHT_DURATION_S)
    fw_df = pd.DataFrame(attestations)

    return feat_df, feature_cols, pd.DataFrame(injected_commands), fw_df


if __name__ == "__main__":
    feat_df, feature_cols, cmd_df, fw_df = build()

    os.makedirs("logs", exist_ok=True)
    feat_df.to_csv("logs/labeled_dataset.csv", index=False)
    cmd_df.to_csv("logs/injected_commands.csv", index=False)
    fw_df.to_csv("logs/firmware_attestation.csv", index=False)

    print("=== Dataset built ===")
    print(f"Main dataset: {feat_df.shape[0]} rows, {feat_df.shape[1]} cols")
    print(f"\nLabel distribution:")
    print(feat_df["gt_label"].value_counts())
    print(f"\nAttack-type distribution:")
    print(feat_df["gt_attack_type"].value_counts())
    print(f"\nInjected commands: {len(cmd_df)}")
    print(f"Firmware attestations: {len(fw_df)}  (mismatches: {(~fw_df['match']).sum()})")
    print(f"\nSaved -> logs/labeled_dataset.csv, logs/injected_commands.csv, logs/firmware_attestation.csv")
