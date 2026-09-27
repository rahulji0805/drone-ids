"""
feature_engine.py

Turns the unified capture DataFrame into the FEATURES our detectors
actually reason about. This is where the "cyber-physical correlation"
differentiator lives: features aren't raw fields, they're CROSS-LAYER
CONSISTENCY CHECKS between independent data sources.

Feature groups (matches the blueprint's 5-layer architecture):
  1. Protocol features      — heartbeat gaps, message-rate stats
  2. Temporal features      — inter-arrival jitter, sequence continuity
  3. Navigation consistency — GPS position vs IMU-integrated motion
  4. Telemetry consistency  — fused telemetry vs raw GPS/attitude
  5. System integrity       — firmware attestation (handled separately,
                               see firmware_check.py — kept out of the
                               per-timestep feature matrix on purpose,
                               it's a periodic event, not a continuous signal)

NOTE: gt_label / gt_attack_type are ground truth for EVALUATION ONLY.
They are explicitly dropped before anything is handed to a detector.
"""

import numpy as np
import pandas as pd


GROUND_TRUTH_COLUMNS = ["gt_label", "gt_attack_type"]


def add_navigation_consistency_features(df):
    """
    Core anti-GPS-spoofing feature: does GPS-reported motion match
    IMU/attitude-implied motion?

    We don't have raw IMU accel history from the unified schema directly
    (attitude only), so we approximate "expected motion" from the FUSED
    GLOBAL_POSITION_INT velocity (gpos_vx/vy) — which in a real system
    comes from the flight-control's IMU-based state estimator, distinct
    from the raw GPS_RAW_INT position we're checking against.

    Two complementary signals are computed:
      - nav_disp_mismatch_m : FRAME-TO-FRAME delta mismatch. Catches
        sudden jumps/discontinuities, but is blind to a constant-offset
        spoof once it has "settled" (offset stays fixed -> deltas look
        normal again).
      - nav_absolute_divergence_m : CUMULATIVE dead-reckoning divergence.
        We integrate velocity independently from t=0 to get a
        GPS-independent position estimate, then compare it to raw GPS
        position directly. A constant-offset spoof stays visible for the
        entire attack window here, not just at the transition edges.
    """
    df = df.copy()
    dt = df["t_us"].diff().fillna(50_000) / 1e6  # seconds

    # --- frame-to-frame mismatch (catches jump transitions) ---
    expected_dx = df["gpos_vx"] * dt
    expected_dy = df["gpos_vy"] * dt
    expected_disp = np.sqrt(expected_dx**2 + expected_dy**2)

    R = 6371000.0
    dlat = df["gps_lat"].diff().fillna(0) * (np.pi / 180) * R
    dlon = (df["gps_lon"].diff().fillna(0) * (np.pi / 180)
            * R * np.cos(np.radians(df["gps_lat"])))
    actual_disp = np.sqrt(dlat**2 + dlon**2)

    df["nav_disp_mismatch_m"] = (actual_disp - expected_disp).abs()
    df["nav_disp_mismatch_ratio"] = df["nav_disp_mismatch_m"] / (expected_disp + 1e-3)
    df["nav_raw_jump_m"] = actual_disp

    # --- raw GPS vs fused-position mismatch (PRIMARY spoofing signal) ---
    # Directly compares the raw GPS_RAW_INT position against the flight
    # controller's independently-fused GLOBAL_POSITION_INT position, at
    # the SAME instant. This is the standard real-world approach (raw GPS
    # vs EKF/fused estimate) and is far more robust than differencing or
    # integrating: it needs no velocity integration (so it's immune to
    # the physics-model turn-transient issue that broke earlier
    # cumulative/windowed divergence attempts), and it's non-zero for
    # EVERY row of a sustained spoof (not just sporadic jitter-driven
    # frame-to-frame crossings), giving reliable full-window coverage.
    dlat_gf = (df["gps_lat"] - df["gpos_lat"]) * (np.pi / 180) * R
    dlon_gf = ((df["gps_lon"] - df["gpos_lon"]) * (np.pi / 180)
               * R * np.cos(np.radians(df["gps_lat"])))
    df["nav_gps_vs_fused_mismatch_m"] = np.sqrt(dlat_gf**2 + dlon_gf**2)
    return df


def add_telemetry_consistency_features(df):
    """
    Fused-altitude vs raw-GPS-altitude consistency (catches TC-04).
    """
    df = df.copy()
    df["telem_alt_mismatch_m"] = (df["gpos_alt"] - df["gps_alt"]).abs()
    # vertical-speed vs actual altitude-change consistency
    dt = df["t_us"].diff().fillna(50_000) / 1e6
    implied_vz = df["gpos_alt"].diff().fillna(0) / dt.replace(0, np.nan)
    df["telem_vz_mismatch"] = (df["gpos_vz"] - implied_vz.fillna(0)).abs()
    return df


def add_protocol_temporal_features(df):
    """
    Heartbeat-gap based DoS signal + time-windowed message-rate density.
    """
    df = df.copy()
    df["hb_gap_s"] = df["t_us"].diff().fillna(0) / 1e6

    # TIME-based message count in the trailing 1.0s (not a fixed sample
    # count): a fixed-sample rolling window (e.g. last 10 rows) spans a
    # variable amount of REAL time when the arrival rate itself is
    # degraded (exactly the condition we're trying to detect), causing a
    # slow "catch-up" lag right when we need fast reaction. A time-based
    # window doesn't have this lag — it directly answers "how many
    # messages arrived in the last real second", regardless of how
    # sparse they were.
    t_s = df["t_us"].values / 1e6
    df["flight_elapsed_s"] = t_s - t_s[0]
    window_start = t_s - 1.0
    j_idx = np.searchsorted(t_s, window_start, side="left")
    row_idx = np.arange(len(df))
    df["msg_count_1s"] = row_idx - j_idx + 1  # inclusive count in [t-1s, t]

    # SHORT window (0.3s) — reacts ~3x faster to a rate drop at attack
    # onset than the 1.0s window, which stays "stale" (full of pre-attack
    # traffic) for up to 1s after an attack starts. Used alongside the
    # long window so onset detection isn't bottlenecked by window length.
    short_start = t_s - 0.3
    j_idx_short = np.searchsorted(t_s, short_start, side="left")
    df["msg_count_0_3s"] = row_idx - j_idx_short + 1

    # kept for reference/comparison; not used as the primary DoS signal
    df["msg_rate_rolling"] = 1.0 / df["hb_gap_s"].replace(0, np.nan).rolling(10, min_periods=1).mean()
    df["msg_rate_rolling"] = df["msg_rate_rolling"].fillna(0)
    return df


def build_feature_matrix(df):
    """
    Full pipeline: raw unified capture -> feature-engineered DataFrame,
    with ground truth columns preserved but clearly separated (for eval),
    and a `feature_cols` list returned for detectors that need to know
    exactly which columns are safe to consume.
    """
    df = add_navigation_consistency_features(df)
    df = add_telemetry_consistency_features(df)
    df = add_protocol_temporal_features(df)

    feature_cols = [
        "nav_disp_mismatch_m", "nav_disp_mismatch_ratio", "nav_raw_jump_m",
        "nav_gps_vs_fused_mismatch_m",
        "telem_alt_mismatch_m", "telem_vz_mismatch",
        "hb_gap_s", "msg_rate_rolling", "msg_count_1s", "msg_count_0_3s", "flight_elapsed_s",
        "anomaly_bad_crc", "anomaly_replayed_seq", "anomaly_unexpected_type",
        "att_roll", "att_pitch", "att_yaw",
        "gps_sats", "gps_fix_type",
    ]
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))
    from flight_physics import NormalFlightSimulator
    from mavlink_encoder import encode_stream
    from capture import capture_to_dataframe

    sim = NormalFlightSimulator()
    records = sim.generate(30)
    df = capture_to_dataframe(encode_stream(records))

    feat_df, feature_cols = build_feature_matrix(df)
    print(f"Feature matrix shape: {feat_df[feature_cols].shape}")
    print(f"\nFeature columns: {feature_cols}")
    print("\nSample stats (should be near-zero for normal flight):")
    print(feat_df[feature_cols].describe().T[["mean", "std", "max"]])
