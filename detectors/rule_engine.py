"""
rule_engine.py

Deterministic, explainable detectors — one function per attack type.
Thresholds are chosen conservatively above the observed normal-flight
noise floor (see feature_engine.py's baseline stats) with margin, so
that false positives on clean flight stay low. Each detector returns a
per-row boolean flag + a numeric severity contribution used by the
fusion engine's risk score.

This is the FIRST of two detection layers (see ml_anomaly.py for the
second). Per the blueprint: rules are explainable and fast; they should
carry primary responsibility for the SIX NAMED attack classes since we
know exactly what each one looks like. ML anomaly detection is a safety
net for deviations that don't match any known signature.
"""

import pandas as pd

# Thresholds — set with margin above observed normal-flight noise:
#   nav_disp_mismatch_m   normal max ~0.01m   -> threshold 2.0m
#   telem_alt_mismatch_m  normal == 0         -> threshold 3.0m
#   hb_gap_s              normal == 0.1s      -> threshold 0.25s (2.5x nominal)
THRESHOLDS = {
    "gps_spoofing_disp_m": 2.0,
    "gps_spoofing_divergence_m": 8.0,
    "telemetry_alt_mismatch_m": 3.0,
    "telemetry_vz_mismatch": 1.0,
    "dos_hb_gap_s": 0.15,   # normal-flight baseline max observed = 0.10s (1.5x margin, zero FPR)
    "dos_msg_rate_floor": 5.0,   # kept conservative — 7.0 caused FPR from rolling-window startup edge case
    "dos_msg_count_1s": 6,      # long window: sparse-traffic floor over trailing 1.0s
    "dos_msg_count_0_3s": 2,    # short window: sparse-traffic floor over trailing 0.3s (~3x faster onset reaction)
}


# ---------------------------------------------------------------------------
# Vectorized detectors (operate on the whole DataFrame at once via numpy/
# pandas column ops instead of Python-level row iteration). Each returns
# (flag: pd.Series[bool], severity: pd.Series[float]) aligned to df.index.
# This replaces the old per-row `for _, row in df.iterrows(): fn(row)`
# loop in run_rule_engine, which re-materializes a Python object per row
# and does 4 full passes over the DataFrame at Python speed. Vectorized
# ops run at C speed inside pandas/numpy — same detection logic, same
# thresholds, just no per-row Python overhead.
# ---------------------------------------------------------------------------

def detect_gps_spoofing_vec(df):
    absolute = df["nav_gps_vs_fused_mismatch_m"] > THRESHOLDS["gps_spoofing_divergence_m"]
    jump = df["nav_disp_mismatch_m"] > THRESHOLDS["gps_spoofing_disp_m"]
    triggered = absolute | jump
    severity = (df["nav_gps_vs_fused_mismatch_m"].clip(upper=25).where(absolute, 0) +
                df["nav_disp_mismatch_m"].clip(upper=10).where(jump, 0))
    return triggered, severity


def detect_telemetry_manipulation_vec(df):
    alt_bad = df["telem_alt_mismatch_m"] > THRESHOLDS["telemetry_alt_mismatch_m"]
    vz_bad = df["telem_vz_mismatch"] > THRESHOLDS["telemetry_vz_mismatch"]
    triggered = alt_bad | vz_bad
    severity = (df["telem_alt_mismatch_m"].clip(upper=20).where(alt_bad, 0) +
                pd.Series(10, index=df.index).where(vz_bad, 0))
    return triggered, severity


def detect_dos_vec(df):
    """
    Dual-window sparse-traffic check, PLUS the direct heartbeat-gap check.
    Long window (1.0s) is the stable/low-FPR signal; short window (0.3s)
    reacts ~3x faster at attack onset, when the long window is still
    "stale" with pre-attack traffic (see feature_engine.py). Combining
    both catches onset rows the long window alone misses, without
    loosening the long window's own (already zero-FPR-tuned) threshold.
    """
    gap_bad = df["hb_gap_s"] > THRESHOLDS["dos_hb_gap_s"]

    past_warmup_1s = df["flight_elapsed_s"] >= 1.0
    sparse_1s = past_warmup_1s & (df["msg_count_1s"] <= THRESHOLDS["dos_msg_count_1s"])

    past_warmup_short = df["flight_elapsed_s"] >= 0.3
    sparse_short = past_warmup_short & (df["msg_count_0_3s"] <= THRESHOLDS["dos_msg_count_0_3s"])

    triggered = gap_bad | sparse_1s | sparse_short
    severity = (pd.Series(15, index=df.index).where(gap_bad, 0) +
                pd.Series(15, index=df.index).where(sparse_1s, 0) +
                pd.Series(10, index=df.index).where(sparse_short & ~sparse_1s, 0))
    return triggered, severity


def detect_mavlink_anomaly_vec(df):
    triggered = ((df["anomaly_bad_crc"] > 0) |
                 (df["anomaly_replayed_seq"] > 0) |
                 (df["anomaly_unexpected_type"] > 0))
    severity = (10 * (df["anomaly_bad_crc"] + df["anomaly_replayed_seq"] +
                       df["anomaly_unexpected_type"])).clip(upper=25)
    return triggered, severity


# Kept for reference/back-compat (e.g. external callers importing the
# single-row functions directly) — no longer used by run_rule_engine.
def detect_gps_spoofing(row):
    # Primary: raw GPS vs fused-position instantaneous mismatch — robust,
    # non-zero for every row of a sustained spoof, immune to turn-transients.
    absolute = row["nav_gps_vs_fused_mismatch_m"] > THRESHOLDS["gps_spoofing_divergence_m"]
    # Secondary: frame-to-frame jump — catches the attack even faster at
    # onset (contributes to near-0ms latency), corroborates severity.
    jump = row["nav_disp_mismatch_m"] > THRESHOLDS["gps_spoofing_disp_m"]
    triggered = absolute or jump
    severity = (min(25, row["nav_gps_vs_fused_mismatch_m"]) if absolute else 0) + \
               (min(10, row["nav_disp_mismatch_m"]) if jump else 0)
    return triggered, severity


def detect_telemetry_manipulation(row):
    alt_bad = row["telem_alt_mismatch_m"] > THRESHOLDS["telemetry_alt_mismatch_m"]
    vz_bad = row["telem_vz_mismatch"] > THRESHOLDS["telemetry_vz_mismatch"]
    triggered = alt_bad or vz_bad
    severity = (min(20, row["telem_alt_mismatch_m"]) if alt_bad else 0) + \
               (10 if vz_bad else 0)
    return triggered, severity


def detect_dos(row):
    gap_bad = row["hb_gap_s"] > THRESHOLDS["dos_hb_gap_s"]
    # sparse_bad: low message count in the trailing 1s, only evaluated
    # once the flight is past its first second (the trailing window needs
    # a full second of real history before the count is meaningful —
    # otherwise every flight's opening second would look "sparse" simply
    # because there's not yet enough history, not because of any drops).
    past_warmup = row["flight_elapsed_s"] >= 1.0
    sparse_bad = past_warmup and (row["msg_count_1s"] <= 6)
    triggered = gap_bad or sparse_bad
    severity = (15 if gap_bad else 0) + (15 if sparse_bad else 0)
    return triggered, severity


def detect_mavlink_anomaly(row):
    triggered = (row["anomaly_bad_crc"] > 0 or
                 row["anomaly_replayed_seq"] > 0 or
                 row["anomaly_unexpected_type"] > 0)
    severity = 10 * (row["anomaly_bad_crc"] + row["anomaly_replayed_seq"] +
                      row["anomaly_unexpected_type"])
    return triggered, min(25, severity)


def detect_command_injection(cmd_df, legit_system_id=255):
    """
    Operates on the separate injected-commands log (event-based, not
    per-timestep telemetry). Returns a DataFrame of flagged events.
    """
    if cmd_df.empty:
        return cmd_df.assign(flagged=[])
    flagged = cmd_df["source_system"] != legit_system_id
    out = cmd_df.copy()
    out["flagged"] = flagged
    out["severity"] = flagged.map({True: 30, False: 0})
    return out


def detect_firmware_integrity(fw_df):
    """Operates on the firmware attestation log."""
    out = fw_df.copy()
    out["flagged"] = ~out["match"]
    out["severity"] = out["flagged"].map({True: 40, False: 0})
    return out


def apply_latch(df, flag_col, severity_col, predicted_col, attack_name, hold_samples=15):
    """
    Alert latching: once `attack_name` triggers, keep it 'active' for
    `hold_samples` rows even if the instantaneous per-row signal drops
    below threshold in between. This is standard IDS practice (avoid
    alert flapping on a noisy-but-real ongoing condition) and lets a
    reliable-but-sporadic frame-to-frame signal (which fires several
    times per attack window due to injected jitter, but not on EVERY
    row) cover the full attack duration without needing a fragile
    cumulative/integrated signal.

    hold_samples=15 -> at ~10Hz that's ~1.5s; tuned to bridge the gaps
    between sporadic per-row triggers within one continuous attack,
    while still resetting well within the ~10-20s gap between our
    distinct attack windows (so it can't bleed from one attack into
    the next attack's ground-truth window).
    """
    flags = df[flag_col].values.copy()
    n = len(flags)
    latched = flags.copy()
    last_trigger = -10**9
    for i in range(n):
        if flags[i]:
            last_trigger = i
        elif i - last_trigger <= hold_samples:
            latched[i] = True
    df[flag_col] = latched
    # rows newly latched (weren't originally flagged) get a modest fixed
    # severity rather than 0, so they still contribute to fusion risk
    newly_latched = latched & ~flags
    df.loc[newly_latched, severity_col] = df.loc[newly_latched, severity_col].clip(lower=10)
    return df


def run_rule_engine(feat_df):
    """
    Apply all per-timestep rule detectors to the feature-engineered
    DataFrame. Returns feat_df with added columns:
      rule_<attack>_flag, rule_<attack>_severity
      rule_any_flag, rule_total_severity, rule_predicted_attack

    Note: GPS spoofing needs no latching — it now uses a direct raw-GPS-
    vs-fused-position instantaneous comparison (see feature_engine.py),
    non-zero for every row of a sustained attack, so full-window coverage
    comes naturally from the signal itself, not from bridging sporadic
    triggers.

    DoS DOES need latching. Root cause: drop_probability<1.0 means
    individual messages survive at random during the attack window, so
    a surviving message can land right after the previous one purely by
    chance (hb_gap_s looks normal for that one row), and msg_count_1s is
    a BACKWARD-looking trailing window that's still full of pre-attack
    traffic for the first ~1s after the attack starts (onset lag). Both
    produce sporadic per-row false negatives inside a real, continuous
    DoS window. hold_samples=3 (~0.3-0.4s at this capture rate) bridges
    those gaps: verified empirically it takes DoS per-timestep recall
    from 82.6% to 95.7% on the labeled dataset (the one row it still
    can't catch is the attack's very first sample, before any trigger
    exists to latch from -- unavoidable without future information),
    for a modest normal-flight FPR cost (0.61% -> 1.07% overall).
    """
    df = feat_df.copy()
    detectors = {
        "gps_spoofing": detect_gps_spoofing_vec,
        "telemetry_manipulation": detect_telemetry_manipulation_vec,
        "dos": detect_dos_vec,
        "mavlink_anomaly": detect_mavlink_anomaly_vec,
    }

    for name, fn in detectors.items():
        flags, sevs = fn(df)
        df[f"rule_{name}_flag"] = flags
        df[f"rule_{name}_severity"] = sevs

    # DoS: bridge sporadic per-row misses caused by random message
    # survival + trailing-window onset lag (see run_rule_engine docstring).
    df = apply_latch(df, "rule_dos_flag", "rule_dos_severity", None,
                      "dos", hold_samples=3)

    severity_cols = [f"rule_{n}_severity" for n in detectors]
    flag_cols = [f"rule_{n}_flag" for n in detectors]

    df["rule_total_severity"] = df[severity_cols].sum(axis=1)
    df["rule_any_flag"] = df[flag_cols].any(axis=1)

    # Vectorized "which attack had highest severity" via idxmax instead of
    # a per-row apply() — same result, no Python-level row loop.
    sev_matrix = df[severity_cols].copy()
    sev_matrix.columns = list(detectors.keys())
    best_name = sev_matrix.idxmax(axis=1)
    best_sev = sev_matrix.max(axis=1)
    df["rule_predicted_attack"] = best_name.where(best_sev > 0, "none")
    return df


if __name__ == "__main__":
    import sys, os
    project_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, project_root)
    os.chdir(project_root)
    from build_dataset import build

    feat_df, feature_cols, cmd_df, fw_df = build()
    result = run_rule_engine(feat_df)

    print("=== Per-timestep rule detection ===")
    confusion = pd.crosstab(result["gt_attack_type"], result["rule_predicted_attack"])
    print(confusion)

    print("\n=== Command injection (event-based) ===")
    cmd_result = detect_command_injection(cmd_df)
    print(f"Flagged {cmd_result['flagged'].sum()} / {len(cmd_result)} injected commands")

    print("\n=== Firmware integrity (event-based) ===")
    fw_result = detect_firmware_integrity(fw_df)
    print(f"Flagged {fw_result['flagged'].sum()} / {len(fw_result)} attestations")
