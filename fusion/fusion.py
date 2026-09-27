"""
fusion.py

Combines the two detection layers into ONE risk score (0-100), attack
classification and severity label, per the blueprint's design:

  - If the rule engine identifies a SPECIFIC known attack signature,
    that classification is trusted as primary (rules are explainable
    and precise for known patterns).
  - ML anomaly score is added as a CORROBORATING severity boost when it
    agrees with the rule engine, and down-weighted (0.5x) when acting
    alone, since the ML layer's standalone FPR (~5%) is higher than the
    rule engine's (~1%) — we don't want to inherit its false-positive
    rate at full strength.
  - If ONLY the ML layer flags an anomaly (no rule matched), we report
    "unknown_anomaly" rather than guessing a specific attack type —
    an honest, defensible design choice: we detected SOMETHING is wrong
    without overclaiming what it is. This is exactly the case that
    catches DoS scenarios the rule engine's thresholds missed (see
    ml_anomaly.py results: ML catches 23/23 DoS rows rules missed).

Risk score composition (example from blueprint):
  rule_total_severity (0-100, capped) + 0.5 * ml_risk_contribution (if
  rules silent) or 1.0x weight (if rules + ML agree) -> capped at 100.

Severity labels: Critical >=70, High >=45, Medium >=20, Low >0, None =0.
"""

import pandas as pd


def severity_label(risk_score):
    if risk_score >= 70:
        return "Critical"
    elif risk_score >= 45:
        return "High"
    elif risk_score >= 20:
        return "Medium"
    elif risk_score > 0:
        return "Low"
    return "None"


def fuse(rule_result_df, ml_scored_df):
    """
    rule_result_df: output of rule_engine.run_rule_engine()
    ml_scored_df  : output of MLAnomalyDetector.score() on the SAME rows
    Both must be row-aligned (same index / same underlying feat_df).
    """
    df = rule_result_df.copy()
    df["ml_anomaly_flag"] = ml_scored_df["ml_anomaly_flag"].values
    df["ml_risk_contribution"] = ml_scored_df["ml_risk_contribution"].values

    def combine(row):
        rule_flag = row["rule_any_flag"]
        rule_sev = row["rule_total_severity"]
        rule_pred = row["rule_predicted_attack"]
        ml_flag = row["ml_anomaly_flag"]
        ml_contrib = row["ml_risk_contribution"]

        if rule_flag:
            # rules found a known pattern; ML corroboration adds full weight
            # if it agrees (both suspicious), half weight if it disagrees
            weight = 1.0 if ml_flag else 0.5
            risk = min(100, rule_sev + weight * ml_contrib)
            predicted = rule_pred
        elif ml_flag:
            # ML alone: down-weighted, honestly labeled as unclassified
            risk = min(100, 0.5 * ml_contrib + 10)  # +10 base since *something* triggered
            predicted = "unknown_anomaly"
        else:
            risk = 0
            predicted = "none"

        return pd.Series({
            "fusion_risk_score": round(risk, 1),
            "fusion_predicted_attack": predicted,
            "fusion_severity": severity_label(risk),
        })

    fused = df.apply(combine, axis=1)
    return pd.concat([df, fused], axis=1)


if __name__ == "__main__":
    import sys, os
    project_root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "detectors"))
    os.chdir(project_root)
    from build_dataset import build
    from rule_engine import run_rule_engine
    from ml_anomaly import MLAnomalyDetector

    feat_df, feature_cols, cmd_df, fw_df = build()
    rule_result = run_rule_engine(feat_df)

    ml = MLAnomalyDetector(contamination=0.02).fit(feat_df, feature_cols)
    ml_scored = ml.score(feat_df)

    fused = fuse(rule_result, ml_scored)

    print("=== Fusion output: confusion (ground truth vs fused prediction) ===")
    print(pd.crosstab(fused["gt_attack_type"], fused["fusion_predicted_attack"]))

    print("\n=== 'Something is wrong' detection rate (fusion OR logic) ===")
    for atk in fused["gt_attack_type"].unique():
        if atk == "none":
            continue
        sub = fused[fused["gt_attack_type"] == atk]
        any_detected = (sub["fusion_predicted_attack"] != "none").mean() * 100
        exact_type = (sub["fusion_predicted_attack"] == atk).mean() * 100
        print(f"  {atk:25s}: flagged-as-suspicious={any_detected:5.1f}%  "
              f"correctly-typed={exact_type:5.1f}%")

    print("\n=== False positive rate (fusion) ===")
    normal = fused[fused["gt_attack_type"] == "none"]
    fpr = (normal["fusion_predicted_attack"] != "none").mean() * 100
    print(f"{fpr:.2f}%  ({(normal['fusion_predicted_attack'] != 'none').sum()}/{len(normal)})")

    print("\n=== Sample risk-score output (attack rows) ===")
    sample = fused[fused["gt_label"] == "attack"].sample(5, random_state=1)
    print(sample[["t_us", "gt_attack_type", "fusion_predicted_attack",
                   "fusion_risk_score", "fusion_severity"]].to_string(index=False))
