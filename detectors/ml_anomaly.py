"""
ml_anomaly.py

Second detection layer: an Isolation Forest trained ONLY on normal-flight
feature vectors. At inference time, any row scoring as an outlier relative
to that learned "normal" distribution is flagged — regardless of whether
it matches one of our 6 known attack signatures.

Why Isolation Forest specifically (worth stating in the proposal):
  - Lightweight: O(n log n) training, O(log n) inference per sample —
    suitable for a resource-constrained onboard deployment (directly
    serves the "computational efficiency" evaluation criterion).
  - No assumption of a particular attack distribution — genuinely a
    safety net for UNSEEN anomalies, not a 7th hand-coded rule.
  - Unsupervised: trains only on normal data, which is exactly what we
    can actually guarantee ground truth for in a real deployment (you
    don't get labeled attack data from the field).

This layer is explicitly NOT meant to replace the rule engine as primary
detector — see blueprint section 5: "The ML component should not be
presented as the sole 'brain' of the system."
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class MLAnomalyDetector:
    def __init__(self, contamination=0.02, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
        )
        self.feature_cols = None
        self.fitted = False

    def fit(self, feat_df, feature_cols):
        """Fit on NORMAL rows only — this is an unsupervised novelty detector."""
        self.feature_cols = feature_cols
        normal = feat_df[feat_df["gt_label"] == "normal"][feature_cols]
        X = self.scaler.fit_transform(normal)
        self.model.fit(X)
        self.fitted = True
        return self

    def score(self, feat_df):
        """
        Returns feat_df with two added columns:
          ml_anomaly_score : raw Isolation Forest decision_function output
                              (lower = more anomalous)
          ml_anomaly_flag  : boolean, True if predicted as outlier (-1)
          ml_risk_contribution : 0-40 scaled severity for fusion
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .score()")
        df = feat_df.copy()
        X = self.scaler.transform(df[self.feature_cols])
        raw_scores = self.model.decision_function(X)   # higher = more normal
        predictions = self.model.predict(X)              # -1 = anomaly, 1 = normal

        df["ml_anomaly_score"] = raw_scores
        df["ml_anomaly_flag"] = predictions == -1

        # scale decision_function output (~[-0.5, 0.5] typical range) to a
        # 0-40 severity contribution for the fusion engine; more negative
        # raw score -> higher severity
        clipped = np.clip(-raw_scores, 0, 0.5)
        df["ml_risk_contribution"] = (clipped / 0.5) * 40
        return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    from build_dataset import build

    feat_df, feature_cols, cmd_df, fw_df = build()

    detector = MLAnomalyDetector(contamination=0.02)
    detector.fit(feat_df, feature_cols)
    scored = detector.score(feat_df)

    print("=== ML Anomaly Layer — flag rate by ground truth ===")
    print(scored.groupby("gt_attack_type")["ml_anomaly_flag"].mean().sort_values(ascending=False))

    print("\n=== False positive rate on normal flight ===")
    normal = scored[scored["gt_attack_type"] == "none"]
    fpr = 100 * normal["ml_anomaly_flag"].mean()
    print(f"{fpr:.2f}%  ({normal['ml_anomaly_flag'].sum()}/{len(normal)})")

    print("\n=== Does ML catch the DoS cases rules missed? ===")
    dos_rows = scored[scored["gt_attack_type"] == "dos"]
    print(f"DoS rows flagged by ML: {dos_rows['ml_anomaly_flag'].sum()} / {len(dos_rows)}")
