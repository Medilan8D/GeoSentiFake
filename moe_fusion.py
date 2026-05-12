"""
pipeline/moe_fusion.py — Stage 7: Mixture-of-Experts Fusion

Combines all feature streams with dynamically computed weights.
Production: transformer gating network.
Demo: confidence-weighted attention over base SHAP weights.
"""

import math
from typing import Dict


class MoEFusion:

    def __init__(self, config):
        self.cfg = config

    def fuse(
        self,
        detection: dict,
        ess:       dict,
        stance:    dict,
        evidence:  dict,
    ) -> dict:
        """
        Fuses six feature streams into a final label + confidence.
        """
        # ── Feature vector ────────────────────────────────────────────────────
        fake_prob   = detection.get("fake_probability", 0.5)
        real_prob   = detection.get("real_probability", 0.5)
        partial_prob = detection.get("partial_probability", 0.1)
        bilstm_prob = detection.get("bilstm_fake_probability", 0.5)
        eis         = ess.get("eis", 0.3)
        dbs         = ess.get("dbs", 0.3)
        cds         = ess.get("cds", 0.2)
        ess_score   = ess.get("ess_score", 30.0)
        credibility = evidence.get("credibility_score", 0.5)
        stance_conf = stance.get("confidence", 0.5)
        stance_deny = 1.0 if stance.get("label") == "denies" else 0.0
        lex_div     = 0.5   # placeholder (in real pipeline: from preprocessed)
        spell_sc    = 0.8   # placeholder

        # ── Gating weights (dynamic — calibrated by stream confidence) ────────
        base_w = self.cfg.moe_base_weights
        stream_confidences = {
            "fakebert":    abs(fake_prob - 0.5) * 2,   # distance from uncertainty
            "eis":         eis,
            "credibility": credibility,
            "dbs":         dbs,
            "stance":      stance_conf,
            "cds":         cds,
            "style":       (lex_div + max(0, spell_sc)) / 2,
        }

        # Attention-weighted gate: final_w[k] = base_w[k] * confidence[k]
        gate_w = {
            k: base_w.get(k, 0.1) * stream_confidences.get(k, 0.5)
            for k in base_w
        }
        total_gate = sum(gate_w.values()) or 1.0
        gate_w = {k: v / total_gate for k, v in gate_w.items()}

        # ── Fused fake probability ────────────────────────────────────────────
        # Weighted combination of the fake signals
        fused_fake = (
            gate_w["fakebert"]    * ((fake_prob + bilstm_prob) / 2)
            + gate_w["eis"]       * eis
            + gate_w["dbs"]       * dbs
            + gate_w["credibility"] * (1 - credibility)   # low cred → fake
            + gate_w["stance"]    * stance_deny
            + gate_w["cds"]       * cds
            + gate_w["style"]     * (1 - lex_div)
        )

        # Clamp
        fused_fake = min(1.0, max(0.0, fused_fake))

        # ── Three-class label ─────────────────────────────────────────────────
        if fused_fake > self.cfg.fake_threshold:
            label      = "FAKE"
            confidence = fused_fake
        elif fused_fake > self.cfg.partial_threshold:
            label      = "PARTIAL"
            confidence = fused_fake
        else:
            label      = "REAL"
            confidence = 1 - fused_fake

        return {
            "label":        label,
            "confidence":   round(confidence, 4),
            "fused_fake":   round(fused_fake, 4),
            "gate_weights": {k: round(v, 4) for k, v in gate_w.items()},
            # Pass-through for explainer
            "_detection":  detection,
            "_ess":        ess,
            "_stance":     stance,
            "_evidence":   evidence,
        }
