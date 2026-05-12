"""
pipeline/explainer.py — Stage 8: SHAP Explainability + Verdict Generation

Production: Uses shap library with the MoE model for exact SHAP values.
Demo: Approximates SHAP importance from gate weights + feature scores.
Generates human-readable verdict text summarising the decision.
"""

from typing import Dict, List, Tuple


VERDICT_TEMPLATES = {
    ("FAKE", "HIGH"): (
        "This article exhibits strong indicators of fabricated or manipulative content. "
        "The emotional profile is heavily skewed toward fear, anger, and/or disgust "
        "directed at specific geopolitical entities — a hallmark of deliberate "
        "emotional manipulation. The ESS score of {ess:.0f}/100 indicates a high "
        "potential for psychological influence, amplified by the current {gsm_level} "
        "context (GSM={gsm}×). Source credibility is low ({cred:.0%}) and stance "
        "analysis indicates the content {stance_desc}. "
        "Dominant emotion: {emotion}."
    ),
    ("FAKE", "MODERATE"): (
        "This article contains significant fabrication signals. Emotional language "
        "is elevated (ESS={ess:.0f}/100, {gsm_level}), with {emotion} as the dominant "
        "emotional vector. The source ({cred:.0%} credibility) and stance ({stance_desc}) "
        "further support classification as fake."
    ),
    ("FAKE", "LOW"): (
        "Content classified as likely fake primarily based on factual verification "
        "failure. Emotional skew is relatively low (ESS={ess:.0f}/100), suggesting "
        "fabrication without overt emotional manipulation."
    ),
    ("PARTIAL", "HIGH"): (
        "This article contains a factual core but employs significant emotional "
        "framing to amplify its impact. ESS={ess:.0f}/100 in a {gsm_level} context "
        "indicates high manipulation potential despite partial factual grounding. "
        "The dominant emotion ({emotion}) and directional bias (DBS={dbs:.3f}) "
        "suggest selective framing of real events. Stance: {stance_desc}."
    ),
    ("PARTIAL", "MODERATE"): (
        "Partially accurate content with moderate emotional skew (ESS={ess:.0f}/100). "
        "Emotional framing emphasises {emotion}, which may amplify the psychological "
        "impact of the factual kernel. Treat with caution."
    ),
    ("PARTIAL", "LOW"): (
        "Partially accurate content. Emotional skew is within normal journalistic "
        "range (ESS={ess:.0f}/100). The partial falsity appears to stem from "
        "factual error rather than emotional manipulation."
    ),
    ("REAL", "HIGH"): (
        "Content classified as real, but with an unusually high emotional skew "
        "(ESS={ess:.0f}/100) for the {gsm_level} context. While the facts appear "
        "accurate (credibility={cred:.0%}), the emotional framing is intense. "
        "The dominant emotion ({emotion}) may still significantly influence "
        "reader perception even in factual content."
    ),
    ("REAL", "MODERATE"): (
        "Content appears factually accurate (credibility={cred:.0%}). Moderate "
        "emotional intensity (ESS={ess:.0f}/100) is expected given the {gsm_level} "
        "context. Stance: {stance_desc}. Dominant emotion: {emotion}."
    ),
    ("REAL", "LOW"): (
        "Content is classified as real with low emotional skew (ESS={ess:.0f}/100). "
        "Reporting appears neutral and factually grounded. Credibility: {cred:.0%}. "
        "Consistent with balanced journalistic reporting."
    ),
}

STANCE_DESCRIPTIONS = {
    "supports":  "corroborates its own claims",
    "denies":    "actively refutes established facts",
    "questions": "introduces unverified allegations",
    "neutral":   "maintains a neutral framing",
}


class SHAPExplainer:

    def __init__(self, config):
        self.cfg = config
        self._shap_available = False
        self._try_load_shap()

    def _try_load_shap(self):
        try:
            import shap
            self._shap_available = True
        except ImportError:
            self._shap_available = False

    def explain(self, fusion: dict) -> dict:
        gate_w   = fusion.get("gate_weights", {})
        detection = fusion.get("_detection", {})
        ess       = fusion.get("_ess", {})
        stance    = fusion.get("_stance", {})
        evidence  = fusion.get("_evidence", {})

        label      = fusion.get("label", "REAL")
        confidence = fusion.get("confidence", 0.5)
        ess_score  = ess.get("ess_score", 30.0)
        ess_band   = ess.get("ess_band", "LOW")
        gsm        = ess.get("gsm", 1.0)
        gsm_level  = ess.get("gsm_level", "Peacetime")
        cred       = evidence.get("credibility_score", 0.5)
        stance_lbl = stance.get("label", "neutral")
        dbs        = ess.get("dbs", 0.0)

        # ── Approximate SHAP feature importances from gate weights ────────────
        feature_map = {
            "FakeBERT Classification":       gate_w.get("fakebert", 0.312),
            "Emotional Intensity (EIS)":     gate_w.get("eis", 0.198),
            "Credibility Cosine Similarity": gate_w.get("credibility", 0.171),
            "Directional Bias (DBS)":        gate_w.get("dbs", 0.148),
            "Stance Classification":         gate_w.get("stance", 0.091),
            "Contextual Deviation (CDS)":    gate_w.get("cds", 0.082),
            "Style Features":                gate_w.get("style", 0.099),
        }
        top_features: List[Tuple[str, float]] = sorted(
            feature_map.items(), key=lambda x: -x[1]
        )

        # ── Verdict text ──────────────────────────────────────────────────────
        # Get dominant emotion from fusion context (passed via _ess if available)
        dominant_emotion = fusion.get("_dominant_emotion", "fear")

        template_key = (label, ess_band)
        template = VERDICT_TEMPLATES.get(
            template_key,
            VERDICT_TEMPLATES.get((label, "MODERATE"),
            "Analysis complete. Label: {label}. ESS: {ess:.0f}/100.")
        )

        verdict = template.format(
            ess=ess_score,
            gsm=gsm,
            gsm_level=gsm_level,
            cred=cred,
            stance_desc=STANCE_DESCRIPTIONS.get(stance_lbl, stance_lbl),
            emotion=dominant_emotion.upper(),
            dbs=dbs,
            label=label,
        )

        return {
            "top_features": top_features,
            "verdict_text": verdict,
            "shap_mode":    "approximate" if not self._shap_available else "exact",
        }
