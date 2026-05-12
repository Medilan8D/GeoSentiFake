"""
pipeline/ess_calculator.py — Stage 5: Emotional Skew Score Computation

This is the core novel contribution of the GeoSentiFake framework.

ESS = (α·EIS + β·DBS + γ·CDS) × GSM

Where:
    EIS = Emotional Intensity Score  = (Arousal_VAD + |Valence_VAD|) / 2
    DBS = Directional Bias Score     = Σ P(Fear+Anger+Disgust) per NER entity
    CDS = Contextual Deviation Score = |EIS_article − EIS_neutral_baseline|
    GSM = Geopolitical Sensitivity Multiplier (from GDELT crisis data)

ESS ∈ [0, 100]   Bands:  LOW [0–33]  ·  MODERATE [34–66]  ·  HIGH [67–100]
"""

import math
from typing import Dict


class ESSCalculator:
    """
    Computes the Emotional Skew Score and all its sub-components.
    """

    def __init__(self, config):
        self.cfg = config

    def compute(
        self,
        preprocessed: dict,
        emotion:       dict,
        evidence:      dict,
        gdelt_region:  str = "Unknown",
    ) -> dict:
        """
        Main entry. Returns full ESS breakdown dict.
        """
        vad     = emotion["vad"]
        plutchik = emotion["plutchik_probs"]
        entities = preprocessed.get("entities", {})
        baseline = evidence.get("neutral_eis_baseline", 0.20)

        # ── Sub-score computation ─────────────────────────────────────────────
        eis = self._compute_eis(vad)
        dbs = self._compute_dbs(plutchik, entities, preprocessed)
        cds = self._compute_cds(eis, baseline)
        gsm, gsm_level = self._compute_gsm(gdelt_region, preprocessed)

        # ── Raw ESS (before scaling) ──────────────────────────────────────────
        α = self.cfg.alpha   # 0.40
        β = self.cfg.beta    # 0.35
        γ = self.cfg.gamma   # 0.25

        raw_ess = (α * eis + β * dbs + γ * cds) * gsm

        # Scale to [0, 100] and clip
        ess_score = round(min(100.0, max(0.0, raw_ess * 100)), 2)

        # ── Band assignment ───────────────────────────────────────────────────
        if ess_score >= self.cfg.ess_high_threshold:
            band = "HIGH"
        elif ess_score >= self.cfg.ess_low_threshold:
            band = "MODERATE"
        else:
            band = "LOW"

        return {
            "eis":       round(eis, 4),
            "dbs":       round(dbs, 4),
            "cds":       round(cds, 4),
            "gsm":       gsm,
            "gsm_level": gsm_level,
            "raw_ess":   round(raw_ess, 4),
            "ess_score": ess_score,
            "ess_band":  band,
            "alpha":     α,
            "beta":      β,
            "gamma":     γ,
        }

    # ── EIS — Emotional Intensity Score ───────────────────────────────────────
    def _compute_eis(self, vad: Dict[str, float]) -> float:
        """
        EIS = (Arousal_VAD + |Valence_VAD|) / 2

        Arousal captures physiological activation (outrage, panic, excitement).
        |Valence| captures emotional extremity in either direction.
        Both normalised to [0,1], so EIS ∈ [0,1].
        """
        arousal = max(0.0, min(1.0, vad.get("arousal", 0.0)))
        valence = abs(vad.get("valence", 0.0))  # magnitude regardless of sign
        valence = min(1.0, valence)
        eis = (arousal + valence) / 2.0
        return round(eis, 6)

    # ── DBS — Directional Bias Score ──────────────────────────────────────────
    def _compute_dbs(
        self,
        plutchik:    Dict[str, float],
        entities:    dict,
        preprocessed: dict,
    ) -> float:
        """
        DBS = concentration of negative manipulation emotions (Fear+Anger+Disgust)
              targeted at specific geopolitical entities.

        A high DBS indicates emotional scapegoating — projecting negative affect
        onto a named entity (country, group, leader).

        DBS ∈ [0, 1]
        """
        manip_emotions = self.cfg.manipulation_emotions  # ["fear","anger","disgust"]
        neg_emotion_mass = sum(
            plutchik.get(e, 0.0) for e in manip_emotions
        )

        # Entity density: how many distinct GPE/ORG entities are present?
        gpe_count = len(entities.get("GPE", []))
        org_count = len(entities.get("ORG", []))
        entity_count = gpe_count + org_count

        # Directional targeting boost: negative emotion proximate to entities
        # (In production, this is computed sentence-level with NER co-occurrence)
        text  = preprocessed.get("cleaned_text", "")
        title = preprocessed.get("title", "").lower()
        targeting_boost = self._entity_targeting_score(text, title, entities)

        if entity_count == 0:
            # No named entities → DBS is purely the raw emotion mass
            dbs = neg_emotion_mass * 0.7
        else:
            # More entities targeted → stronger directional bias signal
            entity_factor = min(1.0, math.log1p(entity_count) / math.log1p(5))
            dbs = neg_emotion_mass * (0.6 + 0.4 * entity_factor) * (1 + targeting_boost * 0.3)

        return round(min(1.0, dbs), 6)

    def _entity_targeting_score(
        self, text: str, title: str, entities: dict
    ) -> float:
        """
        Heuristic for how closely negative emotion words are associated
        with named entities in the text.
        """
        # Targeting verb patterns preceding entity references
        targeting_verbs = {
            "condemns", "blames", "accuses", "attacks", "warns", "threatens",
            "calls", "labels", "declares", "slams", "denounces", "criticises",
        }
        text_words = set(text.lower().split())
        verb_hits  = len(targeting_verbs & text_words)

        # First-person "our" vs "their" / "enemy" framing
        us_them_markers = {"our","we","us","theirs","them","enemy","aggressors","occupiers"}
        us_them_count = sum(1 for w in text.lower().split() if w in us_them_markers)

        title_neg = sum(
            1 for m in {"condemn","attack","strike","kill","genocide","war","crime"}
            if m in title
        )

        score = (
            min(1.0, verb_hits * 0.20)
            + min(0.5, us_them_count * 0.05)
            + min(0.5, title_neg * 0.25)
        )
        return round(min(1.0, score), 4)

    # ── CDS — Contextual Deviation Score ─────────────────────────────────────
    def _compute_cds(self, eis_article: float, neutral_baseline: float) -> float:
        """
        CDS = |EIS_article − EIS_neutral_baseline|

        Measures how far the article's emotional profile deviates from
        what neutral, high-credibility reporting of the same event looks like.
        Normalised to [0, 1] (max deviation = 1.0 when EIS=1 and baseline=0).

        CDS ∈ [0, 1]
        """
        cds = abs(eis_article - neutral_baseline)
        return round(min(1.0, cds), 6)

    # ── GSM — Geopolitical Sensitivity Multiplier ─────────────────────────────
    def _compute_gsm(self, region: str, preprocessed: dict) -> tuple:
        """
        Assigns GSM based on:
        1. GDELT-defined crisis region (from config active/moderate lists)
        2. Article-level crisis signals (as a secondary boost)

        Production: Query GDELT GKG API for GoldsteinScale, NumArticles, EventRootCode.
        Demo: Region-name matching against config crisis lists.

        Returns (gsm_value, gsm_level_string)
        """
        region_lower = region.lower()

        # Check active conflict regions
        is_active = any(
            r.lower() in region_lower or region_lower in r.lower()
            for r in self.cfg.active_conflict_regions
        )
        # Check moderate tension regions
        is_moderate = any(
            r.lower() in region_lower or region_lower in r.lower()
            for r in self.cfg.moderate_tension_regions
        )

        # Article-level crisis signals (secondary boost)
        text_lower = preprocessed.get("cleaned_text", "").lower()
        crisis_words = {
            "war","conflict","crisis","attack","offensive","invasion","siege",
            "bombardment","escalation","ceasefire","sanctions","occupation",
        }
        crisis_density = sum(1 for w in crisis_words if w in text_lower)
        article_boost  = crisis_density >= 3  # ≥3 crisis words → treat as higher context

        if is_active or article_boost:
            return self.cfg.gsm_high, "Active Conflict / High Crisis"
        elif is_moderate:
            return self.cfg.gsm_medium, "Moderate Tensions"
        else:
            # Even in peacetime, check for electoral crisis signals
            electoral_words = {"election","vote","ballot","fraud","coup","protest"}
            if any(w in text_lower for w in electoral_words):
                return self.cfg.gsm_medium, "Electoral / Political Crisis"
            return self.cfg.gsm_low, "Peacetime / Baseline"
