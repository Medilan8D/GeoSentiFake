"""
pipeline/emotion_engine.py — Stage 4: Emotion Detection + VAD Scoring

Production mode (requires: pip install transformers torch):
    → Loads GoEmotions-RoBERTa fine-tuned model
    → Remaps 27 GoEmotions labels → 8 Plutchik primary emotions
    → Computes Valence-Arousal-Dominance via NRC Emotion Lexicon

Offline/Demo mode:
    → Lexicon-based emotion classifier using the NRC Word-Emotion Association
      Lexicon (embedded subset) plus manipulation-pattern rules
    → VAD scores from the embedded NRC-VAD dictionary in config
    → Results are semantically valid, though less nuanced than transformer output
"""

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# ── Embedded emotion lexicons (NRC subset) ────────────────────────────────────
# Format: word → {emotion: weight} where weight ∈ [0, 1]
NRC_EMOTION_LEXICON = {
    # Fear
    "massacre":{"fear":0.97,"anger":0.80,"disgust":0.70,"sadness":0.75},
    "genocide": {"fear":0.95,"anger":0.88,"disgust":0.90,"sadness":0.82},
    "terror":   {"fear":0.98,"anger":0.75,"disgust":0.65,"sadness":0.60},
    "attack":   {"fear":0.82,"anger":0.78,"disgust":0.45},
    "threat":   {"fear":0.88,"anger":0.55,"anticipation":0.60},
    "danger":   {"fear":0.90,"anticipation":0.65},
    "bomb":     {"fear":0.92,"anger":0.70,"surprise":0.60},
    "killed":   {"fear":0.75,"sadness":0.90,"anger":0.65},
    "dead":     {"sadness":0.88,"fear":0.65},
    "death":    {"sadness":0.90,"fear":0.72},
    "murder":   {"fear":0.82,"anger":0.85,"disgust":0.75,"sadness":0.80},
    "violence": {"fear":0.85,"anger":0.80,"disgust":0.72},
    "war":      {"fear":0.80,"anger":0.75,"sadness":0.68},
    "crisis":   {"fear":0.78,"anticipation":0.70,"sadness":0.55},
    "survival": {"fear":0.78,"anticipation":0.70},
    "cannibalism": {"fear":0.95,"disgust":1.00},
    # Anger
    "condemns": {"anger":0.85,"disgust":0.70},
    "condemn":  {"anger":0.84,"disgust":0.68},
    "aggressors":{"anger":0.88,"disgust":0.75,"fear":0.50},
    "savage":   {"anger":0.90,"disgust":0.85,"fear":0.72},
    "enemy":    {"anger":0.75,"fear":0.65,"disgust":0.60},
    "occupation":{"anger":0.72,"disgust":0.65,"sadness":0.55},
    "oppression":{"anger":0.80,"sadness":0.70,"disgust":0.65},
    "betrayal": {"anger":0.85,"sadness":0.75,"disgust":0.70},
    "traitor":  {"anger":0.88,"disgust":0.80},
    "puppet":   {"anger":0.70,"disgust":0.75},
    "killing": {"anger":0.88,"fear":0.75},
    "violence": {"anger":0.80,"fear":0.85},
    "brutality": {"anger":0.90,"disgust":0.88},
    "inhumane": {"anger":0.85,"disgust":0.92},
    # Disgust
    "atrocities":{"disgust":0.92,"anger":0.85,"fear":0.75,"sadness":0.80},
    "crimes":   {"disgust":0.80,"anger":0.75,"sadness":0.60},
    "corruption":{"disgust":0.85,"anger":0.78},
    "hypocrite":{"disgust":0.82,"anger":0.70},
    "human flesh": {"disgust":0.98,"fear":0.90},
    # Sadness
    "civilians":{"sadness":0.65,"fear":0.55},
    "innocent": {"sadness":0.60,"trust":0.50},
    "refugees": {"sadness":0.82,"fear":0.60},
    "victims":  {"sadness":0.88,"fear":0.55},
    "suffering":{"sadness":0.90,"fear":0.60},
    "mourning": {"sadness":0.92,"fear":0.40},
    "disaster": {"sadness":0.80,"fear":0.75,"surprise":0.55},
    "dead": {"sadness":0.90},
    # Surprise
    "unexpected":{"surprise":0.80,"anticipation":0.50},
    "shocking": {"surprise":0.88,"fear":0.60},
    "revealed": {"surprise":0.72,"anticipation":0.60},
    "breaking": {"surprise":0.70,"anticipation":0.75},
    "sudden":   {"surprise":0.75,"fear":0.50},
    "bizarre": {"surprise":0.85,"disgust":0.75},
    # Joy
    "peace":    {"joy":0.88,"trust":0.82,"anticipation":0.65},
    "ceasefire":{"joy":0.65,"trust":0.72,"anticipation":0.70},
    "agreement":{"joy":0.72,"trust":0.78,"anticipation":0.65},
    "freedom":  {"joy":0.85,"trust":0.65,"anticipation":0.72},
    "hopeful":  {"joy":0.78,"trust":0.65,"anticipation":0.80},
    "optimistic":{"joy":0.82,"trust":0.70,"anticipation":0.85},
    # Trust
    "confirmed":{"trust":0.80,"anticipation":0.45},
    "verified": {"trust":0.85,"anticipation":0.40},
    "official": {"trust":0.70,"anticipation":0.40},
    "independent":{"trust":0.78},
    "diplomatic":{"trust":0.72,"anticipation":0.65},
    "agreement":{"trust":0.78,"joy":0.65},
    "report":   {"trust":0.55},
    "evidence": {"trust":0.75},
    # Anticipation
    "upcoming": {"anticipation":0.75},
    "planned":  {"anticipation":0.70},
    "scheduled":{"anticipation":0.72},
    "election": {"anticipation":0.80,"fear":0.45},
    "vote":     {"anticipation":0.75,"trust":0.50},
    "summit":   {"anticipation":0.68,"trust":0.62,"joy":0.45},
    "talks":    {"anticipation":0.65,"trust":0.55},
    "negotiation":{"anticipation":0.68,"trust":0.60},
    "emergency":{"fear":0.80,"anticipation":0.75,"surprise":0.65},
    "urgent":   {"anticipation":0.75,"fear":0.55},
}


class EmotionEngine:
    """
    Detects Plutchik emotions and VAD scores for an article.
    Uses transformers if available; embedded NRC lexicon otherwise.
    """

    def __init__(self, config):
        self.cfg = config
        self._model = None
        self._try_load_model()

    def _try_load_model(self):
        """
        Load GoEmotions-RoBERTa. Replace with your fine-tuned checkpoint.
        Model: SamLowe/roberta-base-go_emotions  (public, ~500MB)
        """
        try:
            from transformers import pipeline
            import torch
            # Uncomment to use transformer:
            # self._model = pipeline(
            #     "text-classification",
            #     model="SamLowe/roberta-base-go_emotions",
            #     device=0 if torch.cuda.is_available() else -1,
            #     top_k=None, truncation=True, max_length=512
            # )
            self._model = None
        except ImportError:
            self._model = None

    # ── Public ────────────────────────────────────────────────────────────────
    def analyse(self, preprocessed: dict, evidence: dict) -> dict:
        text   = preprocessed.get("cleaned_text", "")
        tokens = preprocessed.get("filtered_tokens", [])

        if self._model:
            plutchik, goemo_raw = self._transformer_emotions(text)
        else:
            plutchik = self._lexicon_emotions(tokens, text)
            goemo_raw = {}

        vad = self._compute_vad(tokens, text)
        dominant = max(plutchik, key=plutchik.get)

        return {
            "plutchik_probs":  {k: round(v, 4) for k, v in plutchik.items()},
            "dominant_emotion": dominant,
            "goemo_raw":       goemo_raw,
            "vad":             {k: round(v, 4) for k, v in vad.items()},
        }

    # ── Transformer path ──────────────────────────────────────────────────────
    def _transformer_emotions(self, text: str) -> Tuple[Dict, Dict]:
        results = self._model(text[:512])[0]  # top_k=None returns all labels
        goemo_raw = {r["label"]: r["score"] for r in results}

        # Aggregate GoEmotions → Plutchik
        plutchik = defaultdict(float)
        mapping  = self.cfg.goemo_to_plutchik
        for goemo_label, score in goemo_raw.items():
            plutchik_label = mapping.get(goemo_label)
            if plutchik_label:
                plutchik[plutchik_label] += score

        # Ensure all 8 are present
        for emo in self.cfg.plutchik_emotions:
            if emo not in plutchik:
                plutchik[emo] = 0.01

        # Normalise to sum=1
        total = sum(plutchik.values()) or 1
        plutchik = {k: v / total for k, v in plutchik.items()}
        return dict(plutchik), goemo_raw

    # ── Lexicon path ──────────────────────────────────────────────────────────
    def _lexicon_emotions(self, tokens: List[str], text: str) -> Dict[str, float]:
        """
        NRC lexicon-based emotion scoring.
        For each token found in the lexicon, add its emotion weights.
        Apply phrase-level boosts for known manipulation patterns.
        """
        scores = defaultdict(float)
        for tok in tokens:
            if tok in NRC_EMOTION_LEXICON:
                for emo, weight in NRC_EMOTION_LEXICON[tok].items():
                    scores[emo] += weight

        # Phrase-level boosts
        text_lower = text.lower()
        phrase_boosts = {
            "freedom-loving nation": {"anticipation": 0.4, "trust": 0.3},
            "before it is too late": {"fear": 0.5, "anticipation": 0.4},
            "crime against humanity": {"anger": 0.5, "disgust": 0.5, "fear": 0.4},
            "take to the streets":   {"anger": 0.4, "anticipation": 0.3},
            "silence is complicity": {"anger": 0.4, "disgust": 0.3},
            "rise up":               {"anger": 0.3, "anticipation": 0.3},
            "cautiously optimistic": {"joy": 0.3,  "trust": 0.3},
            "ceasefire agreement":   {"joy": 0.5,  "trust": 0.4},
            "peace summit":          {"joy": 0.4,  "trust": 0.5, "anticipation": 0.3},
            "war crimes":            {"anger": 0.5, "disgust": 0.5, "fear": 0.4},
            "forced displacement":   {"sadness": 0.5,"fear": 0.4},
        }
        for phrase, boosts in phrase_boosts.items():
            if phrase in text_lower:
                for emo, boost in boosts.items():
                    scores[emo] += boost

        # Ensure all 8 emotions present with small baseline
        for emo in self.cfg.plutchik_emotions:
            if emo not in scores or scores[emo] < 0.01:
                scores[emo] = 0.01

        # Normalise to sum=1
        total = sum(scores.values()) or 1.0
        return {emo: scores[emo] / total for emo in self.cfg.plutchik_emotions}

    # ── VAD Computation ───────────────────────────────────────────────────────
    def _compute_vad(self, tokens: List[str], text: str) -> Dict[str, float]:
        """
        Token-frequency-weighted VAD averaging over NRC-VAD lexicon.
        Returns valence ∈ [-1,+1], arousal ∈ [0,1], dominance ∈ [-1,+1].
        """
        vad_sample = self.cfg.nrc_vad_sample
        v_sum = a_sum = d_sum = 0.0
        count = 0

        for tok in tokens:
            if tok in vad_sample:
                v, a, d = vad_sample[tok]
                v_sum += v
                a_sum += a
                d_sum += d
                count += 1

        if count == 0:
            # Neutral baseline
            return {"valence": 0.0, "arousal": 0.25, "dominance": 0.0}

        valence   = round(v_sum  / count, 4)
        arousal   = round(a_sum  / count, 4)
        dominance = round(d_sum  / count, 4)

        # Clamp
        valence   = max(-1.0, min(1.0, valence))
        arousal   = max( 0.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))

        return {"valence": valence, "arousal": arousal, "dominance": dominance}
