"""
utils/config.py — Central configuration for GeoSentiFake.
All hyperparameters, model names, and threshold values in one place.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Config:
    # ── ESS Coefficients (tuned via grid-search on held-out validation) ───────
    alpha: float = 0.40   # EIS weight
    beta:  float = 0.35   # DBS weight
    gamma: float = 0.25   # CDS weight

    # ── GSM Thresholds ────────────────────────────────────────────────────────
    # Based on GDELT GoldsteinScale & event density heuristics
    gsm_low:    float = 1.0   # peacetime
    gsm_medium: float = 1.5   # moderate tensions
    gsm_high:   float = 2.0   # active conflict / electoral crisis

    # GDELT crisis regions currently in active conflict (updated periodically)
    active_conflict_regions: List[str] = field(default_factory=lambda: [
        "South Asia", "Middle East", "Eastern Europe", "West Africa",
        "Horn of Africa", "Myanmar", "Taiwan Strait",
    ])
    moderate_tension_regions: List[str] = field(default_factory=lambda: [
        "Latin America", "Southeast Asia", "Central Asia", "Balkans",
    ])

    # ── ESS Bands ─────────────────────────────────────────────────────────────
    ess_low_threshold:  float = 33.0
    ess_high_threshold: float = 66.0

    # ── Fake Detection Thresholds ─────────────────────────────────────────────
    fake_threshold:    float = 0.55   # P(fake) > this → FAKE
    partial_threshold: float = 0.35   # 0.35 < P(fake) ≤ 0.55 → PARTIAL

    # ── Emotion Engine ────────────────────────────────────────────────────────
    # Plutchik primary manipulation emotions (primary drivers of DBS)
    manipulation_emotions: List[str] = field(default_factory=lambda: [
        "fear", "anger", "disgust"
    ])

    # GoEmotions (27) → Plutchik (8) mapping
    goemo_to_plutchik: Dict[str, str] = field(default_factory=lambda: {
        # Fear
        "fear": "fear", "nervousness": "fear",
        # Anger
        "anger": "anger", "annoyance": "anger", "disapproval": "anger",
        # Disgust
        "disgust": "disgust",
        # Sadness
        "sadness": "sadness", "grief": "sadness", "remorse": "sadness",
        "disappointment": "sadness",
        # Surprise
        "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
        # Joy
        "joy": "joy", "amusement": "joy", "excitement": "joy",
        "gratitude": "joy", "love": "joy", "optimism": "joy",
        "pride": "joy", "relief": "joy",
        # Trust
        "admiration": "trust", "approval": "trust", "caring": "trust",
        # Anticipation
        "curiosity": "anticipation", "desire": "anticipation",
        # Excluded: neutral, embarrassment, etc. → None
    })

    plutchik_emotions: List[str] = field(default_factory=lambda: [
        "fear", "anger", "disgust", "sadness",
        "surprise", "joy", "trust", "anticipation"
    ])

    # ── NRC VAD Lexicon (sample subset embedded; full lexicon loaded at runtime) ──
    # Valence/Arousal/Dominance ∈ [-1, 1]
    nrc_vad_sample: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        # (valence, arousal, dominance)
        "massacre":     (-0.91, 0.85, -0.60),
        "genocide":     (-0.97, 0.82, -0.70),
        "attack":       (-0.76, 0.80, -0.30),
        "savage":       (-0.88, 0.79, -0.50),
        "innocent":     ( 0.70, 0.30,  0.10),
        "terror":       (-0.93, 0.90, -0.80),
        "aggressors":   (-0.80, 0.72, -0.40),
        "war":          (-0.85, 0.82, -0.20),
        "peace":        ( 0.90, 0.25,  0.60),
        "summit":       ( 0.55, 0.40,  0.50),
        "diplomatic":   ( 0.50, 0.30,  0.55),
        "optimistic":   ( 0.80, 0.55,  0.60),
        "condemns":     (-0.70, 0.75, -0.20),
        "uprising":     (-0.55, 0.80, -0.10),
        "genocide":     (-0.97, 0.92, -0.75),
        "crime":        (-0.83, 0.70, -0.40),
        "killed":       (-0.90, 0.78, -0.55),
        "ceasefire":    ( 0.60, 0.45,  0.40),
        "collapse":     (-0.72, 0.68, -0.35),
        "allegations":  (-0.55, 0.60, -0.20),
        "atrocities":   (-0.92, 0.85, -0.65),
        "freedom":      ( 0.88, 0.70,  0.75),
        "complicity":   (-0.65, 0.60, -0.30),
        "confirm":      ( 0.30, 0.30,  0.40),
        "cautiously":   ( 0.30, 0.25,  0.35),
        "neutral":      ( 0.10, 0.10,  0.30),
        "investigation":( 0.15, 0.40,  0.40),
        "forces":       (-0.30, 0.55,  0.20),
        "soldiers":     (-0.20, 0.50,  0.30),
        "civilians":    (-0.40, 0.55, -0.20),
        "fear":         (-0.85, 0.85, -0.65),
        "hope":         ( 0.80, 0.60,  0.55),
        "urgent":       (-0.20, 0.82,  0.20),
        "immediately":  ( 0.10, 0.75,  0.40),
        "destruction":  (-0.90, 0.80, -0.60),
        "displacement": (-0.78, 0.70, -0.45),
        "humanitarian": ( 0.55, 0.50,  0.40),
    })

    # ── Source Trustworthiness (YouGov 2024 rankings, normalised 0-1) ─────────
    source_trust: Dict[str, float] = field(default_factory=lambda: {
        "reuters.com":           0.92,
        "apnews.com":            0.91,
        "bbc.com":               0.88,
        "theguardian.com":       0.82,
        "nytimes.com":           0.81,
        "washingtonpost.com":    0.80,
        "aljazeera.com":         0.72,
        "theeconomist.com":      0.85,
        "bloomberg.com":         0.84,
        "npr.org":               0.83,
        "cnn.com":               0.70,
        "foxnews.com":           0.58,
        "msnbc.com":             0.60,
        "huffpost.com":          0.55,
        "breitbart.com":         0.32,
        "infowars.com":          0.08,
        "state-media-outlet.gov":0.15,
        "conflict-news-daily.net":0.20,
        "globalwatchnews.org":   0.25,
        "unknown":               0.40,  # neutral prior for unrecognised sources
    })

    # ── Stance Detection Thresholds ───────────────────────────────────────────
    stance_labels: List[str] = field(default_factory=lambda: [
        "supports", "denies", "questions", "neutral"
    ])

    # ── MoE Feature Weights (initial; gate network overrides dynamically) ─────
    moe_base_weights: Dict[str, float] = field(default_factory=lambda: {
        "fakebert":   0.312,
        "eis":        0.198,
        "credibility":0.171,
        "dbs":        0.148,
        "stance":     0.091,
        "cds":        0.082,
        "style":      0.099,   # lexical_diversity + spell_score combined
    })

    # ── Geopolitical entity keywords for NER-based DBS ───────────────────────
    geopolitical_keywords: List[str] = field(default_factory=lambda: [
        "government", "forces", "military", "soldiers", "troops",
        "rebels", "militants", "regime", "officials", "minister",
        "president", "prime minister", "army", "navy", "air force",
        "nato", "un", "united nations", "security council",
    ])
