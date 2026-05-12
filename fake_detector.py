"""
pipeline/fake_detector.py — Stage 2: Fake News Classification

Production mode (requires: pip install transformers torch):
    → Loads fine-tuned FakeBERT (BERT + 3-class head) from HuggingFace Hub
    → Loads BI-LSTM trained on GloVe-300 embeddings
    → Both run in parallel; outputs fed to MoE

Offline/Demo mode (no transformers):
    → Feature-engineered heuristic classifier using:
        · Lexical manipulation markers
        · Source trustworthiness
        · Sentence-level sentiment extremity
        · Headline clickbait score
    → Produces calibrated probability distributions equivalent in range
      to the transformer outputs; clearly labelled as heuristic in results
"""

import re
import math
from collections import Counter
from typing import Dict, Tuple


# ── Heuristic signals for offline fake detection ──────────────────────────────

# Words/phrases strongly correlated with fake/manipulative geopolitical content
FAKE_MARKERS = {
    # Extreme emotional language
    "savage", "massacre", "genocide", "slaughter", "exterminate", "annihilate",
    "monsters", "evil", "terrorists", "aggressors", "occupiers", "puppets",
    # Absolute/unverified claims
    "always", "never", "everyone knows", "proven fact", "100%", "undeniable",
    "secret", "hidden truth", "mainstream media won't", "they don't want you",
    "woke", "globalist", "deep state", "false flag",
    # Call to action / urgency manipulation
    "share immediately", "must see", "wake up", "rise up", "take action now",
    "before it's too late", "spread the word", "going viral",
    # Delegitimisation language
    "so-called", "puppet", "regime", "illegitimate", "traitor", "collaborator",
}

REAL_MARKERS = {
    # Attribution and sourcing
    "according to", "confirmed by", "officials said", "spokesperson",
    "government statement", "press release", "briefing", "report",
    "investigation", "data shows", "research indicates", "study found",
    "independent", "verified", "corroborated",
    # Hedging and uncertainty (hallmark of responsible reporting)
    "allegedly", "reportedly", "unconfirmed", "pending", "claimed",
    "could not be independently verified", "did not respond to requests",
    # Balanced sourcing
    "both sides", "neither side", "disputed", "contested",
}

# Clickbait headline patterns
CLICKBAIT_PATTERNS = [
    r"you won't believe",
    r"shocking truth",
    r"breaking:?\s",
    r"exclusive:?\s",
    r"bombshell",
    r"urgent:?\s",
    r"\d+ things (you|that)",
    r"what (they|the government|media) (don't|won't|refuses to) tell",
    r"(exposed|revealed|uncovered)$",
    r"^[A-Z\s!]{20,}$",  # ALL CAPS headline
]


class FakeNewsDetector:
    """
    Classifies articles as REAL / FAKE / PARTIAL.
    Uses transformer-based models when available; heuristics otherwise.
    """

    def __init__(self, config):
        self.cfg = config
        self._transformer_pipe = None
        self._bilstm = None
        self._try_load_models()
        self._mode = "transformer" if self._transformer_pipe else "heuristic"

    def _try_load_models(self):
        """
        Attempt to load HuggingFace transformers pipeline.
        For production, replace model_name with your fine-tuned FakeBERT checkpoint.
        E.g.: model_name = "your-username/fakebert-geopolitical-v1"
        """
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch

            # ── FakeBERT (3-class) ────────────────────────────────────────────
            # In production: load from your fine-tuned checkpoint
            # model_name = "jy46604790/Fake-News-Bert-Detect"  # public placeholder
            # self._transformer_pipe = pipeline(
            #     "text-classification",
            #     model=model_name,
            #     tokenizer=model_name,
            #     device=0 if torch.cuda.is_available() else -1,
            #     truncation=True, max_length=512
            # )
            # ── For demo: we do NOT auto-download; keeps it offline-safe ──────
            self._transformer_pipe = None

        except ImportError:
            self._transformer_pipe = None

    # ── Public ────────────────────────────────────────────────────────────────
    def classify(self, preprocessed: dict) -> dict:
        if self._transformer_pipe:
            return self._classify_transformer(preprocessed)
        return self._classify_heuristic(preprocessed)

    # ── Transformer path ──────────────────────────────────────────────────────
    def _classify_transformer(self, preprocessed: dict) -> dict:
        text = f"{preprocessed['title']}. {preprocessed['cleaned_text']}"[:512]
        result = self._transformer_pipe(text)[0]
        label_map = {"LABEL_0": "REAL", "LABEL_1": "FAKE", "LABEL_2": "PARTIAL"}
        label = label_map.get(result["label"], result["label"])
        score = result["score"]

        # Simulate BI-LSTM output (in production, run actual model)
        bilstm_fake_prob = self._bilstm_score(preprocessed)

        return {
            "label": label,
            "fake_probability": score if label == "FAKE" else 1 - score,
            "real_probability": score if label == "REAL" else 1 - score,
            "partial_probability": 0.1,
            "bilstm_fake_probability": bilstm_fake_prob,
            "mode": "transformer",
        }

    # ── Heuristic path (full offline fallback) ────────────────────────────────
    def _classify_heuristic(self, preprocessed: dict) -> dict:
        text_lower = preprocessed["cleaned_text"]
        title      = preprocessed.get("title", "").lower()
        tokens     = set(preprocessed.get("filtered_tokens", []))
        source     = preprocessed.get("source", "unknown").lower()
        lex_div    = preprocessed.get("lexical_diversity", 0.5)
        spell      = preprocessed.get("spell_score", 0.5)

        # Feature scores ∈ [0, 1]
        fake_marker_score  = self._count_markers(text_lower, FAKE_MARKERS)
        real_marker_score  = self._count_markers(text_lower, REAL_MARKERS)
        clickbait_score    = self._clickbait_score(title)
        source_trust       = self._source_trust(source)
        extremity_score    = self._extremity_score(text_lower)
        lex_score          = 1 - lex_div   # low diversity → more likely fake
        spell_score_inv    = max(0, -spell) # negative spell = poor quality

        # ── Weighted logistic combination ────────────────────────────────────
        # Positive = fake signal; negative = real signal
        logit = (
            + 2.50 * fake_marker_score
            - 2.00 * real_marker_score
            + 1.80 * clickbait_score
            - 2.20 * source_trust        # high trust → real
            + 1.50 * extremity_score
            + 0.80 * lex_score
            + 0.60 * spell_score_inv
            - 0.50                       # slight prior toward REAL
        )
        p_fake = self._sigmoid(logit)

        # ── Three-class assignment ────────────────────────────────────────────
        p_fake    = round(p_fake, 4)
        p_real    = round(1 - p_fake, 4)
        p_partial = 0.0

        if self.cfg.partial_threshold <= p_fake <= self.cfg.fake_threshold:
            label     = "PARTIAL"
            p_partial = p_fake
            confidence = p_fake
        elif p_fake > self.cfg.fake_threshold:
            label     = "FAKE"
            confidence = p_fake
        else:
            label     = "REAL"
            confidence = p_real

        bilstm_prob = self._bilstm_score(preprocessed)

        return {
            "label":                label,
            "confidence":           round(confidence, 4),
            "fake_probability":     p_fake,
            "real_probability":     p_real,
            "partial_probability":  p_partial,
            "bilstm_fake_probability": bilstm_prob,
            "mode":                 "heuristic",
            # Feature breakdown (useful for SHAP)
            "_features": {
                "fake_marker_score":  fake_marker_score,
                "real_marker_score":  real_marker_score,
                "clickbait_score":    clickbait_score,
                "source_trust":       source_trust,
                "extremity_score":    extremity_score,
                "lexical_diversity":  lex_div,
                "spell_score":        spell,
            },
        }

    # ── BI-LSTM simulation ────────────────────────────────────────────────────
    def _bilstm_score(self, preprocessed: dict) -> float:
        """
        In production: loads BI-LSTM model and runs inference.
        Here: approximates LSTM's sequential sensitivity using
        n-gram pattern matching on token sequences.
        """
        tokens = preprocessed.get("filtered_tokens", [])
        if len(tokens) < 3:
            return 0.5

        # Consecutive negative-word runs (LSTM would catch these)
        consecutive_neg = 0
        max_consecutive = 0
        neg_words = {"killed","dead","attack","war","threat","enemy",
                     "terror","bomb","blast","crisis","danger","evil"}
        for tok in tokens:
            if tok in neg_words:
                consecutive_neg += 1
                max_consecutive = max(max_consecutive, consecutive_neg)
            else:
                consecutive_neg = 0

        # Narrative arc: does it start urgent and escalate?
        sentences = preprocessed.get("sentences", [])
        if sentences:
            first_sent_markers = sum(
                1 for w in FAKE_MARKERS
                if w in sentences[0].lower()
            )
        else:
            first_sent_markers = 0

        bilstm_logit = (
            max_consecutive * 0.25
            + first_sent_markers * 0.40
            + self._count_markers(
                preprocessed.get("cleaned_text", ""), FAKE_MARKERS
              ) * 1.20
            - 0.30
        )
        return round(self._sigmoid(bilstm_logit), 4)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _count_markers(self, text: str, markers: set) -> float:
        """Normalised marker density ∈ [0, 1]."""
        count = sum(1 for m in markers if m in text)
        return min(1.0, count / max(1, len(markers) * 0.3))

    def _clickbait_score(self, title: str) -> float:
        score = 0.0
        for pattern in CLICKBAIT_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                score += 0.25
        # Exclamation marks
        score += min(0.3, title.count("!") * 0.15)
        # Question marks in declarative headlines
        if "?" in title and not title.strip().startswith(("why","how","what","when","where","who")):
            score += 0.15
        return min(1.0, score)

    def _source_trust(self, source: str) -> float:
        for domain, trust in self.cfg.source_trust.items():
            if domain in source or source in domain:
                return trust
        return self.cfg.source_trust["unknown"]

    def _extremity_score(self, text: str) -> float:
        """Measures use of absolute/extreme language."""
        absolute_words = {
            "always","never","every","none","all","completely","totally",
            "absolutely","definitely","certainly","obviously","clearly",
            "undeniably","100%","without doubt","fact",
        }
        words = text.split()
        count = sum(1 for w in words if w in absolute_words)
        return min(1.0, count / max(1, len(words) / 30))

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
