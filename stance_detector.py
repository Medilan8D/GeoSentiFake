"""
pipeline/stance_detector.py — Stage 6: Stance Detection

Production: DeBERTa-v3 fine-tuned on FNC-1 + FEVER datasets.
Offline: Rule-based stance classification using lexical patterns.

Stance labels: supports / denies / questions / neutral
"""

import re
from typing import Dict


SUPPORT_PATTERNS = [
    r"\bconfirmed by\b",
    r"\bofficials said\b",
    r"\bauthorities (said|confirmed)\b",
    r"\bministry (said|confirmed)\b",
    r"\baccording to (government|sources|reports?)\b",
    r"\bintelligence (reports?|suggests?)\b",
    r"\bexperts (say|confirm)\b",
    r"\bclearly shows\b",
    r"\bundeniable\b",
]

DENY_PATTERNS = [
    r"\brejected\b",
    r"\bdismiss(ed|es)\b",
    r"\bcalled (it )?baseless\b",
    r"\bstrongly denied\b",
    r"\bno credible evidence\b",
    r"\blacks evidence\b",
    r"\bwithout proof\b",
    r"\bmisleading\b",
    r"\bincorrect\b",
    r"\binaccurate\b",
    r"\bdisputed\b",
]

QUESTION_PATTERNS = [
    r"\baccording to\b",
    r"\bsources said\b",
    r"\bit is believed\b",
    r"\bit appears\b",
    r"\bsuggests that\b",
    r"\bindicates that\b",
    r"\bmay have\b",
    r"\bpossibly\b",
    r"\blikely\b",
    r"\braises questions\b",
    r"\bunclear whether\b",
]

NEUTRAL_PATTERNS = [
    r"\bsaid\b",   # VERY IMPORTANT (high-frequency neutral)
    r"\bstated\b",
    r"\bnoted\b",
    r"\bannounced that\b",
    r"\breleased\b",
    r"\bconfirmed today\b",
    r"\bbriefing\b",
    r"\bpress conference\b",
    r"\bofficial figures\b",
    r"\bdata released\b",
    r"\bupdate\b",
]


BIAS_PATTERNS = [
    r"\bno longer able to dictate\b",
    r"\billegal demands\b",
    r"\birrational demands\b",
    r"\baggression\b",
    r"\bhostile\b",
    r"\boppression\b",
    r"\bregime\b",
    r"\boccupying forces\b",
    r"\bthreat to\b",
    r"\bwarning that\b",
    r"\bmust act\b",
    r"\bglobal outrage\b",
]

class StanceDetector:

    def __init__(self, config):
        self.cfg = config
        self._model = None
        #self._try_load_model()

    """def _try_load_model(self):
        try:
            from transformers import pipeline
            import torch
            # Production: load fine-tuned DeBERTa
            # self._model = pipeline(
            #     "text-classification",
            #     model="your-username/deberta-stance-geopolitical",
            #     device=0 if torch.cuda.is_available() else -1,
            # )
            self._model = None
        except ImportError:
            self._model = None """

    def detect(self, preprocessed: dict, evidence: dict) -> dict:
        text     = preprocessed.get("cleaned_text", "")
        title    = preprocessed.get("title", "")
        combined = f"{title}. {text}"

        if self._model:
            return self._transformer_stance(combined)
        return self._heuristic_stance(combined, evidence)

    def _transformer_stance(self, text: str) -> dict:
        result = self._model(text[:512])[0]
        return {
            "label":      result["label"].lower(),
            "confidence": round(result["score"], 4),
            "mode":       "transformer",
        }

    def _heuristic_stance(self, text: str, evidence: dict) -> dict:
        text_lower = text.lower()

        def count_matches(patterns):
            return sum(
                1 for p in patterns if re.search(p, text_lower)
            )

        sup = count_matches(SUPPORT_PATTERNS)
        den = count_matches(DENY_PATTERNS)
        que = count_matches(QUESTION_PATTERNS)
        neu = count_matches(NEUTRAL_PATTERNS)
        bia = count_matches(BIAS_PATTERNS)
        credibility = evidence.get("credibility_score", 0.5)

        # Boost denial if source credibility is high (trusted sources often fact-check)
        if credibility > 0.80:
            den  += 0.5
            que  += 0.5
           
        scores = {"supports": sup, "denies": den,
                  "questions": que, "neutral": max(1, neu),"bias": bia  }
        total  = sum(scores.values()) or 1
        probs  = {k: round(v / total, 4) for k, v in scores.items()}
        best   = max(probs, key=probs.get)

        return {
            "label":      best,
            "confidence": probs[best],
            "all_probs":  probs,
            "mode":       "heuristic",
        }
