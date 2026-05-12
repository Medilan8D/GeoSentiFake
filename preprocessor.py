"""
pipeline/preprocessor.py — Stage 1: Text Preprocessing & NER

On a full installation (pip install spacy && python -m spacy download en_core_web_lg):
    → Uses spaCy for tokenisation, lemmatisation, and NER.

Offline fallback (no spaCy):
    → Pure-Python regex-based preprocessing with a curated GPE/ORG dictionary.
    → Functionally equivalent for demo purposes.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Tuple


# ── Stopwords (abbreviated; full NLTK set loaded if available) ────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "that","this","these","those","it","its","he","she","they","we","you","i",
    "his","her","their","our","your","my","me","him","us","them","who","which",
    "what","when","where","how","by","from","up","about","into","through","during",
    "before","after","then","than","so","if","not","no","nor","very","just",
    "also","both","each","few","more","most","other","some","such","only",
}

# ── Geopolitical Preservation List — NOT removed as stopwords ────────────────
GPE_PRESERVE = {
    # Regions and descriptors common in geopolitical news
    "north","south","east","west","central","border","region","zone","territory",
    "province","district","city","country","nation","state","republic","kingdom",
    "forces","troops","military","army","navy","government","minister","official",
    "president","parliament","congress","senate","council","security","defense",
    "intelligence","sanctions","conflict","crisis","war","peace","ceasefire",
    "refugees","civilians","casualties","attack","strike","operation","mission",
}

# ── Simple lemmatisation rules (suffix stripping) ────────────────────────────
LEMMA_SUFFIXES = [
    ("nesses", ""), ("ments", ""), ("ations", "ation"), ("ings", ""),
    ("ness", ""), ("ment", ""), ("ation", ""), ("ing", ""),
    ("edly", "ed"), ("edly", ""), ("ies", "y"), ("ied", "y"),
    ("er", ""), ("est", ""), ("ed", ""), ("ly", ""),
    ("s", ""),  # last resort
]

# ── Named entity patterns ─────────────────────────────────────────────────────
# Country / region names (representative subset; full list in production)
KNOWN_GPE = {
    "india","pakistan","china","russia","ukraine","israel","palestine","iran",
    "iraq","syria","afghanistan","myanmar","taiwan","north korea","south korea",
    "usa","united states","uk","britain","france","germany","nato","un","eu",
    "gaza","west bank","kashmir","donbas","crimea","tibet","xinjiang",
}

KNOWN_ORG = {
    "un","nato","eu","iaea","who","icrc","amnesty","human rights watch",
    "pentagon","kremlin","whitehouse","security council","general assembly",
}

MANIPULATION_VERBS = {
    "condemns","condemn","slams","slam","attacks","blasts","warns","accuses",
    "threatens","demands","declares","reveals","exposes","claims","alleges",
}


class Preprocessor:
    """
    Wraps spaCy if available; falls back to regex-based processing.
    Produces a preprocessed dict ready for all downstream modules.
    """

    def __init__(self, config):
        self.cfg = config
        self._nlp = None
        self._try_load_spacy()

    def _try_load_spacy(self):
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_lg")
            except OSError:
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self._nlp = None
        except ImportError:
            self._nlp = None

    # ── Public ────────────────────────────────────────────────────────────────
    def process(self, article: dict) -> dict:
        title = article.get("title", "")
        text  = article.get("text",  "")
        full_text = f"{title}. {text}"

        cleaned    = self._clean(full_text)
        tokens     = self._tokenise(cleaned)
        tokens_flt = self._filter_stopwords(tokens)
        lemmas     = self._lemmatise(tokens_flt)
        entities   = self._extract_entities(full_text, title)

        # Style features
        lex_div    = self._lexical_diversity(tokens_flt)
        spell_rate = self._spell_score(tokens)

        # Sentence list (for downstream stance / claim extraction)
        sentences  = self._split_sentences(full_text)

        return {
            **article,
            "cleaned_text":  cleaned,
            "tokens":        tokens,
            "filtered_tokens": tokens_flt,
            "lemmas":        lemmas,
            "entities":      entities,
            "sentences":     sentences,
            "lexical_diversity": lex_div,
            "spell_score":   spell_rate,
            "word_count":    len(tokens),
            "unique_words":  len(set(tokens)),
        }

    # ── Private ───────────────────────────────────────────────────────────────
    def _clean(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)              # HTML tags
        text = re.sub(r"https?://\S+", " ", text)          # URLs
        text = re.sub(r"[^\w\s\.\!\?\,\-\']", " ", text)  # keep punct
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def _tokenise(self, text: str) -> List[str]:
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens

    def _filter_stopwords(self, tokens: List[str]) -> List[str]:
        return [
            t for t in tokens
            if t not in STOPWORDS or t in GPE_PRESERVE
        ]

    def _lemmatise(self, tokens: List[str]) -> List[str]:
        if self._nlp:
            doc = self._nlp(" ".join(tokens))
            return [token.lemma_ for token in doc]
        # Fallback: suffix-stripping
        result = []
        for tok in tokens:
            lemma = tok
            if len(tok) > 5:
                for suffix, replacement in LEMMA_SUFFIXES:
                    if tok.endswith(suffix) and len(tok) - len(suffix) > 3:
                        lemma = tok[: len(tok) - len(suffix)] + replacement
                        break
            result.append(lemma)
        return result

    def _extract_entities(self, full_text: str, title: str) -> Dict[str, List[str]]:
        entities = {"GPE": [], "PERSON": [], "ORG": [], "DATE": []}

        if self._nlp:
            doc = self._nlp(full_text[:5000])   # cap for speed
            for ent in doc.ents:
                label = ent.label_
                if label in ("GPE", "LOC"):
                    entities["GPE"].append(ent.text.lower())
                elif label == "PERSON":
                    entities["PERSON"].append(ent.text.lower())
                elif label == "ORG":
                    entities["ORG"].append(ent.text.lower())
                elif label == "DATE":
                    entities["DATE"].append(ent.text.lower())
        else:
            # Fallback: keyword matching
            text_lower = full_text.lower()
            for gpe in KNOWN_GPE:
                if gpe in text_lower:
                    entities["GPE"].append(gpe)
            for org in KNOWN_ORG:
                if org in text_lower:
                    entities["ORG"].append(org)
            # Geopolitical keyword targets (used for DBS)
            for kw in self.cfg.geopolitical_keywords:
                if kw in text_lower:
                    entities["ORG"].append(kw)

        # Deduplicate
        for k in entities:
            entities[k] = list(dict.fromkeys(entities[k]))

        return entities

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _lexical_diversity(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / len(tokens), 4)

    def _spell_score(self, tokens: List[str]) -> float:
        """
        Simplified spell score: ratio of plausible English tokens
        (contains only standard characters, no random consonant clusters).
        Range [-1, 1]; higher = better spelling.
        """
        if not tokens:
            return 0.0
        vowels = set("aeiou")
        good = 0
        for tok in tokens:
            # Heuristic: at least one vowel, no run of 4+ consonants
            has_vowel = any(c in vowels for c in tok)
            consonant_run = max(
                (len(m.group()) for m in re.finditer(r"[^aeiou]+", tok)),
                default=0
            )
            if has_vowel and consonant_run < 4:
                good += 1
        ratio = good / len(tokens)
        return round(ratio * 2 - 1, 4)   # scale to [-1, 1]
