

import argparse
import json
import sys
import os
import time
import logging
from pathlib import Path

# ── Local modules ─────────────────────────────────────────────────────────────
from pipeline.preprocessor    import Preprocessor
from pipeline.fake_detector   import FakeNewsDetector
from pipeline.emotion_engine  import EmotionEngine
from pipeline.ess_calculator  import ESSCalculator
from pipeline.stance_detector import StanceDetector
from pipeline.rag_retriever   import RAGRetriever
from pipeline.moe_fusion      import MoEFusion
from pipeline.explainer       import SHAPExplainer
from utils.display            import PipelineDisplay
from utils.config             import Config
from utils.logger             import setup_logger

# ─────────────────────────────────────────────────────────────────────────────
DEMO_ARTICLES = [
    {
        "id": "demo_001",
        "title": "Militants launch deadly cross-border attack, 40 civilians killed",
        "text": (
            "Armed militants crossed the border under cover of darkness, massacring forty "
            "innocent civilians in what officials are calling a savage act of terrorism. "
            "The enemy forces showed no mercy, deliberately targeting women and children "
            "in their homes. Government sources confirm this is an unprovoked act of war "
            "and demand immediate international condemnation of the aggressors."
        ),
        "source": "conflict-news-daily.net",
        "date": "2024-10-15",
        "gdelt_region": "South Asia",
    },
    {
        "id": "demo_002",
        "title": "Ceasefire talks collapse; new war crimes allegations emerge",
        "text": (
            "Diplomatic efforts collapsed today as both sides traded accusations of war "
            "crimes. Human rights groups have documented alleged atrocities committed "
            "by occupying forces, including destruction of civilian infrastructure and "
            "forced displacement. The international community has failed to act, leaving "
            "millions in fear for their lives as the conflict enters its deadliest phase."
        ),
        "source": "globalwatchnews.org",
        "date": "2024-10-20",
        "gdelt_region": "Middle East",
    },
    {
        "id": "demo_003",
        "title": "Government confirms 12 soldiers killed in disputed border region",
        "text": (
            "The Ministry of Defence confirmed today that twelve soldiers were killed "
            "during an exchange of fire in the disputed border region. Both governments "
            "have issued statements; each holds the other responsible. The UN Security "
            "Council has called an emergency session. Families of the deceased have been "
            "notified. The border crossing remains closed pending investigation."
        ),
        "source": "reuters.com",
        "date": "2024-10-18",
        "gdelt_region": "Eastern Europe",
    },
    {
        "id": "demo_004",
        "title": "Peace summit scheduled for next month, parties cautiously optimistic",
        "text": (
            "Representatives from both nations confirmed they will attend a peace summit "
            "scheduled for next month, hosted by a neutral third party. Negotiators "
            "expressed cautious optimism, noting that preliminary agreements on "
            "humanitarian corridors have been reached. Independent monitors will be "
            "present throughout the talks."
        ),
        "source": "bbc.com",
        "date": "2024-10-22",
        "gdelt_region": "Eastern Europe",
    },
    {
        "id": "demo_005",
        "title": "Foreign minister condemns 'genocidal aggression', calls for global uprising",
        "text": (
            "Our foreign minister issued a blistering condemnation today, calling the "
            "enemy's campaign a genocide against our people and demanding that every "
            "freedom-loving nation rise up in support. 'The world is watching a crime "
            "against humanity unfold in real time,' the minister declared, warning that "
            "silence is complicity. Citizens are urged to take to the streets and demand "
            "their governments act immediately before it is too late."
        ),
        "source": "state-media-outlet.gov",
        "date": "2024-10-19",
        "gdelt_region": "South Asia",
    },
]
# ─────────────────────────────────────────────────────────────────────────────


class GeoSentiFakePipeline:
    """
    Orchestrates all nine stages of the GeoSentiFake framework.

    Stage 1  — Preprocessing & NER
    Stage 2  — Fake News Classification (FakeBERT-style + BI-LSTM)
    Stage 3  — RAG Evidence Retrieval & Credibility Scoring
    Stage 4  — Emotion Engine (Plutchik 8-class + VAD continuous)
    Stage 5  — ESS Computation (EIS · DBS · CDS × GSM)
    Stage 6  — Stance Detection
    Stage 7  — MoE Fusion
    Stage 8  — SHAP Explainability
    Stage 9  — Output & Display
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg    = config
        self.log    = logger
        self.display = PipelineDisplay()

        self.log.info("Initialising GeoSentiFake pipeline components…")
        self.preprocessor    = Preprocessor(config)
        self.fake_detector   = FakeNewsDetector(config)
        self.rag_retriever   = RAGRetriever(config)
        self.emotion_engine  = EmotionEngine(config)
        self.ess_calculator  = ESSCalculator(config)
        self.stance_detector = StanceDetector(config)
        self.moe_fusion      = MoEFusion(config)
        self.explainer       = SHAPExplainer(config)
        self.log.info("All components ready.\n")

    def run_article(self, article: dict) -> dict:
        """Process a single article through the full 9-stage pipeline."""
        t0 = time.perf_counter()
        aid = article.get("id", "unknown")
        self.log.debug(f"Processing article [{aid}]")

        # ── Stage 1: Preprocessing 
        preprocessed = self.preprocessor.process(article)

        # ── Stage 2: Fake News Classification 
        detection = self.fake_detector.classify(preprocessed)

        # ── Stage 3: RAG Evidence Retrieval 
        evidence = self.rag_retriever.retrieve(preprocessed)

        # ── Stage 4: Emotion Engine 
        emotion = self.emotion_engine.analyse(preprocessed, evidence)

        # ── Stage 5: ESS Computation 
        ess = self.ess_calculator.compute(
            preprocessed, emotion, evidence,
            article.get("gdelt_region", "Unknown")
        )

        # ── Stage 6: Stance Detection 
        stance = self.stance_detector.detect(preprocessed, evidence)

        # ── Stage 7: MoE Fusion 
        fusion = self.moe_fusion.fuse(detection, ess, stance, evidence)

        # ── Stage 8: SHAP Explainability 
        explanation = self.explainer.explain(fusion)

        # ── Stage 9: Assemble Output 
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        result = {
            "article_id":   aid,
            "title":        article.get("title", ""),
            "source":       article.get("source", ""),
            "date":         article.get("date", ""),
            "latency_ms":   latency_ms,

            # Core outputs
            "label":        fusion["label"],
            "confidence":   fusion["confidence"],
            "ess_score":    ess["ess_score"],
            "ess_band":     ess["ess_band"],
            "gsm":          ess["gsm"],
            "gsm_level":    ess["gsm_level"],

            # Sub-scores
            "eis":          ess["eis"],
            "dbs":          ess["dbs"],
            "cds":          ess["cds"],

            # Emotion breakdown
            "emotions":     emotion["plutchik_probs"],
            "vad":          emotion["vad"],
            "dominant_emotion": emotion["dominant_emotion"],

            # Stance
            "stance":       stance["label"],
            "stance_conf":  stance["confidence"],

            # Detection details
            "fakebert_prob":   detection["fake_probability"],
            "bilstm_prob":     detection["bilstm_fake_probability"],
            "credibility":     evidence["credibility_score"],

            # Explanation
            "shap_top_features": explanation["top_features"],
            "verdict_text":      explanation["verdict_text"],
        }
        return result

    def run_batch(self, articles: list) -> list:
        results = []
        total = len(articles)
        self.display.print_header()
        for i, article in enumerate(articles, 1):
            self.display.print_progress(i, total, article.get("title", "")[:60])
            result = self.run_article(article)
            results.append(result)
            self.display.print_result_card(result)
        self.display.print_summary(results)
        return results


# ─────────────────────────────────────────────────────────────────────────────
def load_articles_from_csv(path: str) -> list:
    """Load articles from a CSV file with columns: id,title,text,source,date,gdelt_region"""
    import csv
    articles = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles.append({
                "id":           row.get("id", f"row_{len(articles)}"),
                "title":        row.get("title", ""),
                "text":         row.get("text", ""),
                "source":       row.get("source", ""),
                "date":         row.get("date", ""),
                "gdelt_region": row.get("gdelt_region", "Unknown"),
            })
    return articles


def interactive_mode(pipeline: GeoSentiFakePipeline):
    """REPL for live analysis of single articles."""
    print("\n" + "═" * 70)
    print("  GeoSentiFake  ·  Interactive Mode  (type 'quit' to exit)")
    print("═" * 70)
    idx = 0
    while True:
        print()
        title = input("  Article title : ").strip()
        if title.lower() in ("quit", "exit", "q"):
            break
        text = input("  Article text  : ").strip()
        if not text:
            text = title
        source = input("  Source domain : ").strip() or "unknown"
        region = input("  Region (e.g. 'South Asia') : ").strip() or "Unknown"
        idx += 1
        article = {
            "id": f"interactive_{idx:03d}",
            "title": title, "text": text,
            "source": source, "date": "2024-01-01",
            "gdelt_region": region,
        }
        result = pipeline.run_article(article)
        pipeline.display.print_result_card(result)


def main():
    parser = argparse.ArgumentParser(
        description="GeoSentiFake — Emotional Skew Quantification Pipeline"
    )
    parser.add_argument("--input",       type=str, help="Path to input CSV file")
    parser.add_argument("--output",      type=str, default="results.json",
                        help="Output JSON path (default: results.json)")
    parser.add_argument("--demo",        action="store_true",
                        help="Run on built-in demo headlines")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive REPL mode")
    parser.add_argument("--verbose",     action="store_true",
                        help="Enable verbose debug logging")
    args = parser.parse_args()

    logger = setup_logger(verbose=args.verbose)
    config = Config()
    pipeline = GeoSentiFakePipeline(config, logger)

    if args.interactive:
        interactive_mode(pipeline)
        return

    if args.demo or not args.input:
        articles = DEMO_ARTICLES
        logger.info(f"Running demo on {len(articles)} built-in articles.")
    else:
        if not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        articles = load_articles_from_csv(args.input)
        logger.info(f"Loaded {len(articles)} articles from {args.input}")

    results = pipeline.run_batch(articles)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
