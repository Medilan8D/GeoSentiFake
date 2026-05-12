# pipeline/main.py

from .preprocessor import Preprocessor
from .fake_detector import FakeNewsDetector
from .rag_retriever import RAGRetriever
from .emotion_engine import EmotionEngine
from .ess_calculator import ESSCalculator
from .stance_detector import StanceDetector
from .moe_fusion import MoEFusion
from .explainer import SHAPExplainer

class GeoSentiFakePipeline:
    """
    Main orchestrator that connects all 8 stages of the GeoSentiFake framework.
    """
    def __init__(self, config, logger):
        self.cfg = config
        self.logger = logger
        
        self.logger.info("Initializing GeoSentiFake Pipeline components...")
        self.preprocessor = Preprocessor(config)
        self.fake_detector = FakeNewsDetector(config)
        self.rag_retriever = RAGRetriever(config)
        self.emotion_engine = EmotionEngine(config)
        self.ess_calculator = ESSCalculator(config)
        self.stance_detector = StanceDetector(config)
        self.moe_fusion = MoEFusion(config)
        self.explainer = SHAPExplainer(config)
        self.logger.info("Pipeline initialized successfully.")

    def run_article(self, article: dict) -> dict:
        """
        Runs the full pipeline on a single article dictionary.
        """
        self.logger.info(f"Processing article: {article.get('title', 'Unknown Title')[:30]}...")

        # Stage 1: Preprocessing
        preprocessed = self.preprocessor.process(article)
        
        # Stage 2: Fake News Classification (Base)
        detection = self.fake_detector.classify(preprocessed)
        
        # Stage 3: RAG Retrieval
        evidence = self.rag_retriever.retrieve(preprocessed)
        
        # Stage 4: Emotion Detection
        emotion = self.emotion_engine.analyse(preprocessed, evidence)
        
        # Stage 5: ESS Computation
        region = article.get("gdelt_region", "Unknown")
        ess = self.ess_calculator.compute(preprocessed, emotion, evidence, region)
        
        # Stage 6: Stance Detection
        stance = self.stance_detector.detect(preprocessed, evidence)
        
        # Stage 7: MoE Fusion (Inject dominant emotion into fusion for the explainer)
        fusion = self.moe_fusion.fuse(detection, ess, stance, evidence)
        fusion["_dominant_emotion"] = emotion["dominant_emotion"] 
        
        # Stage 8: Explainability & Verdict
        explanation = self.explainer.explain(fusion)

        # Format the final output to match what your app.py expects
        return {
            "label": fusion["label"],
            "confidence": fusion["confidence"],
            "ess_score": ess["ess_score"],
            "source": article.get("source", "unknown"),
            "dominant_emotion": emotion["dominant_emotion"],
            "stance": stance["label"],
            "stance_conf": stance["confidence"],
            "eis": ess["eis"],
            "dbs": ess["dbs"],
            "cds": ess["cds"],
            "gsm": ess["gsm"],
            "emotions": emotion["plutchik_probs"],
            "shap_top_features": dict(explanation["top_features"]),
            "verdict_text": explanation["verdict_text"]
        }