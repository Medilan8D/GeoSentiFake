# GeoSentiFake

GeoSentiFake measures emotional skew in geopolitical news and combines that signal with fake‑news and stance detection to produce an interpretable credibility verdict. It’s written in plain Python and can run in a lightweight demo mode (no ML models needed) or a full ML mode (transformers, embeddings, vector search, etc.). This README explains what the project does, how to try it quickly, and how the code is organised so you can extend it.

---

## Quick summary

- Purpose: Detect likely misinformation in geopolitical news while quantifying the emotional intensity and skew of the text (ESS), then fuse those signals into a single, explainable verdict.
- Who it’s for: Researchers and engineers working on misinformation detection, sentiment/emotion analysis in conflict reporting, or reproducible pipelines for NLP experiments.
- Modes: Offline demo (no heavy installs) or full ML pipeline (transformers, PyTorch, FAISS, sentence-transformers).

---

## Key features

- Lightweight demo classifier (heuristic) for fast experiments without downloading large models.
- Production-ready hooks for transformer-based fake-news classifiers and BI-LSTM models.
- Retrieval-Augmented Generation (RAG) style evidence retriever for credibility scoring.
- Emotion engine producing both Plutchik-style discrete probabilities and continuous VAD scores.
- ESS (Emotional Skew Score) computation and GSM (global skew multiplier) categorisation.
- Stance detection, Mixture-of-Experts fusion, and SHAP explainability pipeline.
- Streamlit app for quick visual/interactive analysis and a Jupyter notebook for exploration.

---

## Installation

There are three recommended tiers depending on what you want to do.

1. Tier 0 — Demo (no additional install)
   - Run the demo that uses the heuristic pipeline:
     - python geosentiafake.py --demo

2. Tier 1 — Recommended (full CPU-based ML pipeline)
   - Create a virtual environment and install core requirements:
     - python -m venv .venv
     - source .venv/bin/activate   # or .venv\Scripts\activate on Windows
     - pip install -r requirements.txt

3. Tier 2 — Production (GPU + large models + live web retrieval)
   - Install torch with a CUDA build appropriate for your GPU (example for CUDA 12.1):
     - pip install torch --index-url https://download.pytorch.org/whl/cu121
   - Install any optional GPU FAISS if required:
     - pip install faiss-gpu
   - Set API keys for live web retrieval (optional):
     - SERPAPI_KEY=your_key  OR  GOOGLE_CSE_KEY=key + GOOGLE_CSE_CX=cx

Requirements include transformers, torch, sentence-transformers, faiss (or faiss-cpu), spacy, nltk, shap, pandas, scikit-learn, matplotlib, seaborn and a few convenience utilities (see requirements.txt for recommended versions).

---

## Quickstart — run the demo

1. Run the built-in demo headlines (works with Tier 0):
   - python geosentiafake.py --demo

2. Run an input CSV of articles (columns: id,title,text,source,date,gdelt_region):
   - python geosentiafake.py --input path/to/articles.csv --output results.json

3. Interactive REPL mode:
   - python geosentiafake.py --interactive

4. Streamlit UI (simple visual front-end):
   - pip install streamlit
   - streamlit run app.py

The demo mode uses the offline heuristics so you can experiment without large downloads. Full model paths are referenced in the code and can be enabled by installing the ML dependencies and providing model checkpoints.

---

## Configuration & environment

- API keys for live retrieval (optional):
  - SERPAPI_KEY
  - GOOGLE_CSE_KEY and GOOGLE_CSE_CX
- Model checkpoints (optional):
  - If you have a fine-tuned FakeBERT or other checkpoints, adjust the paths in the model-loading sections (see fake_detector.py).
- Source trust configuration and thresholds are read from the project configuration (see config-like module for defaults). If you add a config file or .env, the code will pick those up when available.

---


