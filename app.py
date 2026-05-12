import streamlit as st
import logging
from utils.config import Config
from utils.logger import setup_logger
from pipeline import GeoSentiFakePipeline #import GeoSentiFakePipeline  # assuming your pipeline file is main.py

# Initialize
st.set_page_config(page_title="GeoSentiFake", layout="wide")
st.title("GeoSentiFake — Fake News Detection with Emotional Skew Analysis")

# Setup pipeline (cached to avoid reloading every time)
@st.cache_resource
def load_pipeline():
    logger = setup_logger(verbose=False)
    config = Config()
    return GeoSentiFakePipeline(config, logger)

pipeline = load_pipeline()

# Sidebar
st.sidebar.header("Settings")
region = st.sidebar.selectbox("Select Region", [
    "South Asia", "Middle East", "Eastern Europe","West Asia", "Global", "Unknown"
])
source = st.sidebar.text_input("Source", "unknown")

# Main input
st.subheader("Enter News Article")
title = st.text_input("Title")
text = st.text_area("Article Text", height=200)

if st.button("Analyze"):
    if not title and not text:
        st.warning("Please enter title or text")
    else:
        article = {
            "id": "streamlit_001",
            "title": title,
            "text": text if text else title,
            "source": source,
            "date": "2024-01-01",
            "gdelt_region": region
        }

        with st.spinner("Analyzing..."):
            result = pipeline.run_article(article)

        # Output Section
        st.success("Analysis Complete")

        col1, col2, col3 = st.columns(3)

        col1.metric("Label", result["label"])
        col2.metric("Confidence", f"{result['confidence']:.2f}")
        col3.metric("ESS Score", f"{result['ess_score']:.2f}")

        st.subheader("Detailed Analysis")

        st.write(f"**Source:** {result['source']}")
        st.write(f"**Dominant Emotion:** {result['dominant_emotion']}")
        st.write(f"**Stance:** {result['stance']} ({result['stance_conf']:.2f})")

        st.subheader("ESS Breakdown")
        st.write({
            "EIS": result["eis"],
            "DBS": result["dbs"],
            "CDS": result["cds"],
            "GSM": result["gsm"],
        })

        st.subheader("Emotion Distribution")
        st.bar_chart(result["emotions"])

        st.subheader("Explainability (Top Features)")
        st.write(result["shap_top_features"])

        st.subheader("Verdict")
        st.info(result["verdict_text"])

# Footer
st.markdown("---")
st.caption("GeoSentiFake Pipeline")
