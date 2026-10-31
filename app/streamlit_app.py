from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predictor import analyze_announcement

st.set_page_config(page_title="BondBrief NLP", page_icon="📄", layout="wide")
st.title("📄 BondBrief NLP")
st.caption("Fixed-income announcement classification, entity extraction and risk-language screening")
st.warning("Demo only: predictions are not production decisions. Review every result against the original source.")

sample = ("ABC Finance Limited announces that the 8.25% secured bonds, ISIN INE123A08012, "
          "will pay interest on 15 December 2026. The coupon will be credited to eligible holders in INR.")
text = st.text_area("Paste a financial announcement", value=sample, height=180)
if st.button("Analyze announcement", type="primary"):
    try:
        result = analyze_announcement(text)
        a, b, c = st.columns(3)
        a.metric("Predicted category", result["category"])
        b.metric("Model confidence", f"{result['confidence']:.1%}")
        c.metric("Risk flags", len(result["risk_flags"]))
        st.subheader("Extracted entities")
        st.json(result["entities"])
        st.subheader("Risk-language flags")
        if result["risk_flags"]:
            st.error(", ".join(result["risk_flags"]))
        else:
            st.success("No configured risk phrases detected.")
        with st.expander("Class probabilities"):
            st.bar_chart(pd.Series(result["probabilities"]).sort_values(ascending=False))
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.subheader("Batch analysis")
upload = st.file_uploader("Upload CSV with a `text` column", type=["csv"])
if upload is not None:
    frame = pd.read_csv(upload)
    if "text" not in frame.columns:
        st.error("CSV must contain a column named `text`.")
    elif st.button("Analyze CSV"):
        rows = []
        for item in frame["text"].fillna("").astype(str):
            try:
                out = analyze_announcement(item)
                rows.append({"category": out["category"], "confidence": round(out["confidence"], 4),
                             "risk_flags": ", ".join(out["risk_flags"]), "isins": ", ".join(out["entities"]["isins"])})
            except Exception as exc:
                rows.append({"category": "ERROR", "confidence": 0, "risk_flags": str(exc), "isins": ""})
        result_df = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        st.dataframe(result_df, use_container_width=True)
        st.download_button("Download results", result_df.to_csv(index=False), "bondbrief_results.csv", "text/csv")
