import streamlit as st
from scoring import score_transcript 


st.set_page_config(page_title="Nirmaan Speaking Evaluation Tool", layout="wide")

st.title("🗣️ Nirmaan AI - Speaking Skill Evaluation Tool")
st.write("Paste your transcript below and get a detailed rubric-based score.")

# Input Box 
transcript = st.text_area("Paste the transcript here:", height=250)

duration_sec = st.number_input(
    "Enter audio duration in seconds (optional, required for Speech Rate scoring):",
    min_value=1.0,
    max_value=600.0,
    value=52.0 
)

# Evaluate Button 
if st.button("Evaluate Transcript"):
    if len(transcript.strip()) == 0:
        st.error("Please paste a transcript before scoring.")
    else:
        with st.spinner("Scoring in progress..."):
            results = score_transcript(transcript, duration_sec)

        st.success("Scoring complete!")

        # Display Overall Score
        st.header("🏆 Overall Score")
        st.metric("Final Score (0–100)", results["overall_score"])

        # Content & Structure
        st.header("📘 Content & Structure (40 points)")
        cs = results["content_structure"]

        st.subheader("Raw Total (Before Scaling):")
        st.write(f"**{cs['raw_total']} / 30 points**")

        st.subheader("Scaled Score (Final):")
        st.write(f"**{cs['scaled_score']} / 40**")

        st.subheader("Per Category Breakdown:")
        for cat, details in cs["per_category"].items():
            with st.expander(f"🔹 {cat}"):
                st.write(details)

        # Speech Rate
        st.header("⏱️ Speech Rate (10 points)")
        sr = results["speech_rate"]
        st.write(f"**WPM:** {sr['wpm']}")
        st.write(f"**Points:** {sr['points']} / 10")

        # Language & Grammar
        st.header("📚 Language & Grammar (20 points)")
        lg = results["language_grammar"]
        st.write(f"**Grammar:** {lg['grammar_points']} / 10")
        st.write(f"**Vocabulary (TTR):** {lg['vocab_points']} / 10")
        st.write(f"**Total:** {lg['total']} / 20")

        # Clarity
        st.header("🔊 Clarity (15 points)")
        cl = results["clarity"]
        st.write(f"**Filler Word %:** {cl['filler_percent']}%")
        st.write(f"**Points:** {cl['points']} / 15")

        # Engagement
        st.header("💬 Engagement (15 points)")
        eg = results["engagement"]
        st.write(f"**Points:** {eg['points']} / 15")

        # JSON Output
        st.header("📦 Full JSON Output")
        st.json(results)
