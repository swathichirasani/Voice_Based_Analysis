from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import streamlit as st

from audio_utils import (
    MAX_AUDIO_SECONDS,
    extract_audio_features_from_samples,
    load_audio,
    save_waveform_from_samples,
)
from report_generator import generate_pdf_report
from scoring_engine import evaluate_understanding, filler_word_ratio
from semantic_eval import DEFAULT_REFERENCE, semantic_similarity
from speech_to_text import speech_to_text


APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / "outputs"
MAX_UPLOAD_MB = 25


def load_css() -> None:
    css_path = APP_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "transcript": "",
        "audio_features": {},
        "similarity": 0.0,
        "filler": 0.0,
        "score": 0,
        "level": "",
        "color": "#f39c12",
        "waveform": "",
        "pdf": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def level_class(level: str) -> str:
    if level.startswith("Strong"):
        return "level-strong"
    if level.startswith("Poor"):
        return "level-poor"
    return "level-moderate"


def save_uploaded_file(uploaded_file) -> Path:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"Audio file is {size_mb:.1f} MB. Please upload a file up to "
            f"{MAX_UPLOAD_MB} MB for this deployment."
        )

    suffix = Path(uploaded_file.name).suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix, dir=OUTPUT_DIR) as temp:
        temp.write(uploaded_file.getbuffer())
        return Path(temp.name)


def render_header() -> None:
    st.title("Voice-Based Concept Understanding Analyser")
    st.markdown(
        '<p class="app-subtitle">Automated evaluation of spoken conceptual explanations using AI.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)


def render_reference(reference_text: str) -> None:
    st.markdown(
        f"""
        <div class="reference-box">
          <h2>Concept Reference</h2>
          <p>{reference_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(audio_path: Path, reference_text: str, transcript_override: str = "") -> None:
    transcript = transcript_override.strip() or speech_to_text(audio_path)
    has_transcript = bool(transcript)
    audio_samples, sample_rate = load_audio(audio_path)
    audio_features = extract_audio_features_from_samples(audio_samples, sample_rate)
    filler = filler_word_ratio(transcript) if has_transcript else 0.0
    similarity = semantic_similarity(transcript, reference_text) if has_transcript else 0.0
    score, level, color = evaluate_understanding(similarity, filler, audio_features)
    display_transcript = transcript or "No transcription provided. Audio-only evaluation was generated."

    waveform_path = OUTPUT_DIR / f"waveform_{uuid4().hex}.png"
    save_waveform_from_samples(audio_samples, sample_rate, waveform_path)

    metrics = {
        "semantic_similarity": similarity,
        "filler_ratio": filler,
        "pause_ratio": audio_features["pause_ratio"],
        "rms_energy": audio_features["rms_energy"],
        "score": score,
        "level": level,
    }
    pdf_path = OUTPUT_DIR / f"concept_report_{uuid4().hex}.pdf"
    generate_pdf_report(pdf_path, reference_text, display_transcript, waveform_path, metrics)

    st.session_state.transcript = display_transcript
    st.session_state.audio_features = audio_features
    st.session_state.similarity = similarity
    st.session_state.filler = filler
    st.session_state.score = score
    st.session_state.level = level
    st.session_state.color = color
    st.session_state.waveform = str(waveform_path)
    st.session_state.pdf = str(pdf_path)


def render_results(reference_text: str) -> None:
    if not st.session_state.transcript:
        st.markdown(
            '<div class="status-box">Upload an audio file to begin analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="success-strip">Analysis Completed</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("Transcribed Explanation")
        st.markdown(
            f'<div class="transcript-text">{st.session_state.transcript}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Final Evaluation")
        st.markdown('<div class="small-muted"><b>Understanding Score</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-number">{st.session_state.score}/100</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="level-label {level_class(st.session_state.level)}">{st.session_state.level}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        f"""
        <div class="metrics-grid">
          <div class="metric-tile">
            <div class="metric-label">Semantic Similarity</div>
            <div class="metric-value">{st.session_state.similarity:.2f}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">Filler Word Ratio</div>
            <div class="metric-value">{st.session_state.filler:.2f}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">Confidence (Energy)</div>
            <div class="metric-value">{st.session_state.audio_features.get("rms_energy", 0):.4f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pdf and Path(st.session_state.pdf).exists():
        st.download_button(
            "Download PDF Report",
            data=Path(st.session_state.pdf).read_bytes(),
            file_name="concept_understanding_report.pdf",
            mime="application/pdf",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Audio Visualization")
    if st.session_state.waveform and Path(st.session_state.waveform).exists():
        st.image(st.session_state.waveform)


def main() -> None:
    st.set_page_config(
        page_title="Voice-Based Concept Understanding Analyser",
        page_icon="V",
        layout="wide",
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    init_state()
    load_css()
    render_header()

    reference_text = st.text_area(
        "Reference Concept",
        value=DEFAULT_REFERENCE,
        height=95,
        help="Edit the target concept that the spoken explanation should align with.",
    )

    upload_col, reference_col = st.columns([1.5, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            f"Upload Student Audio (max {MAX_UPLOAD_MB} MB / {MAX_AUDIO_SECONDS} seconds)",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
        )
        if uploaded_file is not None:
            try:
                audio_path = save_uploaded_file(uploaded_file)
                st.audio(audio_path)
                transcript_override = st.text_area(
                    "Student Transcription (Optional)",
                    value="",
                    height=130,
                    placeholder="Paste the transcribed student explanation here for text-based similarity and filler analysis.",
                )
                if st.button("Analyze Concept Understanding", type="primary"):
                    with st.spinner("Processing and evaluating..."):
                        run_analysis(audio_path, reference_text, transcript_override)
                        st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
        else:
            st.markdown(
                '<div class="status-box">Upload an audio file to begin analysis.</div>',
                unsafe_allow_html=True,
            )

    with reference_col:
        render_reference(reference_text)

    st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)
    render_results(reference_text)


if __name__ == "__main__":
    main()
