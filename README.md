# Voice-Based Concept Understanding Analyser

This project is a Streamlit application for evaluating spoken conceptual explanations. It accepts an uploaded audio file, uses a student transcript, compares it against a reference concept, extracts audio confidence features, computes an understanding score, renders a waveform, and generates a downloadable PDF report.

## Project Structure

- `app.py` - Streamlit front end and workflow orchestration
- `audio_utils.py` - Audio loading, feature extraction, and waveform rendering
- `speech_to_text.py` - Optional Whisper-based transcription for local use
- `semantic_eval.py` - Lightweight semantic similarity computation
- `scoring_engine.py` - Filler word detection and understanding score classification
- `report_generator.py` - PDF report generation using ReportLab
- `style.css` - Streamlit visual styling
- `requirements.txt` - Python dependencies

Full code-level documentation is available in `Project_Documentation.docx`.

## Run Locally

```powershell
python -m venv vbcu_env
.\vbcu_env\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The application opens at `http://localhost:8501`.

For free cloud deployment, paste the transcript in the app before analysis. Local Whisper/Sentence-BERT models are intentionally not required because they exceed many free memory limits.

## Scoring Logic

The final score combines:

- Semantic similarity between the student transcript and reference concept
- Filler word ratio from the transcript
- Pause ratio from the uploaded audio
- RMS energy as a confidence proxy

Scores are classified as Strong, Moderate, or Poor Understanding.
