import streamlit as st
import pandas as pd
import time
import threading
import tempfile
import os
import pyaudio
import wave
from datetime import datetime
from models_enhanced import RealtimeSTTModel, EnhancedDiarizer, EnhancedSummarizer
from evaluation_enhanced import TranscriptionEvaluator, SummaryEvaluator, get_benchmark_report
from export_enhanced import (export_as_json, export_as_markdown, export_as_pdf, 
                             export_as_csv, send_email_summary, create_email_body)
from meeting_logger import MeetingLogger

# Page Configuration
st.set_page_config(page_title="Live Meeting Summarizer", layout="wide", page_icon="🎙️")

# Custom CSS
st.markdown("""
<style>
    .status-recording { color: #ff4b4b; font-weight: bold; }
    .status-transcribing { color: #ffa500; font-weight: bold; }
    .status-summarizing { color: #0066ff; font-weight: bold; }
    .status-idle { color: #28a745; font-weight: bold; }
    .transcript-box { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 5px; 
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .speaker-label { color: #0066ff; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Live Meeting Summarizer Pro")
st.markdown("*Real-time transcription, speaker diarization, and AI summarization*")
st.markdown("---")

# Initialize logger
logger = MeetingLogger()

# Session State Initialization
if 'status' not in st.session_state:
    st.session_state.status = "Idle"
if 'realtime_transcript' not in st.session_state:
    st.session_state.realtime_transcript = []
if 'final_transcript' not in st.session_state:
    st.session_state.final_transcript = ""
if 'diarized_segments' not in st.session_state:
    st.session_state.diarized_segments = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None
if 'wer_score' not in st.session_state:
    st.session_state.wer_score = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = None

# Audio Recording Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class AudioRecorder:
    """Threaded audio recorder"""

    def __init__(self, callback=None):
        self.callback = callback
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording_thread = None

    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.recording_thread = threading.Thread(target=self._record, daemon=True)
        self.recording_thread.start()

    def _record(self):
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            while self.is_recording:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                if self.callback:
                    self.callback(data)
        except Exception as e:
            print(f"Recording error: {e}")

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wf = wave.open(temp_file.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        return temp_file.name

    def cleanup(self):
        self.audio.terminate()

# Sidebar Settings
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("### Model Configuration")
model_choice = st.sidebar.selectbox("STT Model", ["Vosk (Real-time)", "Whisper (High Accuracy)"])
summarizer_choice = st.sidebar.selectbox("Summarizer", ["Groq LLaMA 3.1", "BART", "T5"])
use_diarization = st.sidebar.checkbox("Enable Speaker Diarization", value=True)

st.sidebar.markdown("### Export & Notifications")
enable_email = st.sidebar.checkbox("Enable Email Notifications", value=False)
if enable_email:
    email_recipient = st.sidebar.text_input("Recipient Email", "")

# Status Bar
status_colors = {
    "Idle": "status-idle",
    "Recording": "status-recording",
    "Transcribing": "status-transcribing",
    "Summarizing": "status-summarizing"
}
st.markdown(f"<h3 class='{status_colors.get(st.session_state.status, "status-idle")}'>"             f"📡 Status: {st.session_state.status}</h3>", unsafe_allow_html=True)

# Main Interface
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎙️ Live Recording", 
    "📝 Transcript & Summary", 
    "📊 Analytics", 
    "💾 Export & Email",
    "📚 Session History"
])

with tab1:
    st.header("Real-Time Speech-to-Text")

    col1, col2, col3 = st.columns(3)

    with col1:
        if not st.session_state.recording:
            if st.button("🔴 Start Recording", type="primary", use_container_width=True):
                st.session_state.recording = True
                st.session_state.status = "Recording"
                st.session_state.realtime_transcript = []
                st.session_state.recording_start_time = datetime.now()

                def transcription_callback(text):
                    st.session_state.realtime_transcript.append({
                        "time": time.time(),
                        "text": text
                    })

                st.session_state.recorder = AudioRecorder()
                st.session_state.recorder.start_recording()
                st.rerun()

    with col2:
        if st.session_state.recording:
            if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True):
                st.session_state.recording = False
                st.session_state.status = "Processing"

                audio_path = st.session_state.recorder.stop_recording()
                st.session_state.audio_file_path = audio_path
                st.session_state.recorder.cleanup()
                st.rerun()

    with col3:
        if st.session_state.recording and st.session_state.recording_start_time:
            elapsed = (datetime.now() - st.session_state.recording_start_time).total_seconds()
            st.metric("⏱️ Duration", f"{int(elapsed)}s")

    # Real-time transcript display
    if st.session_state.recording or len(st.session_state.realtime_transcript) > 0:
        st.subheader("📄 Live Transcription")
        transcript_container = st.container()

        with transcript_container:
            st.markdown('<div class="transcript-box">', unsafe_allow_html=True)
            for item in st.session_state.realtime_transcript[-20:]:
                st.text(item['text'])
            st.markdown('</div>', unsafe_allow_html=True)

    # Process recorded audio
    if st.session_state.audio_file_path and st.session_state.status == "Processing":
        with st.status("Processing Recording...", expanded=True) as status:
            try:
                # Calculate duration
                duration_seconds = (datetime.now() - st.session_state.recording_start_time).total_seconds()

                # Step 1: Transcription
                st.write("🎙️ Generating final transcription...")
                st.session_state.status = "Transcribing"

                stt_model = RealtimeSTTModel(
                    model_name="vosk" if "Vosk" in model_choice else "whisper"
                )
                st.session_state.final_transcript = stt_model.transcribe(st.session_state.audio_file_path)
                st.write(f"✓ Transcription complete ({len(st.session_state.final_transcript)} chars)")

                # Step 2: Diarization
                if use_diarization:
                    st.write("👥 Performing speaker diarization...")
                    diarizer = EnhancedDiarizer()
                    st.session_state.diarized_segments = diarizer.diarize_and_transcribe(
                        st.session_state.audio_file_path,
                        st.session_state.final_transcript
                    )
                    st.write(f"✓ Identified {len(st.session_state.diarized_segments)} speaker segments")
                else:
                    st.session_state.diarized_segments = [{
                        "speaker": "Speaker 1",
                        "start": 0,
                        "end": duration_seconds,
                        "text": st.session_state.final_transcript
                    }]

                # Step 3: Summarization
                st.write("✨ Generating AI summary...")
                st.session_state.status = "Summarizing"

                model_map = {
                    "Groq LLaMA 3.1": "groq",
                    "BART": "transformers",
                    "T5": "transformers"
                }
                summarizer = EnhancedSummarizer(model_type=model_map.get(summarizer_choice, "transformers"))
                st.session_state.summary = summarizer.summarize(
                    st.session_state.final_transcript,
                    st.session_state.diarized_segments if use_diarization else None
                )
                st.write(f"✓ Summary generated ({len(st.session_state.summary)} chars)")

                # Step 4: Evaluation
                st.write("📊 Calculating metrics...")
                evaluator = TranscriptionEvaluator()
                realtime_text = " ".join([item['text'].replace('[Partial]', '').strip() 
                                         for item in st.session_state.realtime_transcript])
                if realtime_text and st.session_state.final_transcript:
                    st.session_state.wer_score = evaluator.calculate_wer(
                        st.session_state.final_transcript,
                        realtime_text
                    )

                # Step 5: Save session
                st.write("💾 Saving session...")
                metadata = {
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'duration': f"{int(duration_seconds)}s",
                    'num_speakers': len(set([s['speaker'] for s in st.session_state.diarized_segments])),
                    'model': model_choice,
                    'summarizer': summarizer_choice,
                    'metrics': {
                        'wer': st.session_state.wer_score if st.session_state.wer_score else 0
                    }
                }

                session_id, json_path = logger.save_session(
                    st.session_state.final_transcript,
                    st.session_state.summary,
                    st.session_state.diarized_segments,
                    metadata
                )
                st.session_state.session_id = session_id
                st.write(f"✓ Session saved: {session_id}")

                st.session_state.status = "Idle"
                status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                st.success(f"Meeting processed successfully! Session ID: {session_id}")

            except Exception as e:
                st.error(f"Processing error: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.status = "Idle"

with tab2:
    if st.session_state.diarized_segments:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("📝 Diarized Transcript")
            for seg in st.session_state.diarized_segments:
                with st.expander(f"🗣️ {seg['speaker']} ({seg['start']:.1f}s - {seg['end']:.1f}s)", expanded=False):
                    st.write(seg['text'])

        with col2:
            st.subheader("✨ AI Summary")
            st.info(st.session_state.summary)

            if st.session_state.wer_score is not None:
                st.metric("📊 Word Error Rate", f"{st.session_state.wer_score:.2%}")
    else:
        st.info("👆 Start a recording to see transcript and summary")

with tab3:
    st.header("📊 Performance Analytics")

    if st.session_state.wer_score is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🎯 WER Score", f"{st.session_state.wer_score:.2%}")
        with col2:
            st.metric("📝 Transcript", f"{len(st.session_state.final_transcript)} chars")
        with col3:
            st.metric("👥 Speakers", len(set([s['speaker'] for s in st.session_state.diarized_segments])))
        with col4:
            st.metric("⏱️ Segments", len(st.session_state.diarized_segments))

    st.subheader("Model Benchmarks")
    benchmarks = get_benchmark_report()
    df = pd.DataFrame(benchmarks).T
    st.dataframe(df, use_container_width=True)

    # Overall analytics
    st.subheader("📈 Overall Statistics")
    analytics = logger.get_analytics()
    if analytics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Meetings", analytics['total_meetings'])
        with col2:
            st.metric("Total Words", f"{analytics['total_words']:,}")
        with col3:
            st.metric("Avg Duration", f"{analytics['avg_duration']:.0f}s")

with tab4:
    if st.session_state.diarized_segments and st.session_state.summary:
        st.header("💾 Export & Email")

        # Prepare metadata
        metadata = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': f"{int((datetime.now() - st.session_state.recording_start_time).total_seconds())}s" if st.session_state.recording_start_time else "N/A",
            'num_speakers': len(set([s['speaker'] for s in st.session_state.diarized_segments])),
            'model': model_choice
        }

        st.subheader("📥 Download Options")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            json_data = export_as_json({
                "transcript": st.session_state.final_transcript,
                "segments": st.session_state.diarized_segments,
                "summary": st.session_state.summary,
                "metadata": metadata
            })
            st.download_button(
                "📄 JSON",
                json_data,
                file_name=f"meeting_{st.session_state.session_id}.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            md_data = export_as_markdown(st.session_state.diarized_segments, st.session_state.summary, metadata)
            st.download_button(
                "📝 Markdown",
                md_data,
                file_name=f"meeting_{st.session_state.session_id}.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col3:
            pdf_data = export_as_pdf(st.session_state.diarized_segments, st.session_state.summary, metadata)
            if pdf_data:
                st.download_button(
                    "📕 PDF",
                    pdf_data,
                    file_name=f"meeting_{st.session_state.session_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button("📕 PDF", disabled=True, use_container_width=True)
                st.caption("Install reportlab")

        with col4:
            csv_data = export_as_csv(st.session_state.diarized_segments)
            st.download_button(
                "📊 CSV",
                csv_data,
                file_name=f"meeting_{st.session_state.session_id}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Email section
        st.subheader("📧 Email Summary")

        if enable_email and email_recipient:
            if st.button("📨 Send Email", type="primary"):
                with st.spinner("Sending email..."):
                    subject = f"Meeting Summary – {metadata['date']}"
                    body = create_email_body(st.session_state.summary, st.session_state.diarized_segments, metadata)

                    # Prepare attachments
                    attachments = {
                        f"meeting_{st.session_state.session_id}.md": md_data.encode('utf-8'),
                        f"meeting_{st.session_state.session_id}.json": json_data.encode('utf-8')
                    }

                    if pdf_data:
                        attachments[f"meeting_{st.session_state.session_id}.pdf"] = pdf_data

                    success, message = send_email_summary(email_recipient, subject, body, attachments)

                    if success:
                        st.success(f"✅ Email sent to {email_recipient}")
                    else:
                        st.error(f"❌ {message}")
        else:
            st.info("Enable email notifications in sidebar and provide recipient email")
    else:
        st.info("📤 Complete a recording to export results")

with tab5:
    st.header("📚 Session History")

    sessions = logger.list_sessions()

    if sessions:
        st.write(f"**Total Sessions:** {len(sessions)}")

        # Display sessions table
        df_sessions = pd.DataFrame(sessions)
        st.dataframe(df_sessions, use_container_width=True)

        # Load specific session
        st.subheader("🔍 Load Previous Session")
        selected_session = st.selectbox("Select Session", [s['session_id'] for s in sessions])

        if st.button("Load Session"):
            session_data = logger.load_session(selected_session)
            if session_data:
                st.session_state.final_transcript = session_data['transcript']['full_text']
                st.session_state.summary = session_data['summary']['text']
                st.session_state.diarized_segments = session_data['speakers']['segments']
                st.session_state.session_id = selected_session
                st.success(f"Loaded session: {selected_session}")
                st.rerun()
    else:
        st.info("No previous sessions found. Start recording to create your first session!")
