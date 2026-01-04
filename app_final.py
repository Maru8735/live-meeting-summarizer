import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import threading
import tempfile
import os
import base64
try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
from datetime import datetime, timedelta
from models_enhanced import RealtimeSTTModel, EnhancedDiarizer, EnhancedSummarizer
from evaluation_enhanced import TranscriptionEvaluator, SummaryEvaluator, get_benchmark_report
from export_enhanced import (export_as_json, export_as_markdown, export_as_pdf, 
                             export_as_csv, send_email_summary, create_email_body)
from meeting_logger import MeetingLogger
import auth_utils

# Page Configuration
st.set_page_config(page_title="Live Meeting Summarizer", layout="wide", page_icon="🎙️")

# Function to get base64 of an image
def get_base64_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Try to load background image
bg_img_path = "background.jpg"
if os.path.exists(bg_img_path):
    bg_img_base64 = get_base64_bin_file(bg_img_path)
    bg_style = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("data:image/jpge;base64,{bg_img_base64}");
        background-attachment: fixed;
        background-size: cover;
    }}
    </style>
    """
else:
    bg_style = ""

# Custom CSS for Glassmorphism and Premium Look
st.markdown(bg_style + """
<style>
    /* Global Styles */
    .stApp {
        color: #ffffff;
    }
    
    h1, h2, h3, .stMarkdown {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }

    /* Glassmorphism Containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stExpander),
    div[data-testid="stVerticalBlock"] > div:has(div.stAlert),
    .stTabs [data-baseweb="tab-panel"],
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.2);
        padding: 10px;
        border-radius: 10px;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent !important;
        border-radius: 5px;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* Status Indicators */
    .status-recording { color: #ff4b4b; font-weight: bold; text-shadow: 0 0 10px rgba(255, 75, 75, 0.5); }
    .status-transcribing { color: #ffa500; font-weight: bold; text-shadow: 0 0 10px rgba(255, 165, 0, 0.5); }
    .status-summarizing { color: #00ccff; font-weight: bold; text-shadow: 0 0 10px rgba(0, 204, 255, 0.5); }
    .status-idle { color: #00ff88; font-weight: bold; text-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }

    /* Transcript Box */
    .transcript-box { 
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        padding: 15px; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px; 
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
        color: #e0e0e0;
    }

    /* Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        transform: translateY(-2px);
    }

    /* Success/Warning/Error Overrides */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Live Meeting Summarizer")
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
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None

# Audio Recording Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Authentication Logic
if not st.session_state.authenticated:
    st.title("🔐 Login to Meeting Summarizer")
    
    auth_tab1, auth_tab2 = st.tabs(["Login", "Sign Up"])
    
    with auth_tab1:
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary"):
            success, message = auth_utils.authenticate(login_user, login_pass)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.success(message)
                st.rerun()
            else:
                st.error(message)
                
    with auth_tab2:
        st.subheader("Create New Account")
        new_user = st.text_input("Choose Username", key="new_user")
        new_email = st.text_input("Enter Email", key="new_email")
        new_pass = st.text_input("Choose Password", type="password", key="new_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
        
        if st.button("Sign Up"):
            if not new_user or not new_pass or not new_email:
                st.warning("Please fill in all fields.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match.")
            else:
                success, message = auth_utils.save_user(new_user, new_pass, new_email)
                if success:
                    st.success(message)
                    st.info("You can now log in.")
                else:
                    st.error(message)
    st.stop()  # Stop execution here if not authenticated

# --- Main Application Starts Here (Only if authenticated) ---
st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

class AudioRecorder:
    """Threaded audio recorder"""

    def __init__(self, callback=None):
        self.callback = callback
        self.is_recording = False
        self.frames = []
        if PYAUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = None
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
            import traceback
            traceback.print_exc()

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
        if PYAUDIO_AVAILABLE:
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        else:
            wf.setsampwidth(2) # Default to 16-bit
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        return temp_file.name

    def cleanup(self):
        if self.audio:
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

st.sidebar.markdown("### Advanced Settings")
st.sidebar.markdown("To use **Pyannote Diarization**, provide your Hugging Face Token.")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password", help="Required for pyannote.audio. If empty, falls back to Vosk.")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token # Set env var for compatibility


# Status Bar
status_colors = {
    "Idle": "status-idle",
    "Recording": "status-recording",
    "Transcribing": "status-transcribing",
    "Summarizing": "status-summarizing"
}
current_status_class = status_colors.get(st.session_state.status, "status-idle")
st.markdown(f"<h3 class='{current_status_class}'>"             f"📡 Status: {st.session_state.status}</h3>", unsafe_allow_html=True)

# Main Interface
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎙️ Audio Input", 
    "📝 Transcript & Summary", 
    "📊 Analytics", 
    "💾 Export & Email",
    "📚 Session History"
])

with tab1:
    st.header("Input Source")
    input_mode = st.radio("Select Input Mode", ["Live Recording", "Upload Audio File"], horizontal=True)

    if input_mode == "Live Recording":
        recording_method = st.radio("Recording Method", ["Browser-Based (Cloud Friendly)", "Local Microphone (Real-time)"], horizontal=True)
        
        if recording_method == "Browser-Based (Cloud Friendly)":
            st.subheader("Real-Time Speech-to-Text (Browser-Based)")
            st.info("Uses your browser microphone. Works on Cloud!")
            
            from st_audiorec import st_audiorec
            wav_audio_data = st_audiorec()

            if wav_audio_data is not None:
                 # Manual Process Button
                if st.button("🚀 Process Recording", type="primary"):
                     st.write("Processing audio...")
                     with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                         tmp_file.write(wav_audio_data)
                         st.session_state.audio_file_path = tmp_file.name
                     
                     st.session_state.recording_start_time = datetime.now() 
                     st.session_state.status = "Processing"
                     st.rerun()
            else:
                 st.session_state.audio_file_path = None
                 st.session_state.status = "Idle"
        else:
            st.subheader("Real-time Transcription (Local Microphone)")
            
            if not PYAUDIO_AVAILABLE:
                st.error("❌ Local Microphone is not supported on this environment (Streamlit Cloud). Please use 'Browser-Based' recording.")
                st.stop()
                
            st.info("Uses your local system microphone. Best for local development.")
            
            col_rec1, col_rec2 = st.columns(2)
            
            # Start/Stop buttons
            if not st.session_state.recording:
                if col_rec1.button("🔴 Start Recording", type="primary", use_container_width=True):
                    # Reset state
                    st.session_state.realtime_transcript = []
                    st.session_state.recording = True
                    st.session_state.status = "Recording"
                    st.session_state.recording_start_time = datetime.now()
                    
                    # Initialize models
                    if 'stt_model' not in st.session_state or st.session_state.stt_model.model_name != ("vosk" if "Vosk" in model_choice else "whisper"):
                         st.session_state.stt_model = RealtimeSTTModel(
                            model_name="vosk" if "Vosk" in model_choice else "whisper",
                            callback=lambda text: st.session_state.realtime_transcript.append({"text": text, "time": time.time()})
                        )
                    
                    # Start recorder
                    st.session_state.recorder = AudioRecorder(callback=st.session_state.stt_model.audio_queue.put)
                    st.session_state.recorder.start_recording()
                    
                    # Start transcription
                    # Note: models_enhanced.py seems to expect the stream directly in start_realtime_transcription
                    # but also has an audio_queue. Let's adjust to use audio_queue if possible or pass recorder's stream.
                    # Looking at models_enhanced: _transcribe_stream calls audio_stream.read().
                    # Our AudioRecorder puts data into a queue.
                    
                    # Let's fix the model call or recorder to be compatible.
                    # For now, let's assume we need to pass an object with .read()
                    class QueueStream:
                        def __init__(self, q): self.q = q
                        def read(self, size, exception_on_overflow=False):
                            return self.q.get()
                    
                    st.session_state.stt_model.start_realtime_transcription(QueueStream(st.session_state.stt_model.audio_queue))
                    st.rerun()
            else:
                if col_rec1.button("⏹️ Stop & Process", type="secondary", use_container_width=True):
                    st.session_state.recording = False
                    st.session_state.status = "Processing"
                    
                    if 'recorder' in st.session_state:
                        audio_path = st.session_state.recorder.stop_recording()
                        st.session_state.audio_file_path = audio_path
                        st.session_state.recorder.cleanup()
                    
                    if 'stt_model' in st.session_state:
                         st.session_state.stt_model.stop_realtime_transcription()
                    
                    st.rerun()
            
            # Real-time transcript display
            transcript_placeholder = st.empty()
            with transcript_placeholder.status("Live Transcript", expanded=True):
                if st.session_state.realtime_transcript:
                    for item in st.session_state.realtime_transcript[-5:]: # Show last 5
                        st.write(f"{item['text']}")
                else:
                    st.write("Listening...")

    else:  # Upload Audio File
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader("Choose a file (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
        
        if uploaded_file is not None:
            if st.button("🚀 Process Uploaded File", type="primary"):
                # Save uploaded file to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state.audio_file_path = tmp_file.name
                
                st.session_state.status = "Processing"
                
                # Try to calculate duration for WAV files
                try:
                    import wave
                    import contextlib
                    with contextlib.closing(wave.open(st.session_state.audio_file_path, 'r')) as f:
                        frames = f.getnframes()
                        rate = f.getframerate()
                        duration = frames / float(rate)
                        # Set recording_start_time into the past so that (now - start) equals duration
                        st.session_state.recording_start_time = datetime.now() - timedelta(seconds=duration)
                except Exception:
                    # Fallback for non-wav files or errors: Default to now (0 duration logged initially)
                    # The diarization process might fix segments, but total duration might be off.
                    st.session_state.recording_start_time = datetime.now()

                st.success(f"File uploaded: {uploaded_file.name}")
                st.rerun()




    # Process recorded audio
    if st.session_state.audio_file_path and st.session_state.status == "Processing":
        with st.status("Processing Recording...", expanded=True) as status:
            try:
                # Calculate duration
                duration_seconds = (datetime.now() - st.session_state.recording_start_time).total_seconds()

                # Step 1: Transcription
                st.write("🎙️ Generating final transcription...")
                st.session_state.status = "Transcribing"

                # Transcription & Diarization Logic
                stt_model = RealtimeSTTModel(
                    model_name="vosk" if "Vosk" in model_choice else "whisper"
                )
                
                st.session_state.diarized_segments = None
                
                if use_diarization and "Vosk" in model_choice:
                    st.write("🎙️👥 Transcribing and Diarizing (Vosk Local)...")
                    transcript, segments = stt_model.transcribe_with_diarization(st.session_state.audio_file_path)
                    st.session_state.final_transcript = transcript
                    st.session_state.diarized_segments = segments
                    st.write(f"✓ Process complete. Found {len(segments) if segments else 0} segments.")
                else:
                    st.session_state.final_transcript = stt_model.transcribe(st.session_state.audio_file_path)
                    st.write(f"✓ Transcription complete ({len(st.session_state.final_transcript)} chars)")
                
                # --- VALIDATION CHECK: Empty Transcript ---
                if not st.session_state.final_transcript or not st.session_state.final_transcript.strip():
                     st.warning("⚠️ upload correct audio or recheck your audio file")
                     st.session_state.status = "Idle" # Reset status
                     st.stop() # Stop further processing (diarization/summary)
                
                # Additional Diarization (if using Whisper or Fallback or Pyannote requested)
                # Check if we should use Pyannote (if token present) - force it over Vosk logic if desired, or just use as fallback/enhancement
                # For now, let's allow it to override if token is present and user wants diarization
                
                if use_diarization and (not st.session_state.diarized_segments or hf_token):
                    if hf_token:
                         st.write("👥 Performing detailed speaker diarization (Pyannote)...")
                    else:
                         st.write("👥 Performing speaker diarization (Fallback)...")
                         
                    diarizer = EnhancedDiarizer(auth_token=hf_token)
                    segments = diarizer.diarize_and_transcribe(
                        st.session_state.audio_file_path,
                        st.session_state.final_transcript
                    )
                    
                    if segments:
                        st.session_state.diarized_segments = segments
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

    st.subheader("📊 Model Performance Comparison")
    benchmarks = get_benchmark_report()
    
    # Prepare data for plotting
    plot_data = []
    for model, metrics in benchmarks.items():
        plot_data.append({"Model": model, "Metric": "WER", "Value": metrics["WER"]})
        plot_data.append({"Model": model, "Metric": "CER", "Value": metrics["CER"]})
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create grouped bar chart
    fig = px.bar(
        df_plot, 
        x="Model", 
        y="Value", 
        color="Metric", 
        barmode="group",
        text_auto=".2f",
        title="WER & CER Comparison (Lower is Better)",
        color_discrete_map={"WER": "#ff4b4b", "CER": "#00ccff"}
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        title_font_size=20,
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Error Rate",
        template="plotly_dark",
        height=450
    )
    
    fig.update_traces(
        marker_line_color='rgba(255,255,255,0.2)',
        marker_line_width=1,
        opacity=0.9
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Table
    with st.expander("📄 View Detailed Benchmark Table"):
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
            st.markdown("---")
            st.markdown("**Sender Credentials** (Required for Gmail/SMTP)")
            col_email1, col_email2 = st.columns(2)
            with col_email1:
                sender_email = st.text_input("Your Email (Sender)", placeholder="you@gmail.com")
            with col_email2:
                sender_password = st.text_input("App Password", type="password", help="Use an App Password for Gmail, not your login password.")
            
            if st.button("📨 Send Email", type="primary"):
                if not sender_email or not sender_password:
                    st.error("Please provide both Sender Email and App Password.")
                else:
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

                        # Custom SMTP config
                        smtp_config = {
                            'host': 'smtp.gmail.com',
                            'port': 587,
                            'username': sender_email,
                            'password': sender_password
                        }

                        success, message = send_email_summary(email_recipient, subject, body, attachments, smtp_config=smtp_config)

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
