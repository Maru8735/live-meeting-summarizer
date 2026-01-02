import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import datetime

# Import custom modules
try:
    from models_enhanced import EnhancedSummarizer, RealtimeSTTModel
    from evaluation_enhanced import get_benchmark_report
    from export_enhanced import export_as_json, export_as_markdown, export_as_pdf, export_as_csv
    from meeting_logger import MeetingLogger
except Exception as e:
    st.error(f"Module import error: {e}")

# Page Configuration
st.set_page_config(
    page_title="Live Meeting Summarizer",
    layout="wide",
    page_icon="🎙️",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .cloud-note {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎙️ Live Meeting Summarizer Pro</h1>', unsafe_allow_html=True)
st.markdown("*AI-powered transcription and summarization*", unsafe_allow_html=True)

# Cloud deployment note
st.markdown("""
<div class="cloud-note">
    <strong>📝 Cloud Version:</strong> Upload pre-recorded audio files for transcription and summarization.
    For real-time recording features, please use the local deployment.
</div>
""", unsafe_allow_html=True)

# Initialize logger
logger = MeetingLogger()

# Session State
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'session_id' not in st.session_state:
    st.session_state.session_id = None

# Sidebar configuration
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("### Model Selection")
summarizer_model = st.sidebar.selectbox(
    "Summarization Model",
    ["BART (Local)", "Groq LLaMA 3.1 (API)"],
    help="BART runs locally, Groq requires API key"
)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📝 Results", "📊 Analytics", "💾 Export"])

with tab1:
    st.header("Upload Meeting Recording")

    col1, col2 = st.columns([2, 1])

    with col1:
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a"],
            help="Upload your meeting recording (WAV, MP3, or M4A)"
        )

        if audio_file:
            st.audio(audio_file)
            file_details = {
                "Filename": audio_file.name,
                "FileType": audio_file.type,
                "FileSize": f"{audio_file.size / 1024:.2f} KB"
            }
            st.json(file_details)

    with col2:
        st.markdown("### Processing Options")
        use_whisper = st.checkbox("Auto-transcribe with Whisper", value=True)
        manual_input = st.checkbox("Or paste transcript manually", value=False)

    # Manual transcript input
    if manual_input:
        st.subheader("📝 Manual Transcript Input")
        manual_transcript = st.text_area(
            "Paste your meeting transcript here",
            height=200,
            placeholder="Enter or paste transcript..."
        )

        if st.button("📋 Use Manual Transcript", type="secondary"):
            st.session_state.transcript = manual_transcript
            st.success("✅ Transcript loaded!")

    # Auto-transcription
    if audio_file and use_whisper:
        if st.button("🎙️ Transcribe Audio", type="primary", use_container_width=True):
            with st.status("Processing audio...", expanded=True) as status:
                try:
                    # Save uploaded file temporarily
                    st.write("📁 Saving audio file...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_file.getbuffer())
                        tmp_path = tmp.name

                    # Transcribe
                    st.write("🎙️ Transcribing with Whisper (this may take a minute)...")
                    stt = RealtimeSTTModel(model_name="whisper")
                    transcript = stt.transcribe(tmp_path)

                    st.session_state.transcript = transcript

                    # Cleanup
                    os.unlink(tmp_path)

                    status.update(label="✅ Transcription complete!", state="complete")
                    st.success(f"Generated {len(transcript)} characters of transcript")

                except Exception as e:
                    status.update(label="❌ Transcription failed", state="error")
                    st.error(f"Error: {e}")
                    st.info("💡 Try pasting transcript manually using the checkbox above")

    # Summarization
    if st.session_state.transcript:
        st.markdown("---")
        st.subheader("✨ Generate Summary")

        if st.button("Generate AI Summary", type="primary", use_container_width=True):
            with st.spinner("Generating summary with AI..."):
                try:
                    # Configure API key if using Groq
                    if "Groq" in summarizer_model:
                        groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
                        if groq_key:
                            os.environ["GROQ_API_KEY"] = groq_key
                            model_type = "groq"
                        else:
                            st.warning("⚠️ Groq API key not found, using BART instead")
                            model_type = "transformers"
                    else:
                        model_type = "transformers"

                    # Generate summary
                    summarizer = EnhancedSummarizer(model_type=model_type)
                    summary = summarizer.summarize(st.session_state.transcript)

                    st.session_state.summary = summary

                    # Save session
                    metadata = {
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'model': summarizer_model,
                        'word_count': len(st.session_state.transcript.split())
                    }

                    segments = [{
                        "speaker": "Speaker 1",
                        "start": 0.0,
                        "end": 0.0,
                        "text": st.session_state.transcript
                    }]

                    session_id, _ = logger.save_session(
                        st.session_state.transcript,
                        summary,
                        segments,
                        metadata
                    )

                    st.session_state.session_id = session_id

                    st.success(f"✅ Summary generated! Session ID: {session_id}")
                    st.balloons()

                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

with tab2:
    if st.session_state.transcript or st.session_state.summary:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("📝 Full Transcript")
            if st.session_state.transcript:
                st.text_area(
                    "Transcript",
                    st.session_state.transcript,
                    height=400,
                    disabled=True
                )
                st.caption(f"Word count: {len(st.session_state.transcript.split())}")
            else:
                st.info("No transcript available")

        with col2:
            st.subheader("✨ AI Summary")
            if st.session_state.summary:
                st.info(st.session_state.summary)
                st.caption(f"Summary length: {len(st.session_state.summary)} characters")
            else:
                st.info("Generate summary in the Upload tab")
    else:
        st.info("👆 Upload and process audio in the Upload tab to see results")

with tab3:
    st.header("📊 Performance Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.transcript:
            st.metric("📝 Transcript Words", len(st.session_state.transcript.split()))
        else:
            st.metric("📝 Transcript Words", "N/A")

    with col2:
        if st.session_state.summary:
            st.metric("✨ Summary Length", f"{len(st.session_state.summary)} chars")
        else:
            st.metric("✨ Summary Length", "N/A")

    with col3:
        analytics = logger.get_analytics()
        if analytics:
            st.metric("📚 Total Sessions", analytics.get('total_meetings', 0))
        else:
            st.metric("📚 Total Sessions", 0)

    st.markdown("---")
    st.subheader("🤖 Model Benchmarks")

    try:
        benchmarks = get_benchmark_report()
        df = pd.DataFrame(benchmarks).T
        st.dataframe(df, use_container_width=True)
    except:
        st.info("Benchmark data not available")

    # Session history
    st.markdown("---")
    st.subheader("📚 Recent Sessions")
    sessions = logger.list_sessions()

    if sessions:
        df_sessions = pd.DataFrame(sessions[:10])  # Show last 10
        st.dataframe(df_sessions, use_container_width=True)
    else:
        st.info("No previous sessions found")

with tab4:
    if st.session_state.summary and st.session_state.transcript:
        st.header("💾 Export Options")

        metadata = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': summarizer_model,
            'session_id': st.session_state.session_id
        }

        segments = [{
            "speaker": "Speaker 1",
            "start": 0.0,
            "end": 0.0,
            "text": st.session_state.transcript
        }]

        st.subheader("📥 Download Formats")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            json_data = export_as_json({
                "session_id": st.session_state.session_id,
                "transcript": st.session_state.transcript,
                "summary": st.session_state.summary,
                "metadata": metadata
            })
            st.download_button(
                "📄 Download JSON",
                json_data,
                file_name=f"meeting_{st.session_state.session_id}.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            md_data = export_as_markdown(segments, st.session_state.summary, metadata)
            st.download_button(
                "📝 Download Markdown",
                md_data,
                file_name=f"meeting_{st.session_state.session_id}.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col3:
            try:
                pdf_data = export_as_pdf(segments, st.session_state.summary, metadata)
                if pdf_data:
                    st.download_button(
                        "📕 Download PDF",
                        pdf_data,
                        file_name=f"meeting_{st.session_state.session_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.button("📕 PDF (Error)", disabled=True, use_container_width=True)
            except:
                st.button("📕 PDF (N/A)", disabled=True, use_container_width=True)

        with col4:
            csv_data = export_as_csv(segments)
            st.download_button(
                "📊 Download CSV",
                csv_data,
                file_name=f"meeting_{st.session_state.session_id}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Preview section
        st.markdown("---")
        st.subheader("👀 Export Preview")

        preview_format = st.selectbox("Select format to preview", ["Markdown", "JSON"])

        if preview_format == "Markdown":
            st.code(md_data, language="markdown")
        else:
            st.code(json_data, language="json")

    else:
        st.info("📤 Process a meeting first to enable export options")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit, Whisper, and Transformers</p>
    <p>For campus placements and professional demonstrations</p>
</div>
""", unsafe_allow_html=True)
