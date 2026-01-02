import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Live Meeting Summarizer",
    layout="wide",
    page_icon="🎤"
)

# Custom CSS
st.markdown("""
<style>
    .segment-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .status-recording {
        color: #d32f2f;
        font-weight: bold;
    }
    .status-idle {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎤 Live Meeting Summarizer")
st.markdown("**AI-powered transcription, diarization, and summarization**")
st.markdown("---")

# Initialize session state
if "segments" not in st.session_state:
    st.session_state.segments = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# Sidebar settings
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select STT Model",
    ["Whisper (High Accuracy)", "Vosk (Fast Local)"]
)
use_diarization = st.sidebar.checkbox("Enable Speaker Diarization", value=True)
summarizer_choice = st.sidebar.selectbox(
    "Summarization Model",
    ["BART", "T5", "Groq LLaMA 3.1"]
)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Transcript", "Summary", "Export"])

with tab1:
    st.header("📁 Upload Meeting Recording")
    audio_file = st.file_uploader(
        "Upload meeting recording (WAV, MP3, M4A)",
        type=["wav", "mp3", "m4a"]
    )
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("🚀 Start Processing", key="process_btn"):
            with st.status("Processing Audio...", expanded=True) as status:
                try:
                    # Step 1: Initialize
                    st.write("✅ Step 1: Initializing models...")
                    st.session_state.processing_done = False
                    
                    # Step 2: Transcribe (Simulated for demo)
                    st.write("✅ Step 2: Transcribing audio...")
                    st.session_state.transcript = "This is a sample transcription. The meeting discusses quarterly goals, budget allocation, and team strategies."
                    
                    # Step 3: Diarization (Simulated)
                    if use_diarization:
                        st.write("✅ Step 3: Extracting speaker segments...")
                        st.session_state.segments = [
                            {
                                "speaker": "Speaker 1",
                                "start": "0:00",
                                "end": "0:30",
                                "text": "Let's discuss the quarterly goals."
                            },
                            {
                                "speaker": "Speaker 2",
                                "start": "0:30",
                                "end": "1:00",
                                "text": "We should increase sales by 20%."
                            },
                            {
                                "speaker": "Speaker 1",
                                "start": "1:00",
                                "end": "1:30",
                                "text": "Agreed. Let's allocate budget accordingly."
                            }
                        ]
                    
                    # Step 4: Summarize (Simulated)
                    st.write("✅ Step 4: Generating summary...")
                    st.session_state.summary = "The meeting focused on quarterly goals and budget allocation. Key decisions: increase sales target by 20%, adjust budget for Q1, and implement new team strategies."
                    
                    st.session_state.processing_done = True
                    status.update(label="Processing Complete!", state="complete", expanded=False)
                    st.success("✅ Successfully processed meeting!")
                    
                except Exception as e:
                    st.error(f"❌ Processing Error: {str(e)}")
                    status.update(label="Processing Failed", state="error", expanded=False)
    else:
        st.info("📤 Upload an audio file (WAV, MP3, or M4A) to get started!")

with tab2:
    st.header("📝 Transcript with Speaker Diarization")
    if st.session_state.processing_done:
        if st.session_state.segments:
            for i, seg in enumerate(st.session_state.segments, 1):
                st.markdown(f"""
                <div class="segment-box">
                    <strong>{seg['speaker']}</strong> ({seg['start']} - {seg['end']})<br>
                    {seg['text']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No speaker segments extracted.")
    else:
        st.warning("⚠️ Process an audio file first in the Upload tab.")

with tab3:
    st.header("📊 Summary")
    if st.session_state.processing_done:
        st.info(st.session_state.summary)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            word_count = len(st.session_state.transcript.split())
            st.metric("Total Words", word_count)
        with col2:
            speaker_count = len(set(seg['speaker'] for seg in st.session_state.segments))
            st.metric("Speakers Detected", speaker_count)
        with col3:
            summary_ratio = len(st.session_state.summary) / len(st.session_state.transcript) * 100
            st.metric("Compression Ratio", f"{summary_ratio:.1f}%")
    else:
        st.warning("⚠️ Process an audio file first in the Upload tab.")

with tab4:
    st.header("💾 Export Results")
    if st.session_state.processing_done:
        col1, col2, col3 = st.columns(3)
        
        # Export as JSON
        with col1:
            json_data = json.dumps({
                "transcript": st.session_state.transcript,
                "summary": st.session_state.summary,
                "segments": st.session_state.segments,
                "timestamp": datetime.now().isoformat()
            }, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="meeting_transcript.json",
                mime="application/json"
            )
        
        # Export as Markdown
        with col2:
            md_data = f"""# Meeting Summary
            
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
{st.session_state.summary}

## Transcript
"""
            for seg in st.session_state.segments:
                md_data += f"\n**{seg['speaker']}** ({seg['start']} - {seg['end']})\n{seg['text']}\n"
            
            st.download_button(
                label="📥 Download Markdown",
                data=md_data,
                file_name="meeting_transcript.md",
                mime="text/markdown"
            )
        
        # Export as CSV
        with col3:
            csv_data = "Speaker,Start,End,Text\n"
            for seg in st.session_state.segments:
                csv_data += f"{seg['speaker']},{seg['start']},{seg['end']},\"{seg['text']}\"\n"
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="meeting_transcript.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        st.subheader("Email Summary")
        email = st.text_input("Enter email address to send summary")
        if st.button("📧 Send Email"):
            st.success(f"✅ Summary would be sent to {email}")
    else:
        st.warning("⚠️ Process an audio file first to enable export.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    Live Meeting Summarizer | Powered by Streamlit & AI | v1.0
</div>
""", unsafe_allow_html=True)
