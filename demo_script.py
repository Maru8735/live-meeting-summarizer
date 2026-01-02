"""
Live Meeting Summarizer - Demo Script
======================================

This script demonstrates the core functionality of the Live Meeting Summarizer
with sample audio and walkthrough scenarios.
"""

import os
import sys
import time
from datetime import datetime
from models_enhanced import RealtimeSTTModel, EnhancedDiarizer, EnhancedSummarizer
from evaluation_enhanced import TranscriptionEvaluator, SummaryEvaluator
from meeting_logger import MeetingLogger
from export_enhanced import export_as_pdf, export_as_markdown

class DemoRunner:
    """Run demonstration scenarios"""

    def __init__(self):
        self.logger = MeetingLogger(log_dir="demo_logs")
        print("=" * 60)
        print("Live Meeting Summarizer - Demo Mode")
        print("=" * 60)
        print()

    def demo_transcription(self, audio_file="samples/sample_meeting.wav"):
        """Demonstrate transcription with sample audio"""
        print("\n[DEMO 1] Speech-to-Text Transcription")
        print("-" * 60)

        if not os.path.exists(audio_file):
            print(f"⚠️  Sample file not found: {audio_file}")
            print("Creating synthetic sample...")
            audio_file = self.create_sample_audio()

        print(f"📁 Input: {audio_file}")

        # Transcribe with Vosk
        print("\n🎙️ Transcribing with Vosk (fast)...")
        stt_vosk = RealtimeSTTModel(model_name="vosk")
        start_time = time.time()
        transcript_vosk = stt_vosk.transcribe(audio_file)
        vosk_time = time.time() - start_time

        print(f"✓ Vosk completed in {vosk_time:.2f}s")
        print(f"📝 Transcript preview: {transcript_vosk[:200]}...")

        # Transcribe with Whisper
        print("\n🎙️ Transcribing with Whisper (accurate)...")
        stt_whisper = RealtimeSTTModel(model_name="whisper")
        start_time = time.time()
        transcript_whisper = stt_whisper.transcribe(audio_file)
        whisper_time = time.time() - start_time

        print(f"✓ Whisper completed in {whisper_time:.2f}s")
        print(f"📝 Transcript preview: {transcript_whisper[:200]}...")

        # Compare
        evaluator = TranscriptionEvaluator()
        wer = evaluator.calculate_wer(transcript_whisper, transcript_vosk)

        print(f"\n📊 Performance Comparison:")
        print(f"   Vosk:    {vosk_time:.2f}s")
        print(f"   Whisper: {whisper_time:.2f}s")
        print(f"   WER:     {wer:.2%}")

        return transcript_whisper, audio_file

    def demo_diarization(self, audio_file, transcript):
        """Demonstrate speaker diarization"""
        print("\n[DEMO 2] Speaker Diarization")
        print("-" * 60)

        print("👥 Identifying speakers...")
        diarizer = EnhancedDiarizer()
        segments = diarizer.diarize_and_transcribe(audio_file, transcript)

        print(f"\n✓ Identified {len(set([s['speaker'] for s in segments]))} speakers")
        print(f"📋 Total segments: {len(segments)}\n")

        for i, seg in enumerate(segments[:3], 1):
            print(f"Segment {i}:")
            print(f"  Speaker: {seg['speaker']}")
            print(f"  Time: {seg['start']:.1f}s - {seg['end']:.1f}s")
            print(f"  Text: {seg['text'][:100]}...")
            print()

        if len(segments) > 3:
            print(f"... and {len(segments) - 3} more segments\n")

        return segments

    def demo_summarization(self, transcript, segments):
        """Demonstrate AI summarization"""
        print("\n[DEMO 3] AI Summarization")
        print("-" * 60)

        print("✨ Generating summary with BART...")
        summarizer = EnhancedSummarizer(model_type="transformers")
        summary = summarizer.summarize(transcript, segments)

        print(f"\n📄 Summary ({len(summary)} characters):")
        print("-" * 60)
        print(summary)
        print("-" * 60)

        # Evaluate summary
        evaluator = SummaryEvaluator()
        rouge_scores = evaluator.calculate_rouge(transcript, summary)

        print(f"\n📊 Summary Quality (ROUGE Scores):")
        print(f"   ROUGE-1 F1: {rouge_scores['ROUGE-1']['f1']:.4f}")
        print(f"   ROUGE-2 F1: {rouge_scores['ROUGE-2']['f1']:.4f}")
        print(f"   ROUGE-L F1: {rouge_scores['ROUGE-L']['f1']:.4f}")

        return summary

    def demo_export(self, transcript, summary, segments):
        """Demonstrate export functionality"""
        print("\n[DEMO 4] Export & Logging")
        print("-" * 60)

        metadata = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': '120s',
            'num_speakers': len(set([s['speaker'] for s in segments])),
            'model': 'Whisper + BART'
        }

        # Save session
        print("💾 Saving session...")
        session_id, json_path = self.logger.save_session(transcript, summary, segments, metadata)
        print(f"✓ Session saved: {session_id}")
        print(f"📁 JSON: {json_path}")

        # Export Markdown
        print("\n📝 Exporting to Markdown...")
        md_content = export_as_markdown(segments, summary, metadata)
        md_path = f"demo_logs/{session_id}.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"✓ Markdown saved: {md_path}")

        # Export PDF
        print("\n📕 Exporting to PDF...")
        pdf_content = export_as_pdf(segments, summary, metadata)
        if pdf_content:
            pdf_path = f"demo_logs/{session_id}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_content)
            print(f"✓ PDF saved: {pdf_path}")
        else:
            print("⚠️  PDF export requires reportlab")

        return session_id

    def demo_analytics(self):
        """Show analytics across sessions"""
        print("\n[DEMO 5] Session Analytics")
        print("-" * 60)

        analytics = self.logger.get_analytics()

        if analytics:
            print(f"📈 Total Meetings: {analytics['total_meetings']}")
            print(f"📝 Total Words: {analytics['total_words']:,}")
            print(f"⏱️  Avg Duration: {analytics['avg_duration']:.1f}s")
            print(f"👥 Avg Speakers: {analytics['avg_speakers']:.1f}")
            print(f"\n🤖 Models Used:")
            for model, count in analytics['models_used'].items():
                print(f"   {model}: {count}")
        else:
            print("No analytics available yet")

    def create_sample_audio(self):
        """Create a synthetic sample for testing"""
        print("\n🔊 Generating synthetic sample audio...")

        sample_text = """
        Hello everyone, thank you for joining today's meeting.
        Let's start with the quarterly review.
        Our sales have increased by 20% this quarter.
        The marketing team has done an excellent job.
        We should discuss the budget allocation for next quarter.
        Does anyone have questions or concerns?
        """

        # Note: This is a placeholder. In a real scenario, you would use TTS
        # For now, just create a text file as a reference
        os.makedirs("samples", exist_ok=True)
        sample_path = "samples/sample_transcript.txt"

        with open(sample_path, 'w') as f:
            f.write(sample_text)

        print(f"✓ Sample transcript saved: {sample_path}")
        print("⚠️  For full demo, please provide a real .wav file")

        return sample_path

    def run_full_demo(self, audio_file=None):
        """Run complete demonstration"""
        print("\n🚀 Starting Full Demo...")
        print("\n" + "=" * 60)

        try:
            # Demo 1: Transcription
            if audio_file and os.path.exists(audio_file):
                transcript, audio_path = self.demo_transcription(audio_file)

                # Demo 2: Diarization
                segments = self.demo_diarization(audio_path, transcript)

                # Demo 3: Summarization
                summary = self.demo_summarization(transcript, segments)

                # Demo 4: Export
                session_id = self.demo_export(transcript, summary, segments)

                # Demo 5: Analytics
                self.demo_analytics()

                print("\n" + "=" * 60)
                print("✅ Demo completed successfully!")
                print(f"📊 Session ID: {session_id}")
                print(f"📁 Check demo_logs/ for exported files")
                print("=" * 60)
            else:
                print("\n⚠️  No audio file provided")
                print("Usage: python demo_script.py [path/to/audio.wav]")
                print("\nRunning partial demo with text samples...")

                # Run text-only demos
                sample_text = "This is a sample meeting transcript for demonstration."
                sample_segments = [
                    {"speaker": "Speaker 1", "start": 0, "end": 5, "text": "Hello everyone."},
                    {"speaker": "Speaker 2", "start": 5, "end": 10, "text": "Thanks for joining."}
                ]

                summarizer = EnhancedSummarizer(model_type="transformers")
                summary = summarizer.summarize(sample_text, sample_segments)

                print(f"\n✨ Generated Summary:\n{summary}")

                session_id = self.demo_export(sample_text, summary, sample_segments)
                self.demo_analytics()

        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    runner = DemoRunner()

    # Check for audio file argument
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None

    runner.run_full_demo(audio_file)
