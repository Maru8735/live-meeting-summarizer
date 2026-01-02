import os
import threading
import queue
import time
import numpy as np
from pydub import AudioSegment
import soundfile as sf

class RealtimeSTTModel:
    """Real-time Speech-to-Text using Vosk and Whisper"""

    def __init__(self, model_name="vosk", callback=None):
        self.model_name = model_name
        self.model = None
        self.callback = callback
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.transcription_thread = None

        if model_name == "vosk":
            try:
                from vosk import Model, KaldiRecognizer
                import json
                print("Loading Vosk model...")
                model_path = "model"
                if not os.path.exists(model_path):
                    print("Downloading Vosk model (first time only)...")
                    os.system("wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
                    os.system("unzip vosk-model-small-en-us-0.15.zip")
                    os.system("mv vosk-model-small-en-us-0.15 model")
                self.model = Model(model_path)
                self.recognizer = KaldiRecognizer(self.model, 16000)
                self.recognizer.SetWords(True)
                print("✓ Vosk model loaded successfully")
            except Exception as e:
                print(f"ERROR loading Vosk: {e}")
                self.model = None
        else:
            try:
                import whisper
                print("Loading Whisper model...")
                self.model = whisper.load_model("base")
                print("✓ Whisper model loaded successfully")
            except ImportError as e:
                print(f"ERROR: Install whisper with: pip install openai-whisper")
                self.model = None

    def start_realtime_transcription(self, audio_stream):
        """Start real-time transcription in a separate thread"""
        self.is_running = True
        self.transcription_thread = threading.Thread(
            target=self._transcribe_stream,
            args=(audio_stream,),
            daemon=True
        )
        self.transcription_thread.start()

    def _transcribe_stream(self, audio_stream):
        """Process audio stream and transcribe in real-time"""
        import json

        if self.model_name == "vosk":
            while self.is_running:
                try:
                    data = audio_stream.read(4000, exception_on_overflow=False)
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '')
                        if text and self.callback:
                            self.callback(text)
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        text = partial.get('partial', '')
                        if text and self.callback:
                            self.callback(f"[Partial] {text}")
                except Exception as e:
                    print(f"Stream error: {e}")
                    break
        else:
            print("Whisper real-time mode - using buffered chunks")

    def stop_realtime_transcription(self):
        """Stop real-time transcription"""
        self.is_running = False
        if self.transcription_thread:
            self.transcription_thread.join(timeout=2)

    def transcribe(self, audio_path):
        """Transcribe complete audio file (for final processing)"""
        if self.model is None:
            return "ERROR: Model not loaded"

        try:
            if self.model_name == "vosk":
                from vosk import KaldiRecognizer
                import wave
                import json

                wf = wave.open(audio_path, "rb")
                rec = KaldiRecognizer(self.model, wf.getframerate())
                rec.SetWords(True)

                text_parts = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text_parts.append(result.get('text', ''))

                final_result = json.loads(rec.FinalResult())
                text_parts.append(final_result.get('text', ''))
                return ' '.join(text_parts)
            else:
                print(f"Transcribing: {audio_path}")
                result = self.model.transcribe(audio_path)
                text = result.get("text", "No speech detected")
                print(f"✓ Transcription complete. Text length: {len(text)} chars")
                return text
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            print(error_msg)
            return error_msg


class EnhancedDiarizer:
    """Speaker Diarization using pyannote.audio"""

    def __init__(self):
        self.use_pyannote = False
        try:
            from pyannote.audio import Pipeline
            print("Loading Pyannote pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv("HF_TOKEN")
            )
            self.use_pyannote = True
            print("✓ Pyannote loaded successfully")
        except Exception as e:
            print(f"Note: Pyannote not available ({e}). Using fallback.")
            self.use_pyannote = False

    def diarize_and_transcribe(self, audio_path, transcription_text):
        """Perform diarization and align with transcription"""
        try:
            print("Starting speaker diarization...")

            if not self.use_pyannote:
                return self._fallback_diarization(audio_path, transcription_text)

            # Run diarization
            diarization = self.pipeline(audio_path)

            # Convert to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "text": ""
                })

            # Align transcription with segments
            segments = self._align_text_with_segments(segments, transcription_text)

            print(f"✓ Diarization complete: {len(segments)} segments")
            return segments

        except Exception as e:
            print(f"Diarization error: {e}")
            return self._fallback_diarization(audio_path, transcription_text)

    def _align_text_with_segments(self, segments, text):
        """Align transcription text with speaker segments"""
        words = text.split()
        words_per_segment = len(words) // len(segments) if segments else len(words)

        for i, segment in enumerate(segments):
            start_idx = i * words_per_segment
            end_idx = (i + 1) * words_per_segment if i < len(segments) - 1 else len(words)
            segment["text"] = " ".join(words[start_idx:end_idx])

        return segments

    def _fallback_diarization(self, audio_path, text):
        """Fallback: single speaker"""
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000

        return [{
            "speaker": "Speaker 1",
            "start": 0.0,
            "end": duration,
            "text": text
        }]


class EnhancedSummarizer:
    """Enhanced summarizer with Groq LLaMA, T5, and BART support"""

    def __init__(self, model_type="groq"):
        self.model_type = model_type
        self.summarizer = None

        if model_type == "groq":
            try:
                from groq import Groq
                self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                print("✓ Groq LLaMA API initialized")
            except Exception as e:
                print(f"Groq not available: {e}")
                self.model_type = "transformers"

        if model_type == "transformers" or self.model_type == "transformers":
            try:
                from transformers import pipeline
                print("Loading summarizer model...")
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                print("✓ BART Summarizer loaded")
            except Exception as e:
                print(f"Transformers not available: {e}")
                self.summarizer = None

    def summarize(self, text, diarized_segments=None):
        """Generate summary from diarized transcript"""
        if not text or len(text.strip()) == 0:
            return "No text to summarize."

        # Format with speaker info if available
        if diarized_segments:
            formatted_text = self._format_diarized_text(diarized_segments)
        else:
            formatted_text = text

        if self.model_type == "groq":
            return self._summarize_with_groq(formatted_text)
        elif self.summarizer:
            return self._summarize_with_transformers(text)
        else:
            return self._simple_summary(text)

    def _format_diarized_text(self, segments):
        """Format diarized segments for summarization"""
        formatted = []
        for seg in segments:
            formatted.append(f"[{seg['speaker']}]: {seg['text']}")
        return "\n".join(formatted)

    def _summarize_with_groq(self, text):
        """Summarize using Groq LLaMA 3.1"""
        try:
            prompt = f"""Summarize this meeting transcript. Focus on key decisions, action items, and main discussion points.

Transcript:
{text[:4000]}

Provide a concise summary in bullet points."""

            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return self._simple_summary(text)

    def _summarize_with_transformers(self, text):
        """Summarize using HuggingFace transformers"""
        try:
            words = text.split()
            if len(words) > 500:
                text = " ".join(words[:500])

            summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Transformer error: {e}")
            return self._simple_summary(text)

    def _simple_summary(self, text):
        """Fallback summary"""
        sentences = text.split(". ")
        return ". ".join(sentences[:3]) + "." if len(sentences) > 3 else text
