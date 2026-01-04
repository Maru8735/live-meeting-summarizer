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
        self.full_results = [] # Store full results for diarization

        if model_name == "vosk":
            try:
                from vosk import Model, KaldiRecognizer
                import json
                print("Loading Vosk model...")
                model_path = "model"
                if not os.path.exists(model_path):
                    print("Downloading Vosk model (first time only)...")
                    import urllib.request
                    import zipfile
                    
                    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
                    filename = "vosk-model-small-en-us-0.15.zip"
                    
                    # Download
                    urllib.request.urlretrieve(url, filename)
                    
                    # Unzip
                    print("Extracting model...")
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(".")
                    
                    # Rename
                    if os.path.exists("vosk-model-small-en-us-0.15"):
                        os.rename("vosk-model-small-en-us-0.15", model_path)
                    
                    # Cleanup
                    if os.path.exists(filename):
                        os.remove(filename)
                        
                self.model = Model(model_path)
                
                # Initialize Speaker Model
                self.spk_model = None
                spk_model_path = "model-spk"
                try:
                    from vosk import SpkModel
                    if not os.path.exists(spk_model_path):
                        print("Downloading Vosk Speaker model...")
                        url = "https://alphacephei.com/vosk/models/vosk-model-spk-0.4.zip"
                        filename = "vosk-model-spk-0.4.zip"
                        urllib.request.urlretrieve(url, filename)
                        with zipfile.ZipFile(filename, 'r') as zip_ref:
                            zip_ref.extractall(".")
                        if os.path.exists("vosk-model-spk-0.4"):
                            os.rename("vosk-model-spk-0.4", spk_model_path)
                        if os.path.exists(filename):
                            os.remove(filename)
                    
                    self.spk_model = SpkModel(spk_model_path)
                    print("✓ Vosk Speaker model loaded")
                    self.recognizer = KaldiRecognizer(self.model, 16000, self.spk_model)
                except Exception as e:
                    print(f"Speaker model not loaded: {e}")
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
            print("Start live transcription...")
            while self.is_running:
                try:
                    data = audio_stream.read(4000, exception_on_overflow=False)
                    if len(data) == 0:
                        continue
                        
                    if self.recognizer.AcceptWaveform(data):
                        res = json.loads(self.recognizer.Result())
                        self.full_results.append(res)
                        text = res.get('text', '')
                        if text:
                            print(f"Live Update: {text}")
                            if self.callback:
                                self.callback(text)
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        text = partial.get('partial', '')
                        if text and self.callback:
                            self.callback(f"[Partial] {text}")
                except Exception as e:
                    print(f"Stream error: {e}")
                    break
            
            # Capture Final Result when stopped
            try:
                res = json.loads(self.recognizer.FinalResult())
                self.full_results.append(res)
                text = res.get('text', '')
                if text:
                    print(f"Final Live Result: {text}")
                    if self.callback:
                        self.callback(text)
            except Exception as e:
                print(f"Final result error: {e}")
                
        else:
            print("Whisper real-time mode - using buffered chunks")

    def get_diarized_output(self):
        """Process stored live results for diarization"""
        if not self.full_results:
            return None
            
        # Add final result if needed? usually stored in stream loop but FinalResult needs handling
        # For simplicity, we process what we have.
        
        segments = []
        vectors = []
        
        for res in self.full_results:
            if 'spk' in res:
                vectors.append(res['spk'])
                segments.append({
                    'text': res.get('text', ''),
                    'start': 0, 'end': 0, # Live stream doesn't give easy timestamps without processing frames
                    'vector_idx': len(vectors) - 1
                })
            elif 'text' in res and res['text'].strip():
                 segments.append({
                    'text': res['text'],
                    'vector_idx': -1
                 })

        if not vectors:
            return None

        # Clustering Logic (Reused)
        import numpy as np
        X = np.array(vectors)
        speaker_labels = [-1] * len(X)
        speakers = []
        
        for i in range(len(X)):
            if speaker_labels[i] != -1:
                continue
            
            speaker_id = len(speakers) + 1
            speaker_labels[i] = speaker_id
            speakers.append(X[i])
            
            for j in range(i + 1, len(X)):
                if speaker_labels[j] == -1:
                    dot = np.dot(X[i], X[j])
                    norm = np.linalg.norm(X[i]) * np.linalg.norm(X[j])
                    sim = dot / norm
                    if sim > 0.75:
                        speaker_labels[j] = speaker_id
        
        final_segments = []
        for seg in segments:
            if seg.get('vector_idx', -1) != -1:
                lbl = speaker_labels[seg['vector_idx']]
                seg['speaker'] = f"Speaker {lbl}"
            else:
                seg['speaker'] = "Speaker 1"
            if 'vector_idx' in seg: del seg['vector_idx']
            # Fake timestamps for live flow if missing (simple accumulation)
            # A real implementation would track frames/time.
            final_segments.append(seg)
            
        return final_segments

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
                
                # Check and convert audio if needed
                if not audio_path.endswith(".wav"):
                    print(f"Converting {audio_path} to WAV...")
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(audio_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    # Overwrite/create new wav path
                    audio_path = audio_path.rsplit('.', 1)[0] + "_converted.wav"
                    audio.export(audio_path, format="wav")
                    print(f"Converted to: {audio_path}")
                
                # Check WAV properties even if extension is .wav
                try:
                    wf = wave.open(audio_path, "rb")
                except wave.Error:
                    # Might be a WAV container but different codec, try converting
                    print("Wave open failed (RIFF error), attempting conversion...")
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(audio_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio_path = audio_path + "_fixed.wav"
                    audio.export(audio_path, format="wav")
                    wf = wave.open(audio_path, "rb")

                if wf.getnchannels() != 1 or wf.getframerate() != 16000:
                    print(f"Audio format mismatch ({wf.getnchannels()}ch, {wf.getframerate()}Hz). Converting...")
                    wf.close()
                    from pydub import AudioSegment
                    audio = AudioSegment.from_wav(audio_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio_path = audio_path.replace(".wav", "_16k.wav")
                    audio.export(audio_path, format="wav")
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

    def _prepare_audio_file(self, audio_path):
        """Helper to prepare audio file for Vosk (convert if needed)"""
        import wave
        from pydub import AudioSegment
        
        # Check if file needs conversion (extension)
        if not audio_path.endswith(".wav"):
            print(f"Converting {audio_path} to WAV...")
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio_path = audio_path.rsplit('.', 1)[0] + "_converted.wav"
            audio.export(audio_path, format="wav")
            
        # Check if file relies on valid WAV header
        try:
            wf = wave.open(audio_path, "rb")
        except wave.Error:
             print("Wave open failed, attempting conversion...")
             audio = AudioSegment.from_file(audio_path)
             audio = audio.set_frame_rate(16000).set_channels(1)
             audio_path = audio_path + "_fixed.wav"
             audio.export(audio_path, format="wav")
             wf = wave.open(audio_path, "rb")
             
        # Check audio properties
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
             wf.close()
             print(f"Audio format mismatch. Converting to 16kHz Mono...")
             audio = AudioSegment.from_wav(audio_path)
             audio = audio.set_frame_rate(16000).set_channels(1)
             audio_path = audio_path.replace(".wav", "_16k.wav")
             audio.export(audio_path, format="wav")
             wf = wave.open(audio_path, "rb")
             
        return wf

    def transcribe_with_diarization(self, audio_path):
        """Transcribe and diarize using Vosk Speaker Model"""
        if self.model is None:
            return "ERROR: Model not loaded", []
            
        try:
            if self.model_name == "vosk":
                from vosk import KaldiRecognizer
                import json
                import numpy as np
                
                wf = self._prepare_audio_file(audio_path)
                
                # Enable speaker model if available
                if hasattr(self, 'spk_model') and self.spk_model:
                     rec = KaldiRecognizer(self.model, 16000, self.spk_model)
                else:
                     print("Warning: Speaker model not loaded, falling back to standard recognition")
                     rec = KaldiRecognizer(self.model, 16000)
                
                rec.SetWords(True)

                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        results.append(res)
                
                res = json.loads(rec.FinalResult())
                results.append(res)
                
                # Process results for diarization
                segments = []
                vectors = []
                
                for res in results:
                    if 'spk' in res:
                        vectors.append(res['spk'])
                        # Aggregate text from result
                        text = res.get('text', '')
                        start = 0.0
                        end = 0.0
                        if 'result' in res and len(res['result']) > 0:
                            start = res['result'][0]['start']
                            end = res['result'][-1]['end']
                        
                        segments.append({
                            'text': text,
                            'start': start,
                            'end': end,
                            'vector_idx': len(vectors) - 1
                        })
                    elif 'text' in res and res['text'].strip():
                         segments.append({
                            'text': res['text'],
                            'start': 0.0, 'end': 0.0, 'vector_idx': -1
                         })

                # Perform Clustering
                if vectors:
                    X = np.array(vectors)
                    # Simple Greedy Clustering based on Cosine Distance
                    # Distance = 1 - (u . v) / (|u| |v|)
                    
                    speaker_labels = [-1] * len(X)
                    speakers = [] # List of representative vectors (centroids)
                    
                    for i in range(len(X)):
                        if speaker_labels[i] != -1:
                            continue
                        
                        # Create new speaker cluster
                        speaker_id = len(speakers) + 1
                        speaker_labels[i] = speaker_id
                        speakers.append(X[i])
                        
                        # Find all similar vectors
                        for j in range(i + 1, len(X)):
                            if speaker_labels[j] == -1:
                                # Cosine similarity
                                dot_product = np.dot(X[i], X[j])
                                norm_i = np.linalg.norm(X[i])
                                norm_j = np.linalg.norm(X[j])
                                cosine_sim = dot_product / (norm_i * norm_j)
                                
                                # Threshold: 0.75 similarity => 0.25 distance
                                if cosine_sim > 0.75: 
                                    speaker_labels[j] = speaker_id
                    
                    # Assign labels to segments
                    final_segments = []
                    for seg in segments:
                        if seg['vector_idx'] != -1:
                            lbl = speaker_labels[seg['vector_idx']]
                            seg['speaker'] = f"Speaker {lbl}"
                        else:
                            seg['speaker'] = "Speaker 1"
                        
                        # Cleanup internal keys
                        del seg['vector_idx']
                        final_segments.append(seg)
                        
                    full_text = " ".join([s['text'] for s in final_segments])
                    print(f"✓ Diarization complete. Found {len(speakers)} speakers.")
                    return full_text, final_segments

                else:
                    print("No speaker vectors extracted.")
                    full_text = " ".join([r.get('text', '') for r in results if 'text' in r])
                    return full_text, None

            else:
                # Whisper fallback
                result = self.model.transcribe(audio_path)
                return result["text"], None
        except Exception as e:
            print(f"Diarization error: {e}")
            import traceback
            traceback.print_exc()
            return str(e), None


class EnhancedDiarizer:
    """Speaker Diarization using pyannote.audio"""

    def __init__(self, auth_token=None):
        self.use_pyannote = False
        try:
            from pyannote.audio import Pipeline
            
            # Use provided token or env var
            token = auth_token if auth_token else os.getenv("HF_TOKEN")
            
            if token:
                print("Loading Pyannote pipeline...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token
                )
                self.use_pyannote = True
                print("✓ Pyannote loaded successfully")
            else:
                 print("No HF Token provided for Pyannote.")
                 self.use_pyannote = False
                 
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
        
        # Check against minimum length to avoid hallucinations (CNN/DailyMail models tend to hallucinate on short text)
        word_count = len(text.split())
        if word_count < 50:
            return f"Note: Input too short for AI summary. Here is the transcript:\n\n{text}"

        # Format with speaker info if available
        if diarized_segments:
            formatted_text = self._format_diarized_text(diarized_segments)
        else:
            formatted_text = text

        if self.model_type == "groq" and os.getenv("GROQ_API_KEY"):
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
            input_len = len(words)
            
            # Dynamic length constraints
            max_len = min(130, int(input_len * 0.7))
            min_len = min(30, int(input_len * 0.3))
            
            if input_len > 500:
                text = " ".join(words[:500])
                max_len = 130
                min_len = 30

            summary = self.summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Transformer error: {e}")
            return self._simple_summary(text)

    def _simple_summary(self, text):
        """Fallback summary"""
        sentences = text.split(". ")
        return ". ".join(sentences[:3]) + "." if len(sentences) > 3 else text
