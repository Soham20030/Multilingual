import logging
from typing import Dict, Any
import uuid
import os
import time
import speech_recognition as sr
from utils.logger import get_logger
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.replay_buffer import replay_buffer
from config.settings import MODEL_CONFIG

logger = get_logger(__name__)

class AudioAgent:
    """Agent for processing audio inputs using multiple fallback methods."""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.model_config = MODEL_CONFIG.get("edumentor_agent", {})
        
        # Try to initialize Wav2Vec2 if available
        self.wav2vec2_available = False
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2_available = True
            logger.info("Wav2Vec2 model loaded successfully")
        except Exception as e:
            logger.warning(f"Wav2Vec2 not available: {e}")

    def load_audio_with_fallback(self, audio_path: str, chunk_duration: int = 30):
        """Load audio using multiple backends as fallback, with chunking for large files."""
        # Check file size first
        file_size = os.path.getsize(audio_path)
        max_file_size = 10 * 1024 * 1024  # 10MB limit
        
        if file_size > max_file_size:
            logger.info(f"Large file detected ({file_size} bytes), will process in chunks")
            return self._load_audio_in_chunks(audio_path, chunk_duration)
        
        # Method 1: Try soundfile
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_path)
            logger.info(f"Loaded audio with soundfile: {samplerate}Hz")
            return data, samplerate, "soundfile"
        except Exception as e:
            logger.warning(f"soundfile failed: {e}")
        
        # Method 2: Try librosa
        try:
            import librosa
            data, samplerate = librosa.load(audio_path, sr=None)
            logger.info(f"Loaded audio with librosa: {samplerate}Hz")
            return data, samplerate, "librosa"
        except Exception as e:
            logger.warning(f"librosa failed: {e}")
        
        # Method 3: Try torchaudio
        try:
            import torchaudio
            waveform, samplerate = torchaudio.load(audio_path)
            data = waveform.numpy().flatten()
            logger.info(f"Loaded audio with torchaudio: {samplerate}Hz")
            return data, samplerate, "torchaudio"
        except Exception as e:
            logger.warning(f"torchaudio failed: {e}")
        
        # Method 4: Try pydub
        try:
            from pydub import AudioSegment
            import numpy as np
            audio = AudioSegment.from_file(audio_path)
            data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                data = data.reshape((-1, 2)).mean(axis=1)  # Convert stereo to mono
            data = data / (2**15)  # Normalize
            samplerate = audio.frame_rate
            logger.info(f"Loaded audio with pydub: {samplerate}Hz")
            return data, samplerate, "pydub"
        except Exception as e:
            logger.warning(f"pydub failed: {e}")
        
        raise Exception("All audio loading methods failed")

    def _load_audio_in_chunks(self, audio_path: str, chunk_duration: int = 30):
        """Load large audio files in chunks to prevent memory issues."""
        try:
            # Try using pydub for chunking
            from pydub import AudioSegment
            import math
            
            # Load the audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Calculate number of chunks
            chunk_length_ms = chunk_duration * 1000  # Convert to milliseconds
            num_chunks = math.ceil(len(audio) / chunk_length_ms)
            
            logger.info(f"Splitting audio into {num_chunks} chunks of {chunk_duration}s each")
            
            chunks = []
            for i in range(num_chunks):
                start_time = i * chunk_length_ms
                end_time = min((i + 1) * chunk_length_ms, len(audio))
                
                # Extract chunk
                chunk = audio[start_time:end_time]
                
                # Convert to numpy array
                import numpy as np
                data = np.array(chunk.get_array_of_samples(), dtype=np.float32)
                if chunk.channels == 2:
                    data = data.reshape((-1, 2)).mean(axis=1)  # Convert stereo to mono
                data = data / (2**15)  # Normalize
                samplerate = chunk.frame_rate
                
                chunks.append((data, samplerate, "pydub_chunk"))
                logger.debug(f"Processed chunk {i+1}/{num_chunks}")
            
            return chunks, audio.frame_rate, "chunked_pydub"
            
        except Exception as e:
            logger.error(f"Failed to load audio in chunks: {e}")
            raise Exception(f"Failed to process large audio file: {e}")

    def _process_audio_chunk(self, data, samplerate):
        """Process a single audio chunk with Wav2Vec2."""
        import torch
        import numpy as np
        from scipy import signal
        
        # Ensure we have enough audio data
        min_samples = 1600  # 0.1 seconds at 16kHz
        if len(data) < min_samples:
            logger.warning(f"Audio chunk too short: {len(data)} samples, padding to {min_samples}")
            data = np.pad(data, (0, min_samples - len(data)), mode='constant')
        
        # Resample to 16kHz if needed
        if samplerate != 16000:
            logger.info(f"Resampling chunk from {samplerate}Hz to 16000Hz")
            num_samples = int(len(data) * 16000 / samplerate)
            data = signal.resample(data, num_samples)
            samplerate = 16000
        
        # Ensure data is float32 and normalized
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        
        # Process with Wav2Vec2
        inputs = self.processor(data, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription

    def transcribe_with_wav2vec2(self, audio_path: str, timeout: int = 120) -> str:
        """Transcribe using Wav2Vec2 model with timeout handling."""
        try:
            import torch
            import numpy as np
            from scipy import signal
            
            logger.info(f"Loading audio for Wav2Vec2: {audio_path}")
            
            # Load audio using fallback methods
            data, samplerate, method = self.load_audio_with_fallback(audio_path)
            logger.info(f"Audio loaded with {method}: {samplerate}Hz, {len(data)} samples")
            
            # For large files, process in chunks
            if isinstance(data, list) and all(isinstance(chunk, tuple) for chunk in data):
                # This is chunked data, process each chunk
                transcriptions = []
                start_time = time.time()
                
                for i, (chunk_data, chunk_samplerate, chunk_method) in enumerate(data):
                    if time.time() - start_time > timeout:
                        logger.warning(f"Wav2Vec2 processing timeout reached")
                        break
                        
                    chunk_transcription = self._process_audio_chunk(chunk_data, chunk_samplerate)
                    transcriptions.append(chunk_transcription)
                    logger.info(f"Processed chunk {i+1}/{len(data)}")
                
                return " ".join(transcriptions)
            
            # Process single file
            transcription = self._process_audio_chunk(data, samplerate)
            logger.info(f"Wav2Vec2 transcription successful: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Wav2Vec2 traceback: {traceback.format_exc()}")
            raise

    def transcribe_with_speech_recognition(self, audio_path: str, timeout: int = 120) -> str:
        """Transcribe using SpeechRecognition library with timeout handling."""
        logger.info(f"Using SpeechRecognition for: {audio_path}")
        start_time = time.time()
        try:
            # Convert to WAV if needed using pydub
            temp_wav_path = audio_path
            if not audio_path.lower().endswith('.wav'):
                try:
                    from pydub import AudioSegment
                    import tempfile
                    audio = AudioSegment.from_file(audio_path)
                    temp_wav_path = tempfile.mktemp(suffix='.wav')
                    audio.export(temp_wav_path, format="wav")
                    logger.info(f"Converted audio to WAV: {temp_wav_path}")
                except Exception as e:
                    logger.error(f"Failed to convert audio format: {e}")
                    raise Exception(f"Audio format conversion failed: {e}")
            
            logger.info(f"Processing WAV file: {temp_wav_path}")
            
            try:
                with sr.AudioFile(temp_wav_path) as source:
                    logger.info("Recording audio data...")
                    # Check timeout
                    if time.time() - start_time > timeout:
                        raise Exception("Processing timeout exceeded")
                    
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self.recognizer.record(source)
                    logger.info(f"Audio data recorded, duration: {len(audio_data.frame_data)} bytes")
            except Exception as e:
                logger.error(f"Failed to read audio file: {e}")
                raise Exception(f"Audio file reading failed: {e}")
                
            # Check timeout before recognition
            if time.time() - start_time > timeout:
                raise Exception("Processing timeout exceeded")
                
            # Try Google Speech Recognition
            try:
                logger.info("Attempting Google Speech Recognition...")
                transcription = self.recognizer.recognize_google(audio_data)
                logger.info(f"Google Speech Recognition successful: {transcription}")
                return transcription
            except sr.RequestError as e:
                logger.error(f"Google Speech Recognition request error: {e}")
                raise Exception(f"Google Speech Recognition service error: {e}")
            except sr.UnknownValueError as e:
                logger.error(f"Google Speech Recognition could not understand audio: {e}")
                raise Exception("Could not understand audio - no speech detected or audio quality too poor")
            except Exception as e:
                logger.error(f"Unexpected Google Speech Recognition error: {e}")
                raise Exception(f"Speech recognition error: {e}")
        
        except Exception as e:
            logger.error(f"Speech recognition failed with error: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        finally:
            # Clean up temp file if created
            if temp_wav_path != audio_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                    logger.info(f"Cleaned up temp file: {temp_wav_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

    def process_audio(self, audio_path: str, task_id: str) -> Dict[str, Any]:
        """Transcribe audio using available methods with timeout handling."""
        logger.info(f"Processing audio file: {audio_path}")
        start_time = time.time()
        timeout = 120  # 2 minutes timeout
        
        try:
            if not os.path.exists(audio_path):
                error_msg = f"Audio file not found: {audio_path}"
                logger.error(error_msg)
                return {"error": error_msg, "status": 500, "keywords": []}
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            logger.info(f"Audio file size: {file_size} bytes")
            
            if file_size == 0:
                error_msg = "Audio file is empty"
                logger.error(error_msg)
                return {"error": error_msg, "status": 500, "keywords": []}
            
            # For large files, set longer timeout
            if file_size > 10 * 1024 * 1024:  # > 10MB
                timeout = 300  # 5 minutes for large files
                logger.info(f"Large file detected, extending timeout to {timeout} seconds")
            
            transcription = None
            method_used = None
            
            # Check if we've already exceeded timeout
            if time.time() - start_time > timeout:
                raise Exception("Processing timeout exceeded before transcription")
            
            # Try Wav2Vec2 first if available (offline method)
            if self.wav2vec2_available:
                try:
                    logger.info("Attempting Wav2Vec2 transcription...")
                    transcription = self.transcribe_with_wav2vec2(audio_path, timeout - (time.time() - start_time))
                    method_used = "wav2vec2"
                    logger.info(f"Wav2Vec2 transcription successful: {transcription[:100]}...")
                except Exception as e:
                    logger.warning(f"Wav2Vec2 failed: {e}")
            
            # Check if we've exceeded timeout
            if time.time() - start_time > timeout:
                raise Exception("Processing timeout exceeded during transcription")
            
            # Fallback to SpeechRecognition if Wav2Vec2 failed
            if transcription is None:
                try:
                    logger.info("Starting transcription with SpeechRecognition...")
                    transcription = self.transcribe_with_speech_recognition(audio_path, timeout - (time.time() - start_time))
                    method_used = "speech_recognition"
                    logger.info(f"SpeechRecognition successful: {transcription[:100]}...")
                except Exception as e:
                    error_msg = f"Speech recognition failed: {str(e)}"
                    logger.error(error_msg)
                    return {"error": f"Audio processing failed: {str(e)}", "status": 500, "keywords": []}
            
            if not transcription or transcription.strip() == "":
                error_msg = "No speech detected in audio"
                logger.warning(error_msg)
                return {"error": error_msg, "status": 500, "keywords": []}
            
            processing_time = time.time() - start_time
            logger.info(f"Audio transcribed successfully using {method_used} in {processing_time:.2f}s: {transcription[:50]}...")
            return {
                "result": transcription,
                "method": method_used,
                "model": "audio_agent",
                "tokens_used": len(transcription.split()),
                "cost_estimate": 0.0,
                "status": 200,
                "keywords": ["audio", "transcription", method_used],
                "processing_time": processing_time
            }
            
        except Exception as e:
            error_msg = f"Unexpected error in process_audio: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": error_msg, "status": 500, "keywords": []}

    def run(self, input_path: str, live_feed: str = "", model: str = "edumentor_agent", input_type: str = "audio", task_id: str = None) -> Dict[str, Any]:
        task_id = task_id or str(uuid.uuid4())
        logger.info(f"AudioAgent processing task {task_id}, input: {input_path}")
        result = self.process_audio(input_path, task_id)
        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, input_path, result, "audio_agent", model, reward)
        return result

if __name__ == "__main__":
    agent = AudioAgent()
    test_input = "test_audio.wav"
    result = agent.run(test_input, input_type="audio")
    print(result)
