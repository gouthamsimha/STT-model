import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import time
from pynput import keyboard
import threading
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
from transformers import pipeline
import warnings
import webrtcvad
import struct

# Suppress the specific FutureWarning about 'inputs' deprecation
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")

class WhisperTranscriber:
    def __init__(self):
        """Initialize Distil-Whisper model"""
        print("Loading Distil-Whisper model (medium.en)... Please wait...")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "distil-whisper/distil-medium.en",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(device)
        except ImportError:
            # Fallback if accelerate isn't available
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "distil-whisper/distil-medium.en",
                torch_dtype=torch_dtype,
                use_safetensors=True
            ).to(device)
        
        self.processor = AutoProcessor.from_pretrained("distil-whisper/distil-medium.en")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={
                "max_new_tokens": 256,
                "temperature": 0.76,
                "do_sample": True,
                "top_k": 45,
                "top_p": 0.85,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 2,
                "num_beams": 2
            }
        )
        
        self.output_dir = "transcriptions"
        self.audio_dir = "recordings"
        self.is_recording = False
        self.audio_chunks = []
        self.should_exit = False
        
        # Simplified to just Command key (⌘)
        self.TRIGGER_KEY = keyboard.Key.cmd  # Command key (⌘)
        self.current_keys = set()
        
        # Create both directories
        for directory in [self.output_dir, self.audio_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Add VAD initialization
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (0-3)
        self.silence_threshold = 10.0  # Changed from 0.5 to 10 seconds
        self.silence_counter = 0
        self.frame_duration = 30  # Duration of each frame in milliseconds
    
    def on_press(self, key):
        """Handle key press events"""
        if key == self.TRIGGER_KEY:
            self.toggle_recording()

    def on_release(self, key):
        """Handle key release events"""
        if key == keyboard.Key.esc:  # ESC to exit
            print("\n👋 Exiting program...")
            self.should_exit = True
            if self.is_recording:
                self.stop_recording()
            return False  # Stop listener

    def transcribe_with_dual_output(self, audio_path):
        """Generate transcription with optimized parameters"""
        try:
            audio_input, sample_rate = sf.read(audio_path)
            
            # Calculate approximate duration in seconds
            duration = len(audio_input) / sample_rate
            # Adjust max_new_tokens based on audio duration (roughly 3 tokens per second of audio)
            max_tokens = max(256, int(duration * 3))
            
            result = self.pipe(
                {
                    "raw": audio_input,
                    "sampling_rate": sample_rate
                },
                generate_kwargs={
                    "max_new_tokens": max_tokens,
                    "temperature": 0.75,
                    "do_sample": True,
                    "top_k": 50,
                    "top_p": 0.92,
                    "repetition_penalty": 1.3,
                    "no_repeat_ngram_size": 2,
                    "num_beams": 3,
                    "return_timestamps": True
                }
            )
            
            # Extract just the text if timestamps are returned
            return result["text"] if isinstance(result, dict) else result
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return "Error during transcription"

    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.audio_chunks = []
        print("\n🎤 Recording started... (Press Command (⌘) again to stop)")
        
        # Start recording in a separate thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()

    def stop_recording(self):
        """Stop recording and process audio"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        # Wait a moment for the recording thread to finish
        time.sleep(0.1)
        print("\n✅ Recording stopped!")
        
        if not self.should_exit:
            try:
                if not self.audio_chunks:
                    print("\n⚠️ No audio recorded - recording may have been too short")
                    print(f"Number of audio chunks: {len(self.audio_chunks)}")
                    return
                    
                audio_data = np.concatenate(self.audio_chunks)
                print(f"Audio length: {len(audio_data)} samples")
                
                if len(audio_data) < 1600:
                    print("\n⚠️ Recording too short - please record for longer")
                    return
                
                # Generate unique filename for this recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_filename = f"{self.audio_dir}/recording_{timestamp}.wav"
                
                # Save audio with unique filename
                self.save_audio(audio_data, filename=audio_filename)
                
                print("\n🔄 Processing recording...")
                processing_start = time.time()
                
                transcription = self.transcribe_with_dual_output(audio_filename)
                
                processing_time = time.time() - processing_start
                print(f"\n⏱️ Processing time: {processing_time:.2f} seconds")
                
                # Display result
                print("\n📝 Transcription:")
                print(transcription)
                
                # Save transcription
                self.save_transcription(transcription)
                    
            except Exception as e:
                print(f"\n❌ Error during processing: {str(e)}")

    def _record_audio(self):
        """Background recording function with VAD"""
        sample_rate = 16000
        frame_length = int(sample_rate * (self.frame_duration / 1000.0))
        stream = None
        
        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=frame_length
            )
            stream.start()
            
            consecutive_silence = 0
            
            while self.is_recording:
                audio_chunk, _ = stream.read(frame_length)
                
                # Convert float32 to int16 for VAD
                audio_int16 = (audio_chunk.flatten() * 32768).astype(np.int16)
                frame_bytes = struct.pack("%dh" % len(audio_int16), *audio_int16)
                
                # Check if frame contains speech
                is_speech = self.vad.is_speech(frame_bytes, sample_rate)
                
                if is_speech:
                    consecutive_silence = 0
                    self.audio_chunks.append(audio_chunk)
                else:
                    consecutive_silence += self.frame_duration / 1000.0
                    
                    # Still append audio during short silences
                    self.audio_chunks.append(audio_chunk)
                    
                    # Stop recording after silence_threshold seconds of silence
                    if consecutive_silence >= self.silence_threshold:
                        print("\n🤫 Silence detected, stopping recording...")
                        self.is_recording = False
                        break
                
        except Exception as e:
            print(f"Recording error: {str(e)}")
        finally:
            if stream is not None:
                stream.stop()
                stream.close()

    def save_audio(self, audio_data, filename=None, sample_rate=16000):
        """Save recorded audio"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.audio_dir}/recording_{timestamp}.wav"
        sf.write(filename, audio_data, sample_rate)
        return filename
    
    def save_transcription(self, text):
        """Save transcription to a text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/transcription_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        return filename

def main():
    transcriber = WhisperTranscriber()
    print(f"\n🎯 Whisper Speech-to-Text Ready!")
    print(f"Press Command (⌘) to start/stop recording")
    print("Press ESC to exit")
    
    try:
        with keyboard.Listener(
            on_press=transcriber.on_press,
            on_release=transcriber.on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\n Program terminated by user")
    finally:
        if transcriber.is_recording:
            transcriber.stop_recording()
        print("\n✨ Program ended successfully")
        os._exit(0)

if __name__ == "__main__":
    main()