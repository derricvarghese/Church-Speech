import time
import collections
import re
import sys
import argparse
import difflib
from dataclasses import dataclass
from typing import List

import yaml
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from Quartz import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap

# --- MAC KEYSTROKE SIMULATOR ---
MAC_KEYCODES = {"right": 124, "left": 123, "space": 49, "return": 36}

def press_key(name: str) -> None:
    keycode = MAC_KEYCODES.get(name, 124)
    down = CGEventCreateKeyboardEvent(None, keycode, True)
    up = CGEventCreateKeyboardEvent(None, keycode, False)
    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)

# --- CONFIGURATION CLASSES ---
@dataclass
class Rule:
    text: str
    threshold: float

@dataclass
class Slide:
    id: int
    name: str
    context: str
    leader: Rule
    people: Rule
    sequence_repeats: int

class Config:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        g = raw["global"]
        self.device_name = g["device_name"]
        self.sample_rate = int(g["sample_rate"])
        self.whisper_model = g["whisper_model"]
        self.advance_key = g["advance_key"]
        self.vad_aggressiveness = int(g["vad_aggressiveness"])
        self.window_sec = float(g["window_sec"])
        self.shift_sec = float(g["shift_sec"])
        self.hotwords = list(g.get("hotwords", []))
        self.filler_words = list(g.get("filler_words", []))
        
        self.slides = []
        for s in raw["slides"]:
            leader = Rule(s["leader"]["text"], float(s["leader"]["threshold"]))
            people = Rule(s["people"]["text"], float(s["people"]["threshold"]))
            seq_reps = int(s.get("sequence_repeats", 1))
            self.slides.append(Slide(s["id"], s["name"], s.get("context", ""), leader, people, seq_reps))

# --- LOGIC & MATH ---
def clean_text(text: str, filler_words: List[str]) -> str:
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = [w for w in text.split() if w not in filler_words]
    return " ".join(words)

def elastic_fuzzy_match(spoken_text: str, trigger_phrase: str, threshold: float) -> bool:
    """Checks for a mathematical match using an elastic window size."""
    words = spoken_text.split()
    trigger_len = len(trigger_phrase.split())
    
    if trigger_len == 0 or len(words) == 0:
        return False
        
    for i in range(len(words)):
        for offset in [-1, 0, 1]:
            window_size = trigger_len + offset
            if window_size <= 0 or i + window_size > len(words):
                continue
            chunk = " ".join(words[i:i + window_size])
            if difflib.SequenceMatcher(None, chunk, trigger_phrase).ratio() >= threshold:
                return True
    return False

def contains_speech(audio_data: np.ndarray, sample_rate: int, vad: webrtcvad.Vad) -> bool:
    """Uses WebRTC to check if a block actually contains human speech."""
    frame_size = int(sample_rate * 0.03) # 30ms frames
    speech_frames = 0
    total_frames = 0
    
    # Convert numpy array to raw bytes for WebRTC
    raw_bytes = audio_data.tobytes()
    frame_bytes = frame_size * 2
    
    for i in range(0, len(raw_bytes) - frame_bytes, frame_bytes):
        frame = raw_bytes[i:i + frame_bytes]
        total_frames += 1
        if vad.is_speech(frame, sample_rate):
            speech_frames += 1
            
    return (speech_frames / max(1, total_frames)) > 0.05

# --- MAIN ENGINE ---
class SilentAcolyte:
    def __init__(self, config_path: str):
        self.cfg = Config(config_path)
        
        # The Live-View Circular Buffer (Prevents Lag entirely)
        self.max_samples = int(self.cfg.sample_rate * self.cfg.window_sec)
        self.audio_buffer = collections.deque(maxlen=self.max_samples)
        
        self.vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)
        
        print(f"Loading Lightning Fast AI Engine ({self.cfg.whisper_model})...")
        self.model = WhisperModel(self.cfg.whisper_model, device="cpu", compute_type="int8")

    def audio_callback(self, indata, frames, time_info, status):
        # Constantly pipe microphone audio directly into the circular buffer
        self.audio_buffer.extend(indata[:, 0])

    def get_device_index(self):
        for i, d in enumerate(sd.query_devices()):
            if self.cfg.device_name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                return i
        raise ValueError(f"Could not find input device: {self.cfg.device_name}")

    def run(self):
        device_idx = self.get_device_index()
        current_slide_idx = 0
        total_slides = len(self.cfg.slides)
        
        # State tracking
        leader_cleared = False
        current_sequence = 0

        print("\n=== Silent Acolyte Automation Started ===")
        
        with sd.InputStream(samplerate=self.cfg.sample_rate, channels=1, dtype='int16', 
                            device=device_idx, callback=self.audio_callback):
            
            while current_slide_idx < total_slides:
                slide = self.cfg.slides[current_slide_idx]
                leader_target = clean_text(slide.leader.text, self.cfg.filler_words)
                people_target = clean_text(slide.people.text, self.cfg.filler_words)
                prompt = f"{', '.join(self.cfg.hotwords)}, {slide.context}"

                if not leader_cleared and current_sequence == 0 and len(self.audio_buffer) == 0:
                    print(f"\n[Slide {slide.id}: {slide.name}]")
                    print(f"🔒 Waiting for LEADER: '{leader_target}'")

                # If we don't have enough audio yet, wait a moment
                if len(self.audio_buffer) < self.max_samples // 2:
                    time.sleep(0.1)
                    continue

                # Take a snapshot of the live audio buffer
                live_audio = np.array(self.audio_buffer, dtype=np.int16)

                # 1. Check for silence to save CPU
                if not contains_speech(live_audio, self.cfg.sample_rate, self.vad):
                    time.sleep(self.cfg.shift_sec)
                    continue

                # 2. Transcribe
                audio_float32 = live_audio.astype(np.float32) / 32768.0
                segments, _ = self.model.transcribe(audio_float32, initial_prompt=prompt, condition_on_previous_text=False)
                spoken_text = " ".join([seg.text for seg in segments])
                spoken_text = clean_text(spoken_text, self.cfg.filler_words)

                if spoken_text:
                    print(f"  > Heard: {spoken_text}")

                    # 3. Match Logic
                    if not leader_cleared:
                        if elastic_fuzzy_match(spoken_text, leader_target, slide.leader.threshold):
                            leader_cleared = True
                            print(f"🔓 LEADER CLEARED! Listening for PEOPLE: '{people_target}'")
                            
                            # Flush the buffer so the people's check doesn't accidentally trigger on old audio
                            self.audio_buffer.clear() 
                    else:
                        if elastic_fuzzy_match(spoken_text, people_target, slide.people.threshold):
                            current_sequence += 1
                            
                            if current_sequence >= slide.sequence_repeats:
                                print(f"✅ Full Sequence completed ({slide.sequence_repeats}x). Advancing slide...")
                                press_key(self.cfg.advance_key)
                                
                                current_slide_idx += 1
                                leader_cleared = False
                                current_sequence = 0
                                self.audio_buffer.clear()
                                
                                if current_slide_idx >= total_slides:
                                    print("\n=== Service Automation Finished ===")
                            else:
                                print(f"   -> [Sequence {current_sequence}/{slide.sequence_repeats} done] Waiting for LEADER again...")
                                leader_cleared = False
                                self.audio_buffer.clear()

                # Pace the loop so we aren't burning up your Mac's CPU
                time.sleep(self.cfg.shift_sec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    app = SilentAcolyte(args.config)
    app.run()
