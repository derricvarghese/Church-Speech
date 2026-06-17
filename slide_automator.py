import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import difflib
import queue
import sys
import numpy as np
import sounddevice as sd
import whisper
import pyautogui

# --- CONFIGURATION ---
SLIDE_TRIGGERS = [
    (
        "mercy be upon us for ever amen", 
        1, 
        "In the name of the Father and of the Son and of the Holy Spirit, one true God. Glory be to Him; and may His grace and mercy be upon us for ever. Amen."
    ),         
    (
        "glory be to him in the highest", 
        1, 
        "Holy, Holy, Holy, Lord God Almighty, by whose glory, the heaven and the earth are filled; Hosanna in the highest. Blessed is He who has come, and is to come in the name of the Lord; glory be to Him in the highest."
    ),  
    (
        "crucified for us have mercy on us", 
        3, 
        "Holy art Thou, O God! Holy art Thou, Almighty, Holy art Thou, Immortal, Crucified for us, have mercy on us."
    ),      
    (
        "worship and have mercy on us", 
        1, 
        "Lord, have mercy upon us, Lord, be kind and have mercy; Lord, accept our prayers and worship and have mercy on us."
    )
]

MATCH_THRESHOLD = 0.70  
SAMPLE_RATE = 16000     
SILENCE_THRESHOLD = 0.01  

# --- STREAMING PARAMETERS ---
BLOCK_SIZE = 16000 * 4  # Captures 4-second semantic evaluation blocks dynamically
# ------------------------------

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This function is called by sounddevice automatically for every new audio buffer."""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def fuzzy_match(spoken_text, trigger_phrase):
    spoken_text = spoken_text.lower()
    trigger_phrase = trigger_phrase.lower()
    
    if trigger_phrase in spoken_text:
        return True
        
    words = spoken_text.split()
    trigger_len = len(trigger_phrase.split())
    
    for i in range(len(words) - trigger_len + 1):
        phrase_chunk = " ".join(words[i:i + trigger_len])
        similarity = difflib.SequenceMatcher(None, phrase_chunk, trigger_phrase).ratio()
        if similarity >= MATCH_THRESHOLD:
            return True
    return False

def main():
    print("Loading accurate AI Voice Engine (Whisper Base)...")
    model = whisper.load_model("base")
    
    current_slide_index = 0
    total_triggers = len(SLIDE_TRIGGERS)
    match_counter = 0  
    
    base_vocabulary = "Amen, Barekhmor, Trisagion, Kuriyelaison, Moryo rahem, Jacobite, Orthodox, Hosanna"

    print("\n=== Continuous Stream Slide Automation Started ===")
    print(f"Waiting for Slide 1 trigger: '{SLIDE_TRIGGERS[current_slide_index][0]}'")

    # Open a non-blocking stream loop that records in the background constantly
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        buffer = np.zeros((0, 1), dtype='float32')
        
        while current_slide_index < total_triggers:
            target_phrase, required_counts, slide_context = SLIDE_TRIGGERS[current_slide_index]
            dynamic_prompt = f"{base_vocabulary}, {slide_context}"
            
            # Keep grabbing tiny chunks from the stream queue until we hit our target 4-second block size
            while len(buffer) < BLOCK_SIZE:
                try:
                    data = audio_queue.get(timeout=1.0)
                    buffer = np.vstack((buffer, data))
                except queue.Empty:
                    continue
            
            # Extract out the active 4-second window evaluating block
            eval_block = buffer[:BLOCK_SIZE].flatten()
            # Shift buffer down to handle sliding audio window logic without gaps
            buffer = buffer[BLOCK_SIZE // 2:] 
            
            volume_level = np.sqrt(np.mean(eval_block**2))
            if volume_level < SILENCE_THRESHOLD:
                continue
                
            result = model.transcribe(eval_block, fp16=False, initial_prompt=dynamic_prompt)
            spoken_text = str(result.get("text", "")).strip()
            
            if spoken_text:
                print(f"Heard: {spoken_text}")
                
                if fuzzy_match(spoken_text, target_phrase):
                    match_counter += 1
                    print(f"-> Match detected! ({match_counter}/{required_counts})")
                    
                    if match_counter >= required_counts:
                        print("--> Required repetitions met. Advancing slide...")
                        pyautogui.press('right')
                        
                        current_slide_index += 1
                        match_counter = 0  
                        buffer = np.zeros((0, 1), dtype='float32') # Flush pipeline for new slide context
                        
                        if current_slide_index < total_triggers:
                            next_phrase, next_count, _ = SLIDE_TRIGGERS[current_slide_index]
                            print(f"\n[Slide {current_slide_index + 1}] Waiting for: '{next_phrase}' ({next_count}x)")
                        else:
                            print("\n=== Service Automation Finished ===")

if __name__ == "__main__":
    main()