import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import difflib
import queue
import sys
import re
import numpy as np
import sounddevice as sd
import whisper
import pyautogui

# --- THE CHECKPOINT CONFIGURATION ---
# Triggers are now SHORTENED to only the final 3-4 words of the section.
# This prevents the fuzzy matcher from passing a sentence before it is fully finished.
# --- THE CHECKPOINT CONFIGURATION ---
SLIDE_TRIGGERS = [
    # --- INTRODUCTORY PRAYERS ---
    {
        "slide_name": "Prologue",
        "leader_trigger": "one true god",  
        "people_trigger": "for ever amen", 
        "reps": 1,
        "context": "In the name of the Father and of the Son and of the Holy Spirit, one true God. Glory be to Him; and may His grace and mercy be upon us for ever. Amen."
    },
    {
        "slide_name": "Hosanna",
        "leader_trigger": "in the highest",
        "people_trigger": "in the highest",
        "reps": 1,
        "context": "Holy, Holy, Holy, Lord God Almighty, by whose glory, the heaven and the earth are filled; Hosanna in the highest. Blessed is He who has come, and is to come in the name of the Lord; glory be to Him in the highest."
    },
    {
        "slide_name": "Trisagion (3x Repetition)",
        "leader_trigger": "thou o god",
        "people_trigger": "mercy on us",
        "reps": 3,  
        "context": "Holy art Thou, O God! Holy art Thou, Almighty, Holy art Thou, Immortal, Crucified for us, have mercy on us."
    },
    {
        "slide_name": "Lord have mercy",
        "leader_trigger": "mercy upon us",
        "people_trigger": "mercy on us",
        "reps": 1,
        "context": "Lord, have mercy upon us, Lord, be kind and have mercy; Lord, accept our prayers and worship and have mercy on us."
    },
    {
        "slide_name": "Glory be to Thee",
        "leader_trigger": "thee o god",
        "people_trigger": "thy servants",
        "reps": 1,
        "context": "Glory be to Thee, O God! Glory be to Thee, O creator; Glory be to Thee, O Christ, the King who does pity sinners, Thy servants. Barekhmor."
    },
    {
        "slide_name": "The Lords Prayer",
        "leader_trigger": "as it is in heaven",
        "people_trigger": "ever and ever amen",
        "reps": 1,
        "context": "Our Father who Art in Heaven, Hallowed be Thy name. Thy Kingdom come; Thy will be done on earth, as it is in heaven. Give us this day our daily bread: and forgive us our trespasses as we forgive those who trespass against us. Lead us not into temptation, but deliver us from the evil one; for Thine is the Kingdom, the Power and the glory for ever and ever. Amen."
    },
    {
        "slide_name": "Hail Mary",
        "leader_trigger": "fruit of thy womb",
        "people_trigger": "our death amen",
        "reps": 1,
        "context": "Peace to you, Mary, full of Grace. Our Lord is with Thee. Blessed art Thou among women, and blessed is the fruit of Thy womb, our Lord, Jesus Christ. O Virgin Saint Mary, O Mother of God, pray for us sinners, now and at all times, and at the hour of our death. Amen."
    },

    # --- 9TH HOUR ---
    {
        "slide_name": "9th Hour Qolo",
        "leader_trigger": "life to the dead",
        "people_trigger": "your precious blood", # Stopped before Syriac phrase for reliability
        "reps": 1,
        "context": "Praise be to You O God, who gives life to the dead. Praise be to You O God, who grants resurrection to the entombed. We praise You and glorify Your Father who did send you and the Holy Spirit. Barekhmor. O Lord, one of Trinity, who by Your own will stayed in the tomb for three days, give resurrection to our departed ones, for they were saved by Your precious blood. Moryo rahem alay noo adarayn."
    },
    {
        "slide_name": "Song - Prince of Life",
        "leader_trigger": "throne above",
        "people_trigger": "life to all",
        "reps": 1,
        "context": "Comes the Prince of life from His glorious throne above, Raising those who in their graves take rest. From their graves they'll rise With them our departed ones, praising Him who giveth life to all. Barekmor."
    },
    {
        "slide_name": "Celebrant Shub'ho",
        "leader_trigger": "holy spirit",
        "people_trigger": "holy spirit",
        "reps": 1,
        "context": "Glory be to the Father, Son, and Holy Spirit"
    },
    {
        "slide_name": "Song - Praised be Jesus Words",
        "leader_trigger": "drink my blood",
        "people_trigger": "everlasting life",
        "reps": 1,
        "context": "Praised be Jesus' words In His Gospel given to us, Those who eat my flesh and drink my blood Them I will not leave Bound in hell for them died Giving all the everlasting life. Moriyo Rahem Mela Nooah Dharain"
    },
    {
        "slide_name": "Bovutho 1",
        "leader_trigger": "day of resurrection",
        "people_trigger": "hope in you",
        "reps": 1,
        "context": "O! merciful Lord, renew Your creation on the day of resurrection. O! Lord, grant rest and comfort to our beloved departed ones who have lived and died with hope in You."
    },
    {
        "slide_name": "Bovutho 2",
        "leader_trigger": "isaac and jacob",
        "people_trigger": "departed amen",
        "reps": 1,
        "context": "O! Lord grant rest to our faithful departed in the bosom of Abraham, Isaac and Jacob. May the souls and bodies together cry aloud and say: glory be to the one who has come and is to come to resurrect the departed. Amen."
    },
    {
        "slide_name": "Song - O Lord full of compassion",
        "leader_trigger": "await your coming",
        "people_trigger": "is to come",
        "reps": 1,
        "context": "O Lord, full of compassion, renew Your Creation on the day of resurrection. Grant rest and pardon to those departed Slept in Your hope and await Your coming. May Your servants rest in the bosom of Abraham, Isaac and Jacob, O Lord! May the bodies and souls cry together Blessed is He Who has come and is to come."
    },
    
    # --- REPEATING SET ---
    {
        "slide_name": "Trisagion 2",
        "leader_trigger": "thou o god",
        "people_trigger": "mercy on us",
        "reps": 3,  
        "context": "Holy art Thou, O God! Holy art Thou, Almighty, Holy art Thou, Immortal, Crucified for us, have mercy on us."
    },
    {
        "slide_name": "Lord have mercy 2",
        "leader_trigger": "mercy upon us",
        "people_trigger": "mercy on us",
        "reps": 1,
        "context": "Lord, have mercy upon us, Lord, be kind and have mercy; Lord, accept our prayers and worship and have mercy on us."
    },
    {
        "slide_name": "Glory be to Thee 2",
        "leader_trigger": "thee o god",
        "people_trigger": "thy servants",
        "reps": 1,
        "context": "Glory be to Thee, O God! Glory be to Thee, O creator; Glory be to Thee, O Christ, the King who does pity sinners, Thy servants. Barekhmor."
    },
    {
        "slide_name": "The Lords Prayer 2",
        "leader_trigger": "as it is in heaven",
        "people_trigger": "ever and ever amen",
        "reps": 1,
        "context": "Our Father who Art in Heaven, Hallowed be Thy name. Thy Kingdom come; Thy will be done on earth, as it is in heaven. Give us this day our daily bread: and forgive us our trespasses as we forgive those who trespass against us. Lead us not into temptation, but deliver us from the evil one; for Thine is the Kingdom, the Power and the glory for ever and ever. Amen."
    },
    {
        "slide_name": "Hail Mary 2",
        "leader_trigger": "fruit of thy womb",
        "people_trigger": "our death amen",
        "reps": 1,
        "context": "Peace to you, Mary, full of Grace. Our Lord is with Thee. Blessed art Thou among women, and blessed is the fruit of Thy womb, our Lord, Jesus Christ. O Virgin Saint Mary, O Mother of God, pray for us sinners, now and at all times, and at the hour of our death. Amen."
    },

    # --- EVENING PRAYERS ---
    {
        "slide_name": "Trisagion 3",
        "leader_trigger": "thou o god",
        "people_trigger": "mercy on us",
        "reps": 3,  
        "context": "Holy art Thou, O God! Holy art Thou, Almighty, Holy art Thou, Immortal, Crucified for us, have mercy on us."
    },
    {
        "slide_name": "Lord have mercy 3",
        "leader_trigger": "mercy upon us",
        "people_trigger": "mercy on us",
        "reps": 1,
        "context": "Lord, have mercy upon us, Lord, be kind and have mercy; Lord, accept our prayers and worship and have mercy on us."
    },
    {
        "slide_name": "Glory be to Thee 3",
        "leader_trigger": "thee o god",
        "people_trigger": "thy servants",
        "reps": 1,
        "context": "Glory be to Thee, O God! Glory be to Thee, O creator; Glory be to Thee, O Christ, the King who does pity sinners, Thy servants. Barekhmor."
    },
    {
        "slide_name": "The Lords Prayer 3",
        "leader_trigger": "as it is in heaven",
        "people_trigger": "ever and ever amen",
        "reps": 1,
        "context": "Our Father who Art in Heaven, Hallowed be Thy name. Thy Kingdom come; Thy will be done on earth, as it is in heaven. Give us this day our daily bread: and forgive us our trespasses as we forgive those who trespass against us. Lead us not into temptation, but deliver us from the evil one; for Thine is the Kingdom, the Power and the glory for ever and ever. Amen."
    },
    {
        "slide_name": "Hail Mary 3",
        "leader_trigger": "fruit of thy womb",
        "people_trigger": "our death amen",
        "reps": 1,
        "context": "Peace to you, Mary, full of Grace. Our Lord is with Thee. Blessed art Thou among women, and blessed is the fruit of Thy womb, our Lord, Jesus Christ. O Virgin Saint Mary, O Mother of God, pray for us sinners, now and at all times, and at the hour of our death. Amen."
    },
    
    # --- PSALM 51 ---
    {
        "slide_name": "Psalm 51 - Part 1",
        "leader_trigger": "my transgressions",
        "people_trigger": "ever before me", # Psalm 51 alternates Leader/People/Leader, so we use the very last sentence
        "reps": 1,
        "context": "Have mercy on me, O God, according to your steadfast love; according to your abundant mercy blot out my transgressions. Wash me thoroughly from my iniquity, and cleanse me from my sin. For I know my transgressions, and my sin is ever before me."
    },
    {
        "slide_name": "Psalm 51 - Part 2",
        "leader_trigger": "mother conceived me",
        "people_trigger": "whiter than snow",
        "reps": 1,
        "context": "Against you, you alone, have I sinned, and done what is evil in your sight, so that you are justified in your sentence and blameless when you pass judgment. Indeed, I was born guilty, a sinner when my mother conceived me. You desire truth in the inward being; therefore teach me wisdom in my secret heart. Purge me with hyssop, and I shall be clean; wash me, and I shall be whiter than snow."
    },
    {
        "slide_name": "Psalm 51 - Part 3",
        "leader_trigger": "all my iniquities",
        "people_trigger": "spirit from me",
        "reps": 1,
        "context": "Let me hear joy and gladness; let the bones that you have crushed rejoice. Hide your face from my sins, and blot out all my iniquities. Create in me a clean heart, O God, and put a new and right spirit within me. Do not cast me away from your presence, and do not take your holy spirit from me."
    },
    {
        "slide_name": "Psalm 51 - Part 4",
        "leader_trigger": "return to you",
        "people_trigger": "declare your praise",
        "reps": 1,
        "context": "Restore to me the joy of your salvation, and sustain in me a willing spirit. Then I will teach transgressors your ways, and sinners will return to you. Deliver me from bloodshed, O God, O God of my salvation, and my tongue will sing aloud of your deliverance. O Lord, open my lips, and my mouth will declare your praise."
    },
    {
        "slide_name": "Psalm 51 - Part 5",
        "leader_trigger": "will not despise",
        "people_trigger": "praise o god", 
        "reps": 1,
        "context": "For you have no delight in sacrifice; if I were to give a burnt offering, you would not be pleased. The sacrifice acceptable to God is a broken spirit; a broken and contrite heart, O God, you will not despise. Do good to Zion. Rebuild the walls of Jerusalem."
    }
]

MATCH_THRESHOLD = 0.80  # Increased strictness to prevent early triggers
SAMPLE_RATE = 16000     
SILENCE_THRESHOLD = 0.01  

# --- OVERLAPPING SLIDING WINDOW ---
BLOCK_SIZE = int(16000 * 6.0)  # Expanded to a 6-second view to prevent sentence chopping
SHIFT_SIZE = int(16000 * 2.0)  # Slides forward by 2 seconds at a time for fast reaction

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def count_elastic_fuzzy_matches(spoken_text, trigger_phrase, threshold=MATCH_THRESHOLD):
    """
    Counts multiple occurrences in a single sentence and 
    uses an elastic window to handle added or omitted words.
    """
    spoken_text = re.sub(r'[^\w\s]', '', spoken_text.lower())
    trigger_phrase = re.sub(r'[^\w\s]', '', trigger_phrase.lower())
    
    words = spoken_text.split()
    trigger_words = trigger_phrase.split()
    trigger_len = len(trigger_words)
    
    if trigger_len == 0 or len(words) == 0:
        return 0
        
    matches = 0
    skip_until = -1
    
    for i in range(len(words)):
        if i < skip_until:
            continue
            
        best_sim = 0
        for window_offset in [-1, 0, 1]:
            window_size = trigger_len + window_offset
            if window_size <= 0 or i + window_size > len(words):
                continue
                
            phrase_chunk = " ".join(words[i:i + window_size])
            similarity = difflib.SequenceMatcher(None, phrase_chunk, trigger_phrase).ratio()
            
            if similarity > best_sim:
                best_sim = similarity
                
        if best_sim >= threshold:
            matches += 1
            skip_until = i + trigger_len - 1 
            
    return matches

def main():
    print("Loading accurate AI Voice Engine (Whisper Base)...")
    model = whisper.load_model("base") #[cite: 4]
    
    current_slide_index = 0
    total_slides = len(SLIDE_TRIGGERS) #[cite: 4]
    
    leader_cleared = False
    match_counter = 0  
    
    base_vocabulary = "Amen, Barekhmor, Trisagion, Kuriyelaison, Moryo rahem melayn no adharan, Jacobite, Orthodox, Hosanna, shubho labo labaro val roho qadeesho" #[cite: 4]

    print("\n=== Checkpoint-Steered Slide Automation Started ===") #[cite: 4]
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback): #[cite: 4]
        buffer = np.zeros((0, 1), dtype='float32') #[cite: 4]
        
        while current_slide_index < total_slides:
            slide_data = SLIDE_TRIGGERS[current_slide_index] #[cite: 4]
            leader_target = slide_data["leader_trigger"] #[cite: 4]
            people_target = slide_data["people_trigger"] #[cite: 4]
            required_reps = slide_data["reps"] #[cite: 4]
            dynamic_prompt = f"{base_vocabulary}, {slide_data['context']}" #[cite: 4]
            
            if not leader_cleared and match_counter == 0 and len(buffer) == 0: #[cite: 4]
                print(f"\n[Slide {current_slide_index + 1}: {slide_data['slide_name']}]") #[cite: 4]
                print(f"🔒 Waiting for LEADER: '{leader_target}'") #[cite: 4]
            
            while len(buffer) < BLOCK_SIZE:
                try:
                    data = audio_queue.get(timeout=1.0) #[cite: 4]
                    buffer = np.vstack((buffer, data)) #[cite: 4]
                except queue.Empty: #[cite: 4]
                    continue
            
            eval_block = buffer[:BLOCK_SIZE].flatten() #[cite: 4]
            # Slide the buffer down by the SHIFT_SIZE instead of chopping it in half
            buffer = buffer[SHIFT_SIZE:] 
            
            volume_level = np.sqrt(np.mean(eval_block**2)) #[cite: 4]
            if volume_level < SILENCE_THRESHOLD: #[cite: 4]
                continue
                
            result = model.transcribe(eval_block, fp16=False, initial_prompt=dynamic_prompt) #[cite: 4]
            spoken_text = str(result.get("text", "")).strip() #[cite: 4]
            
            if spoken_text:
                print(f"  > Heard: {spoken_text}") #[cite: 4]
                
                if not leader_cleared:
                    if count_elastic_fuzzy_matches(spoken_text, leader_target) > 0:
                        leader_cleared = True
                        print(f"🔓 LEADER CLEARED! Now listening for PEOPLE: '{people_target}' ({required_reps}x)") #[cite: 4]
                        buffer = np.zeros((0, 1), dtype='float32') 
                else:
                    new_matches = count_elastic_fuzzy_matches(spoken_text, people_target) #[cite: 4]
                    
                    if new_matches > 0:
                        match_counter += new_matches #[cite: 4]
                        print(f"   -> Match count increased! ({match_counter}/{required_reps})") #[cite: 4]
                        
                        if match_counter >= required_reps: #[cite: 4]
                            print("✅ Required repetitions met. Advancing slide...") #[cite: 4]
                            pyautogui.press('right') #[cite: 4]
                            
                            current_slide_index += 1 #[cite: 4]
                            leader_cleared = False
                            match_counter = 0  
                            buffer = np.zeros((0, 1), dtype='float32') #[cite: 4]
                            
                            if current_slide_index >= total_slides:
                                print("\n=== Service Automation Finished ===") #[cite: 4]

if __name__ == "__main__":
    main() #[cite: 4]
