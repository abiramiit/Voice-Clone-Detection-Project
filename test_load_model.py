import os
import torch
import librosa
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import warnings

warnings.filterwarnings("ignore")

def fetch_real_audio():
    try:
        # Utilizing live streamer limiting payload execution natively against proxy blocks 
        stream = load_dataset("LanceaKing/asvspoof2019", "LA", split="test", streaming=True, trust_remote_code=True)
        for item in stream:
            audio_dict = item.get('audio')
            if not audio_dict or 'array' not in audio_dict:
                continue
            orig_sr = audio_dict['sampling_rate']
            arr = np.array(audio_dict['array'])
            return arr, orig_sr, item['label']
    except Exception as e:
        print("[CRITICAL] Hard-coded dataset API constraint intrinsically crashed huggingface zip block.")
        print(f"HF Server Fault logic: {e}")
        return None, None, None

def main():
    print("\n--- 1. INITIATING ARCHITECTURE LOAD ---")
    save_dir = "models/voice_clone_detector"
    
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} completely missing.")
        return
        
    try:
        # Load implicitly verified configurations & weights previously instanced natively
        model = Wav2Vec2ForSequenceClassification.from_pretrained(save_dir)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(save_dir)
        print("[*] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Checkpoint initialization bounds failed: {e}")
        return
        
    print("\n--- 2. ACQUIRING REAL AUDIO INFERENCE SAMPLE ---")
    audio_arr, sr, label = fetch_real_audio()
    
    if audio_arr is None:
        print("\n[HALT] Blocked natively by API fault. Absolutely No Dummy/Mock Audio matrices generated per strict rule bounds.")
        return
        
    print(f"Input Audio Acquired Successfully: True shape mapping -> {(audio_arr.shape,)} across {sr}Hz.")
    if sr != 16000:
        audio_arr = librosa.resample(y=audio_arr, orig_sr=sr, target_sr=16000)
        
    print("\n--- 3. PURE INFERENCE EXECUTION ---")
    inputs = feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(inputs.input_values)
        
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_class_id = torch.argmax(probs, dim=-1).item()
    
    # Decoding predictions empirically via dynamically saved configurations mappings
    pred_label = model.config.id2label[pred_class_id]
    
    print("\n=== FINAL VERIFIED RESULTS ===")
    print(f"Input Audio Sample Label Source: {label} Native Class")
    print(f"Predicted Class Output: {pred_label}")
    print(f"Prediction Score (Spoof Certainty Level): {probs[0][1].item()*100:.2f}%\n")

if __name__ == "__main__":
    main()
