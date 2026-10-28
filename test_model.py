import torch
import numpy as np
import requests
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import warnings

warnings.filterwarnings("ignore")

def safe_fetch_real_sample():
    print("Safely fetching 1 real empirical ASVspoof instance remotely to satisfy Constraints...")
    try:
        url = "https://datasets-server.huggingface.co/rows?dataset=LanceaKing%2Fasvspoof2019&config=LA&split=train&offset=0&length=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        row_data = data['rows'][0]['row']
        audio_dict = row_data['audio']
        
        # Datasets-server parses directly identically
        if isinstance(audio_dict, list): audio_dict = audio_dict[0]
        
        # Directly extract embedded decoded raw PCM values
        arr = np.array(audio_dict['array'])
        sr = audio_dict['sampling_rate']
        return [(arr, sr)]
    except Exception as e:
        print(f"Remote API payload fetch structurally unsupported without parquet parsing: {e}")
        return []

def main():
    print("=========================================================")
    print("   STAGE 4 & 5: WAV2VEC2 BINARY SPOOF DETECTOR MODEL     ")
    print("=========================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- HARDWARE DISCOVERED ---")
    print(f"Executing tensors locally via: {device.type.upper()}\n")
    
    print("--- LOADING PRETRAINED CHECKPOINT ---")
    print("Backbone: 'facebook/wav2vec2-base'")
    # Adding num_labels=2 dynamically binds a structural Binary Classification Head 
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
    model.to(device)
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n--- MODEL ARCHITECTURE SUMMARY ---")
    print(model)
    
    print(f"\n--- MODEL PARAMETERS ---")
    print(f"Total globally instanced parameters: {total_params:,}")
    print(f"Trainable actively back-proped params: {trainable_params:,}\n")
    
    print("--- REAL DATA TENSOR INFERENCE EVALUATION ---")
    samples = safe_fetch_real_sample()
    
    if not samples:
        print("[CRITICAL HALT] Cannot resolve a true genuine ASVspoof item from legacy API due to format mapping error.")
        print("Bypassing manual forward pass intentionally. Rule: 'Do not create synthetic/dummy audio/fabricated outputs'.")
        return
        
    for i, (audio_arr, orig_sr) in enumerate(samples):
        print(f"\n[Test Case {i+1}] Original Rate: {orig_sr} Hz, Points: {audio_arr.shape}")
        
        if orig_sr != 16000:
            print("Applying structural resample scaling mapping to 16kHz bounds...")
            audio_arr = librosa.resample(y=audio_arr, orig_sr=orig_sr, target_sr=16000)
            
        print("Funneling valid array properties through Hugging Face Extractor...")
        inputs = feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        print("Dispatching tensors sequentially through base layers into unified Binary Classification Logit Head...")
        model.eval()
        with torch.no_grad():
            outputs = model(input_values)
            
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        
        print("\n=== FORWARD INFERENCE METRICS EXPOSED ===")
        print(f"Raw Output Logits (Tensor Shape {logits.shape}): {logits.cpu().numpy()[0]}")
        print(f"Mapped Real Probability Array [Bona Fide, Spoof]: {probs.cpu().numpy()[0]}")
        print(f"Final Discriminant Predicted Class ID: {pred_class} (0=Bona Fide, 1=Spoof)")

if __name__ == "__main__":
    main()
