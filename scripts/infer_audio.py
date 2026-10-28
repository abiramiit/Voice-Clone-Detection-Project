import argparse
import os
import sys
import time
import json
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification

def infer_audio(audio_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("========================================")
    print("WAV2VEC2 AUDIO INFERENCE")
    print("========================================")
    
    try:
        waveform, sr = sf.read(audio_path, dtype='float32')
    except Exception as e:
        print(f"Error: audio file cannot be decoded. Exception: {e}")
        sys.exit(1)
        
    duration = len(waveform) / sr
    orig_sr = sr
    
    if sr != 16000:
        try:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000
        except ImportError:
            from scipy.signal import resample
            samples = round(len(waveform) * float(16000) / sr)
            waveform = resample(waveform, samples)
            sr = 16000

    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)

    print(f"Audio: {audio_path}")
    if orig_sr != sr:
        print(f"Sample rate: {orig_sr} (Resampled to {sr})")
    else:
        print(f"Sample rate: {orig_sr}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Device: {device}")
    
    model_dir = "models/voice_clone_detector_stage5_2"
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    start_time = time.time()
    
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits.float()
    probs = F.softmax(logits, dim=-1)
    
    p0 = probs[0, 0].item()
    p1 = probs[0, 1].item()
    pred_class = 0 if p0 > p1 else 1
    pred_label_str = "BONAFIDE" if pred_class == 0 else "SPOOF"
    
    latency = (time.time() - start_time) * 1000
    
    print("\nBonafide probability: {:.6f}".format(p0))
    print("Spoof probability: {:.6f}".format(p1))
    print("\nPrediction: {}".format(pred_label_str))
    print("Inference latency: {:.2f} ms".format(latency))
    print("========================================\n")
    
    res_dict = {
        "audio_path": os.path.abspath(audio_path),
        "actual_duration": duration,
        "sample_rate": orig_sr,
        "probabilities": {
            "0": p0,
            "1": p1
        },
        "logits": {
            "0": logits[0,0].item(),
            "1": logits[0,1].item()
        },
        "prediction": pred_label_str,
        "latency_ms": latency
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/inference_result.json", "w") as f:
        json.dump(res_dict, f, indent=4)
        
    print("MODEL:")
    print(f"{model_dir}/")
    print("")
    print("AUDIO:")
    print(f"{os.path.normpath(audio_path)}")
    print("")
    print("PREDICTION:")
    print(f"{pred_label_str}")
    print("")
    print("PROBABILITIES:")
    print(f"bonafide = {p0:.6f}")
    print(f"spoof = {p1:.6f}")
    print("")
    print("LATENCY:")
    print(f"{latency:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wav2Vec2 Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: audio file {args.audio} not found.")
        sys.exit(1)
        
    infer_audio(args.audio)
