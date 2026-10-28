import os
import time
import json
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification
import librosa

def load_audio(path):
    w, sr = sf.read(path, dtype='float32')
    if len(w.shape) > 1 and w.shape[1] > 1:
        w = np.mean(w, axis=1)
    return w, sr

def infer(model, processor, waveform, sr, device):
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        with torch.autocast(device_type=device.type if torch.cuda.is_available() else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
            outputs = model(**inputs)
    logits = outputs.logits.float()
    probs = F.softmax(logits, dim=-1)
    p0 = probs[0, 0].item()
    p1 = probs[0, 1].item()
    return logits.cpu().numpy(), p0, p1, "BONAFIDE" if p0 > p1 else "SPOOF"

def compare():
    audio_path = "E:/resume-project1/data/real_subset/validation/bonafide/LA_D_1849134.flac"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = "E:/resume-project1/models/voice_clone_detector_stage5_2"
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # 1. Offline Mode
    w, sr = load_audio(audio_path)
    dur = len(w) / sr
    if sr != 16000:
        w_off = librosa.resample(w, orig_sr=sr, target_sr=16000)
    else:
        w_off = w
    rms_off = np.sqrt(np.mean(w_off**2))
    peak_off = np.max(np.abs(w_off))
    
    logits_off, p0_off, p1_off, pred_off = infer(model, processor, w_off, 16000, device)

    # 2. Streaming Sim (Take exactly 1 second, no resampling to match browser chunks, or what does it do?)
    # Stream converts whatever is given to 16000. Wait, browser gives chunks of 48000!
    # A 1-second chunk of the audio file in the original sample rate!
    chunk = w[:sr] # 1 sec chunk
    
    # 2a. Stream Backend Processing
    if sr != 16000:
        samples = round(len(chunk) * float(16000) / sr)
        indices = np.linspace(0, len(chunk)-1, samples).astype(int)
        w_str = chunk[indices]
    else:
        w_str = chunk

    rms_str = np.sqrt(np.mean(w_str**2))
    peak_str = np.max(np.abs(w_str))
    
    logits_str, p0_str, p1_str, pred_str = infer(model, processor, w_str, 16000, device)
    
    # Compute Diff
    logit_diff = float(np.sum(np.abs(logits_off - logits_str)))

    # Output CSV
    data = []
    data.append({
        "audio_id": os.path.basename(audio_path),
        "offline_prediction": pred_off,
        "offline_bonafide_probability": p0_off,
        "offline_spoof_probability": p1_off,
        "stream_prediction": pred_str,
        "stream_bonafide_probability": p0_str,
        "stream_spoof_probability": p1_str,
        "sample_rate": sr,
        "sample_count": len(chunk),
        "duration": dur,
        "rms": rms_off,
        "peak": peak_off,
        "logit_difference": logit_diff
    })
    
    df = pd.DataFrame(data)
    os.makedirs("E:/resume-project1/reports", exist_ok=True)
    df.to_csv("E:/resume-project1/reports/stage16_offline_vs_streaming.csv", index=False)
    print("DONE! File generated.")

if __name__ == "__main__":
    compare()
