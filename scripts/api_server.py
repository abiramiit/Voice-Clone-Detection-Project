import os
import sys
import time
import uvicorn
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification
import io

app = FastAPI(title="Voice Clone Detector API")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if str(device) != 'cuda':
    print("Warning: CUDA is physically unavailable. Defaulting execution paths securely over CPU mappings natively.")
model_dir = "models/voice_clone_detector_stage5_2"

print("Loading processor...")
try:
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
except Exception as e:
    print(f"Processor load error: {e}")
    sys.exit(1)

print("Loading model...")
try:
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Model load error: {e}")
    sys.exit(1)

print("Model loaded successfully on", device)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "model_path": os.path.abspath(model_dir)
    }

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not audio.filename.endswith(('.wav', '.flac')):
        raise HTTPException(status_code=400, detail="Only .wav or .flac files are supported")
        
    start_time = time.time()
    
    try:
        content = await audio.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Payload size exceeded 10MB bounds safely")
            
        with io.BytesIO(content) as buffer:
            waveform, sr = sf.read(buffer, dtype='float32')
            
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
            
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model(**inputs)
            
        logits = outputs.logits.float()
        probs = F.softmax(logits, dim=-1)
        
        p0 = probs[0, 0].item()
        p1 = probs[0, 1].item()
        
        pred_class = 0 if p0 > p1 else 1
        pred_label_str = "BONAFIDE" if pred_class == 0 else "SPOOF"
        
        latency = (time.time() - start_time) * 1000
        
        return JSONResponse({
            "prediction": pred_label_str,
            "bonafide_probability": p0,
            "spoof_probability": p1,
            "latency_ms": latency,
            "sample_rate": orig_sr,
            "duration_seconds": duration
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
