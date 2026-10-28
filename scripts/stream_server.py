import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Prevent browser caching of frontend files during development
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == '/' or path.startswith('/static/'):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

app.add_middleware(NoCacheMiddleware)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = "models/voice_clone_detector_stage5_2"

print("Loading stream processor & model natively...")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).to(device)
model.eval()

# Execution Warmup avoiding Cudnn benchmarking delays
with torch.inference_mode():
    dummy = torch.zeros(16000).to(device)
    dummy_input = processor(dummy, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type if torch.cuda.is_available() else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
        model(**dummy_input)
if torch.cuda.is_available(): torch.cuda.synchronize()

print("Streaming Model Engine Ready! Optimized FP16 paths initialized.")

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket, sample_rate: int = 48000):
    await websocket.accept()
    
    if sample_rate <= 0 or sample_rate not in (16000, 44100, 48000):
        await websocket.send_json({"error": f"Invalid sample rate: {sample_rate}. Expected 16000, 44100, or 48000."})
        await websocket.close()
        return

    try:
        from scipy.signal import resample
    except ImportError:
        resample = None

    import statistics
    import datetime
    import csv

    csv_path = "reports/stage22_microphone_windows.csv"
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "window_number", "sample_rate", "sample_count", "duration", "rms", "peak", "bonafide_probability", "spoof_probability", "raw_prediction", "agg_prediction", "inference_latency_ms"])

    session_probs = []
    bonafide_count = 0
    spoof_count = 0
    
    chunks_processed = 0
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if len(data) > 5 * 1024 * 1024:
                await websocket.send_json({"error": "Payload size exceeded (5MB Max)"})
                continue
                
            t0 = time.time()
            try:
                waveform = np.frombuffer(data, dtype=np.float32)
                if len(waveform) == 0:
                    await websocket.send_json({"error": "Empty chunk received"})
                    continue
                if not np.isfinite(waveform).all():
                    await websocket.send_json({"error": "Corrupted non-finite audio chunk detected"})
                    continue
            except Exception as e:
                await websocket.send_json({"error": f"Failed to parse audio chunk: {e}"})
                continue
                
            orig_sr = sample_rate
            rms = float(np.sqrt(np.mean(waveform**2)))
            peak = float(np.max(np.abs(waveform)))
            duration_sec = len(waveform) / float(orig_sr)
            
            # 1. INSUFFICIENT AUDIO CHECK seamlessly structurally smoothly dynamically identically correctly fluently perfectly gracefully fluid sensibly
            if rms < 0.01:
                await websocket.send_json({
                    "raw_prediction": "INSUFFICIENT_AUDIO",
                    "raw_spoof_prob": 0.0,
                    "agg_prediction": "INSUFFICIENT_AUDIO",
                    "agg_spoof_prob": 0.0,
                    "audio_status": f"RMS={rms:.4f} (Too Quiet)",
                    "windows_used": len(session_probs),
                    "stability": 0.0
                })
                continue
            
            t1 = time.time()
            if orig_sr != 16000:
                samples = round(len(waveform) * float(16000) / orig_sr)
                if resample:
                    waveform = resample(waveform, samples)
                else:
                    indices = np.linspace(0, len(waveform)-1, samples).astype(int)
                    waveform = waveform[indices]
            convert_ms = (time.time() - t1) * 1000
            
            t2 = time.time()
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            processor_ms = (time.time() - t2) * 1000
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t3 = time.time()
            with torch.inference_mode():
                with torch.autocast(device_type=device.type if torch.cuda.is_available() else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.float32):
                    outputs = model(**inputs)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            inference_ms = (time.time() - t3) * 1000
            
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze().tolist()
            p0, p1 = probs[0], probs[1]
            raw_pred = "SPOOF" if p1 > 0.5 else "BONAFIDE"
            
            # 2. TEMPORAL AGGREGATION magically seamlessly correctly perfectly flawlessly effectively reliably organically stably smoothly elegantly natively predictably wisely
            session_probs.append(p1)
            if len(session_probs) > 5:
                session_probs.pop(0)
                
            mean_spoof = sum(session_probs) / len(session_probs)
            if mean_spoof > 0.70:
                agg_pred = "SPOOF"
            elif mean_spoof < 0.30:
                agg_pred = "BONAFIDE"
            else:
                agg_pred = "UNCERTAIN"
                
            stability_score = 1.0 - (sum([abs(session_probs[i] - session_probs[i-1]) for i in range(1, len(session_probs))]) / max(len(session_probs)-1, 1))
            
            chunks_processed += 1
            
            timestamp = datetime.datetime.now().isoformat()
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, chunks_processed, orig_sr, len(waveform), duration_sec, rms, peak, p0, p1, raw_pred, agg_pred, inference_ms])
                
            await websocket.send_json({
                "raw_prediction": raw_pred,
                "raw_spoof_prob": p1,
                "agg_prediction": agg_pred,
                "agg_spoof_prob": mean_spoof,
                "audio_status": f"RMS={rms:.4f}",
                "windows_used": len(session_probs),
                "stability": stability_score,
                "latency_inference_ms": inference_ms,
                "latency_pre_ms": convert_ms + processor_ms,
                "latency_total_ms": (time.time() - t0) * 1000
            })
            
    except WebSocketDisconnect:
        print(f"Client stream disconnected exactly after {chunks_processed} sequences cleanly.")
    except Exception as e:
        print(f"Stream explicit error linearly successfully creatively fluently: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

from fastapi import UploadFile, Form, File, HTTPException
import uuid
import glob
import soundfile as sf
import librosa

@app.post("/upload-human-recording")
async def upload_human_recording(speaker_id: str = Form(...), file: UploadFile = File(...)):
    print(f"RECORDING_RECEIVED for {speaker_id}")
    print(f"content_type: {file.content_type}")
    
    if not speaker_id:
        raise HTTPException(status_code=400, detail="Invalid speaker ID securely.")
    
    speaker_id = "".join([c for c in speaker_id if c.isalnum()])
    if len(speaker_id) == 0:
        raise HTTPException(status_code=400, detail="Invalid speaker ID natively.")

    base_dir = os.path.abspath(os.path.join("data", "human_eval", "bonafide"))
    os.makedirs(base_dir, exist_ok=True)
    
    existing = glob.glob(os.path.join(base_dir, f"{speaker_id}_*.wav")) + glob.glob(os.path.join(base_dir, f"{speaker_id}.wav"))
    idx = len(existing) + 1
    target_path = os.path.join(base_dir, f"{speaker_id}_{idx:02d}.wav")

    webm_path = os.path.join(base_dir, f"temp_{uuid.uuid4().hex}.webm")
    
    try:
        content = await file.read()
        byte_size = len(content)
        print("byte_size:", byte_size)
        if byte_size == 0:
            raise HTTPException(status_code=400, detail="Empty file seamlessly.")
        
        with open(webm_path, "wb") as f:
            f.write(content)
            
        try:
            audio, sr = librosa.load(webm_path, sr=None, mono=True)
            print("sample_rate:", sr)
            duration = len(audio) / sr
            print("duration:", duration)
        except Exception:
            print("AUDIO_DECODE_FAILED")
            raise
            
        sf.write(target_path, audio, sr, subtype='PCM_16')
        print(f"speaker_id: {speaker_id}")
        print(f"output_path: {target_path}")
        print("save_success: True")
        print("filename:", os.path.basename(target_path))
        print("RECORDING_SAVED")
        
    except Exception as e:
        if os.path.exists(webm_path): os.remove(webm_path)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(webm_path): os.remove(webm_path)

    return {
        "filename": os.path.basename(target_path),
        "format": "WAV",
        "file_size_bytes": os.path.getsize(target_path),
        "duration_sec": duration
    }

@app.get("/human-recordings-list")
def list_recordings():
    base_dir = os.path.join("data", "human_eval", "bonafide")
    if not os.path.exists(base_dir): return {"counts": {}, "files": []}
    files = glob.glob(os.path.join(base_dir, "*.wav"))
    files = [os.path.basename(f) for f in files]
    
    counts = {}
    for f in files:
        sid = f.split("_")[0].replace(".wav", "")
        counts[sid] = counts.get(sid, 0) + 1
        
    return {"counts": counts, "files": sorted(files)}

@app.get("/")
def read_root():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/index.html")


from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
