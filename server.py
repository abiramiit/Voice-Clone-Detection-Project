from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import onnxruntime as ort
import numpy as np
import json
import uvicorn
import os

app = FastAPI()

# Phase 2 Deliverable: Load quantized CNN ONNX model
model_path = "models/cnn_model_quantized.onnx"
if not os.path.exists(model_path):
    print("Warning: Model not found. Mocking predictions.")
    ort_session = None
else:
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def process_audio_chunk(audio_bytes):
    # In a real app we'd convert the raw PCM audio_bytes to Log-Mel Spectrogram here.
    # We mock it to output the input shape the CNN expects: (1, 1, 128, 100)
    return np.random.randn(1, 1, 128, 100).astype(np.float32)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Log demo mode: storing predictions for Phase 4
    demo_log = []
    
    try:
        while True:
            # We receive audio chunks from the frontend WebRTC
            audio_bytes = await websocket.receive_bytes()
            features = process_audio_chunk(audio_bytes)
            
            if ort_session:
                # Inference (Target latency: < 100ms CPU)
                ort_inputs = {"input_values": features}
                ort_outs = ort_session.run(None, ort_inputs)
                logits = ort_outs[0][0]
                
                # Softmax
                exp_L = np.exp(logits - np.max(logits))
                probs = exp_L / np.sum(exp_L)
                fake_prob = float(probs[1])
            else:
                fake_prob = float(np.random.rand())
                
            is_fake = fake_prob > 0.5
            
            result = {
                "fake_prob": fake_prob,
                "label": "AI-GENERATED" if is_fake else "HUMAN"
            }
            demo_log.append(result)
            
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        # Save demo log to disk when client disconnects (Phase 4 requirement)
        with open("demo_log.json", "w") as f:
            json.dump(demo_log, f, indent=4)
        print("Client disconnected, demo log saved.")

@app.get("/")
async def get():
    return HTMLResponse(open("index.html", encoding="utf-8").read())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
