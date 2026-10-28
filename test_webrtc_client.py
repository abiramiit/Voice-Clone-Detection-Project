import asyncio
import websockets
import soundfile as sf
import time
import json
import subprocess
import sys
import os

async def test_stream():
    url = "ws://127.0.0.1:8001/ws/stream?sample_rate=16000"
    audio_path = "data/real_subset/validation/bonafide/LA_D_1047731.flac"
    
    # Read Flac audio file representing actual browser representations PCM Float32 mappings natively.
    waveform, sr = sf.read(audio_path, dtype='float32')
    
    print("Connecting to secure WebSocket WebRTC Server endpoint...")
    async with websockets.connect(url) as websocket:
        print("Connected globally. Transmitting raw payload mappings.")
        chunk_size = int(sr * 1.0) # 1.0 second windows natively matched
        success_count = 0
        failed_count = 0
        e2e_arr = []
        gpu_arr = []
        
        for i in range(30): # Force 30 executions validating real stability requests boundaries natively.
            start = i * int(sr * 1.0)
            end = start + chunk_size
            if start >= len(waveform):
                # Rewind bounds for stability testing duration tracking natively
                start = start % len(waveform)
                end = start + chunk_size
                if end > len(waveform):
                    end = len(waveform)
                
            chunk = waveform[start:end]
            
            t0 = time.time()
            try:
                await websocket.send(chunk.tobytes())
                response = await websocket.recv()
                e2e = (time.time() - t0) * 1000
                
                res = json.loads(response)
                
                print(f"--- WINDOW {i+1} ---")
                print(f"Audio Duration: {len(chunk)/sr:.2f} seconds")
                print(f"Server Preprocessing/Convert Latency: {res['convert_ms']:.2f} ms")
                print(f"GPU Inference Latency: {res['inference_ms']:.2f} ms")
                print(f"End-to-End Actual Latency: {e2e:.2f} ms")
                print(f"Prediction: {res.get('prediction', 'ERROR')}")
                
                success_count += 1
                e2e_arr.append(e2e)
                gpu_arr.append(res['inference_ms'])
            except Exception as e:
                print("Failed inference window boundary:", e)
                failed_count += 1
                
            await asyncio.sleep(1.0)
            
        print("\n============================")
        print("--- STABILITY TEST COMPLETE ---")
        print(f"Total Windows: 30")
        print(f"Successful Windows: {success_count}")
        print(f"Failed Windows: {failed_count}")
        if len(e2e_arr) > 0:
            import numpy as np
            e2e_array = np.array(e2e_arr)
            print(f"E2E Mean Latency: {np.mean(e2e_array):.2f} ms")
            print(f"E2E Median Latency: {np.median(e2e_array):.2f} ms")
            print(f"E2E P95 Latency: {np.percentile(e2e_array, 95):.2f} ms")
            
            gpu_array = np.array(gpu_arr)
            print(f"GPU Mean Latency: {np.mean(gpu_array):.2f} ms")
        print("============================\n")

def main():
    print("Initializing Streaming Target server endpoints...")
    py = sys.executable
    server = subprocess.Popen([py, "-u", "scripts/stream_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8) 
    
    try:
        asyncio.run(test_stream())
    finally:
        print("Safely terminating proxy boundaries globally.")
        server.terminate()

if __name__ == "__main__":
    main()
