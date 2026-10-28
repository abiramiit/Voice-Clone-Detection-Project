import requests
import json
import sys
import time

print("Testing /health ...", flush=True)
try:
    h_resp = requests.get("http://127.0.0.1:8000/health")
    print(json.dumps(h_resp.json(), indent=2))
except Exception as e:
    print(f"Health check failed: {e}")
    sys.exit(1)

audio_path = "data/real_subset/validation/bonafide/LA_D_1047731.flac"
print(f"\nTesting /predict with {audio_path} ...", flush=True)
try:
    with open(audio_path, 'rb') as f:
        files = {'audio': (audio_path, f, 'audio/flac')}
        start_req = time.time()
        p_resp = requests.post("http://127.0.0.1:8000/predict", files=files)
        req_lat = (time.time() - start_req)*1000
    if p_resp.status_code == 200:
        data = p_resp.json()
        print("Prediction API response:\n", json.dumps(data, indent=2))
        print(f"\nTotal API roundtrip latency: {req_lat:.2f} ms")
    else:
        print(f"API Error {p_resp.status_code}: {p_resp.text}")
except Exception as e:
    print(f"/predict check failed: {e}")
