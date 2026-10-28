import sys, os, time, subprocess, requests
import fastapi, uvicorn, websockets, torch

print("Phase 1: Environment Check")
assert r"E:\resume-project1\.venv_dataset\Scripts\python.exe".lower() in sys.executable.lower(), "Incorrect Python Executable!"
assert os.path.exists("models/voice_clone_detector_stage5_2"), "Model missing"
assert os.path.exists("frontend/index.html"), "Frontend missing"
assert os.path.exists("scripts/stream_server.py"), "stream_server missing"
print("Phase 1: Environment Passed.")

print("Phase 2: Start Backend")
server = subprocess.Popen([sys.executable, "-u", "scripts/stream_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(10) 
try:
    r = requests.get("http://127.0.0.1:8001/")
    print(f"Phase 2: Frontend Loads: HTTP {r.status_code}")
except Exception as e:
    print(f"Phase 2 Error: {e}")
finally:
    server.terminate()
