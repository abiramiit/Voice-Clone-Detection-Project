import subprocess
import time
import requests
import sys

test_log = []

def run_tests():
    print("Executing rigorous Python structural analyses...")
    code = subprocess.call([sys.executable, "-m", "py_compile", "scripts/api_server.py", "scripts/stream_server.py"])
    test_log.append(f"1. Python Syntax Module Check: {'PASS' if code == 0 else 'FAIL'}")

    print("Executing explicit FastAPI API startup evaluations...")
    api = subprocess.Popen([sys.executable, "-u", "scripts/api_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(6) # Wait for model load
    
    try:
        r = requests.get("http://127.0.0.1:8000/health")
        test_log.append(f"2. API Health Execution Bindings: {'PASS' if r.status_code == 200 else 'FAIL'}")
        
        with open("data/real_subset/validation/bonafide/LA_D_1047731.flac", "rb") as f:
            r = requests.post("http://127.0.0.1:8000/predict", files={"audio": ("1047731.flac", f, "audio/flac")})
            if r.status_code == 200 and "prediction" in r.json():
                test_log.append(f"3. /predict Logic Bound Evaluations: PASS")
            else:
                test_log.append(f"3. /predict Logic Bound Evaluations: FAIL (Status {r.status_code})")
    except Exception as e:
        test_log.append(f"2/3. API Validations: FAIL ({e})")
    finally:
        api.terminate()

    print("Initializing Streaming WebSocket constraints locally...")
    stream = subprocess.Popen([sys.executable, "-u", "scripts/stream_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    
    try:
        ws_test = subprocess.run([sys.executable, "-u", "test_webrtc_client.py"], capture_output=True, text=True)
        if "STABILITY TEST COMPLETE" in ws_test.stdout:
            test_log.append("4. WebSocket Real-Time Integration Limits: PASS")
        else:
            print("WebRTC Test Output:\n", ws_test.stdout)
            print("WebRTC Test Error:\n", ws_test.stderr)
            test_log.append("4. WebSocket Real-Time Integration Limits: FAIL")
    except Exception as e:
        test_log.append(f"4. WebSocket Validations: FAIL ({e})")
    finally:
        stream.terminate()
        
    print("\n--- FINAL TEST SYSTEM RESULTS ---")
    for log in test_log:
        print(log)

if __name__ == "__main__":
    run_tests()
