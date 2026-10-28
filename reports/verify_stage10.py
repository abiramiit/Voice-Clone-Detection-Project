import subprocess, time, sys

print("Spawning backend securely...")
api = subprocess.Popen([sys.executable, "-u", "scripts/stream_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Wait for CUDA loading and port binding
time.sleep(25)

import asyncio
import websockets

async def run_tests():
    print("Beginning handshake validations cleanly...")
    for sr in [16000, 48000]:
        try:
            print(f"Attempting ws://127.0.0.1:8001/ws/stream?sample_rate={sr}")
            async with websockets.connect(f"ws://127.0.0.1:8001/ws/stream?sample_rate={sr}") as ws:
                print(f"-> HANDSHAKE {sr} SUCCESS")
        except Exception as e:
            print(f"-> HANDSHAKE {sr} FAILED ERROR: {e}")

try:
    asyncio.run(run_tests())
except Exception as e:
    print(f"Testing architecture crashed natively: {e}")

api.terminate()
time.sleep(1)
print("Validation Finished.")
