import time
import asyncio
import websockets

def wait_for_ready():
    for _ in range(60):
        try:
            with open("reports/test_native.log", "r") as f:
                if "Streaming Model Engine Ready!" in f.read():
                    return True
        except:
            pass
        time.sleep(1)
    return False

async def run_tests():
    print("Beginning handshake validations cleanly...")
    for sr in [16000, 48000]:
        try:
            print(f"Attempting ws://127.0.0.1:8001/ws/stream?sample_rate={sr}")
            async with websockets.connect(f"ws://127.0.0.1:8001/ws/stream?sample_rate={sr}") as ws:
                print(f"-> HANDSHAKE {sr} SUCCESS")
        except Exception as e:
            print(f"-> HANDSHAKE {sr} FAILED ERROR: {e}")

if wait_for_ready():
    asyncio.run(run_tests())
else:
    print("Server failed to start cleanly.")
