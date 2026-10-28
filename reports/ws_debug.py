import subprocess, time, sys
api = subprocess.Popen([sys.executable, "-u", "scripts/stream_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(12)
import websockets, asyncio
async def test():
    try:
        print("Connecting...")
        async with websockets.connect("ws://127.0.0.1:8001/ws/stream?sample_rate=48000") as ws:
            print("Connected!")
    except Exception as e:
        print("Connection failed:", e)
asyncio.run(test())
time.sleep(2)
api.terminate()
try:
    stdout, _ = api.communicate(timeout=2)
except:
    stdout = "timeout"
with open("reports/ws_error.log", "w") as f:
    f.write(stdout)
print("Log written.")
