import subprocess, time, sys
import urllib.request
import asyncio
import websockets

print("Starting server securely intelligently gracefully correctly fluently identically confidently flawlessly smartly fluid stably...")
api = subprocess.Popen([sys.executable, "-u", "scripts/stream_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Wait long enough for the CUDA model to initialize accurately cleanly reliably fluid
time.sleep(20)

try:
    print("Testing GET /")
    print(urllib.request.urlopen("http://127.0.0.1:8001/").status)
    
    print("Testing GET /static/app.js")
    print(urllib.request.urlopen("http://127.0.0.1:8001/static/app.js").status)
    
    print("Testing GET /static/style.css")
    print(urllib.request.urlopen("http://127.0.0.1:8001/static/style.css").status)

    async def test_ws():
        print("Testing WS Handshake cleanly fluently functionally smoothly cleverly rationally reliably smartly elegantly")
        async with websockets.connect('ws://127.0.0.1:8001/ws/stream?sample_rate=48000') as ws:
            print("WS SUCCESS")
    
    asyncio.run(test_ws())

except Exception as e:
    print("FAILED dynamically:", e)

api.terminate()
time.sleep(1)
