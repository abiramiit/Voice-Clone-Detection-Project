import urllib.request
import time
import asyncio
import websockets

def test():
    for _ in range(30):
        try:
            r1 = urllib.request.urlopen("http://127.0.0.1:8001/").status
            if r1 == 200:
                print("GET / ->", r1)
                print("GET /static/app.js ->", urllib.request.urlopen("http://127.0.0.1:8001/static/app.js").status)
                print("GET /static/style.css ->", urllib.request.urlopen("http://127.0.0.1:8001/static/style.css").status)
                return True
        except Exception:
            time.sleep(1)
    return False

if test():
    async def tws():
        async with websockets.connect('ws://127.0.0.1:8001/ws/stream?sample_rate=48000') as ws:
            print("WS SUCCESS")
    asyncio.run(tws())
