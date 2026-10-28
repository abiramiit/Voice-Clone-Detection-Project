from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import uvicorn
import threading, time
import asyncio
import websockets

app = FastAPI()

@app.websocket("/ws/stream")
async def ws_test(websocket: WebSocket, sample_rate: int = 48000):
    await websocket.accept()
    await websocket.send_text("Hello")
    await websocket.close()

# Mounted AFTER websocket
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="error")

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(2)

async def test():
    try:
        async with websockets.connect("ws://127.0.0.1:8002/ws/stream?sample_rate=48000") as ws:
            print("Received:", await ws.recv())
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
