import asyncio
import websockets

async def test():
    try:
        async with websockets.connect("ws://127.0.0.1:8001/ws/stream?sample_rate=16000") as ws:
            print("Connected 16000!")
    except Exception as e:
        print(f"Error 16000: {e}")
        
    try:
        async with websockets.connect("ws://127.0.0.1:8001/ws/stream?sample_rate=48000") as ws:
            print("Connected 48000!")
    except Exception as e:
        print(f"Error 48000: {e}")

if __name__ == "__main__":
    asyncio.run(test())
