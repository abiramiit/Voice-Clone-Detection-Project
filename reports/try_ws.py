import asyncio, websockets
async def test():
    try:
        async with websockets.connect('ws://127.0.0.1:8001/ws/stream?sample_rate=48000') as ws:
            print("Connected!")
    except Exception as e:
        print("Failed:", str(e))
asyncio.run(test())
