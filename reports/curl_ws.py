import urllib.request
req = urllib.request.Request("http://127.0.0.1:8001/ws/stream?sample_rate=16000", headers={"Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Version": "13", "Sec-WebSocket-Key": "SGVsbG8sIHdvcmxkIQ=="})
try:
    resp = urllib.request.urlopen(req)
    print(resp.read().decode())
except Exception as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print("Body:", e.read().decode())
