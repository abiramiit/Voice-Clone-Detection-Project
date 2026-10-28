import urllib.request
import sys

def check_url(url):
    try:
        req = urllib.request.urlopen(url)
        content = req.read().decode('utf-8')
        print(f"PASS: {url} -> {req.status}")
        return True
    except Exception as e:
        print(f"FAIL: {url} -> {e}")
        return False

sys.exit(0 if check_url("http://127.0.0.1:8001/") and check_url("http://127.0.0.1:8001/static/app.js") and check_url("http://127.0.0.1:8001/static/style.css") else 1)
