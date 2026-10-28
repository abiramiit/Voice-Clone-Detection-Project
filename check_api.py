import requests
import json
try:
    resp = requests.get("https://datasets-server.huggingface.co/rows?dataset=Bisher%2FASVspoof_2019_LA&config=default&split=test&offset=0&length=3", timeout=15)
    rows = resp.json().get('rows', [])
    for r in rows:
        print(r['row'].keys())
        print(f"Key: {r['row'].get('key')}, Label: {r['row'].get('label')}")
except Exception as e:
    print(f"Failed: {e}")
