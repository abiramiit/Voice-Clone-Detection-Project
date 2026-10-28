import subprocess, time, sys
api = subprocess.Popen([sys.executable, "-u", "scripts/api_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(3)
api.terminate()
try:
    stdout, _ = api.communicate(timeout=2)
    print("SERVER OUTPUT:")
    print(stdout)
except Exception as e:
    print(e)
