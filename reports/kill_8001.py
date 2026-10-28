import subprocess
out = subprocess.check_output('netstat -ano | findstr :8001', shell=True).decode()
for line in out.splitlines():
    if "LISTENING" in line:
        pid = line.strip().split()[-1]
        print("Killing PID:", pid)
        subprocess.run(f"taskkill /F /PID {pid}", shell=True)
