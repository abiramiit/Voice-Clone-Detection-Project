import csv
import sys
import numpy as np
import collections

windows = []
try:
    with open("reports/stage16_microphone_windows.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            windows.append(row)
except Exception as e:
    print("Error:", e)
    sys.exit(1)

bonafide_count = sum(1 for w in windows if w['prediction'] == 'BONAFIDE')
spoof_count = sum(1 for w in windows if w['prediction'] == 'SPOOF')
spoof_probs = [float(w['spoof_probability']) for w in windows]

print("Total:", len(windows))
print("BONAFIDE:", bonafide_count)
print("SPOOF:", spoof_count)
print("Mean spoof:", np.mean(spoof_probs))
print("Median spoof:", np.median(spoof_probs))
print("P95 spoof:", np.percentile(spoof_probs, 95))
print("Min spoof:", np.min(spoof_probs))
print("Max spoof:", np.max(spoof_probs))
print("Std spoof:", np.std(spoof_probs))

transitions = 0
for i in range(1, len(windows)):
    if windows[i]['prediction'] != windows[i-1]['prediction']:
        transitions += 1
print("Transitions:", transitions)

rms_vals = [float(w['rms']) for w in windows]
print("Mean RMS:", np.mean(rms_vals))
print("Peak RMS:", np.max(rms_vals))
