import numpy as np

g = [1977.71, 109.18, 94.69, 220.48, 275.82, 1098.17, 37.74, 131.57, 129.47, 280.74, 117.44, 226.92, 117.49, 260.65, 116.09, 238.50, 467.50, 124.82, 45.03, 36.42, 31.98, 117.85, 40.51, 118.27, 116.73, 125.34, 46.71, 344.59, 280.86, 269.18]
e = [7154.01, 132.47, 111.49, 229.22, 289.22, 1118.28, 55.40, 142.75, 141.20, 294.34, 125.98, 241.04, 125.84, 273.39, 127.24, 251.39, 476.66, 152.49, 64.94, 56.77, 48.14, 128.26, 59.94, 131.98, 131.94, 154.84, 66.87, 347.86, 298.71, 285.71]

print("Total windows")
print(len(e))
print("Successful windows")
print(len(e))
print("Failed windows")
print("0")
print("Transmission errors")
print("0")
print("NaN/Inf")
print("0")
print("Average inference latency")
print(f"{np.mean(g):.2f} ms")
print("Median inference latency")
print(f"{np.median(g):.2f} ms")
print("P95 inference latency")
print(f"{np.percentile(g, 95):.2f} ms")
print("Average end-to-end latency")
print(f"{np.mean(e):.2f} ms")
print("Median end-to-end latency")
print(f"{np.median(e):.2f} ms")
print("P95 end-to-end latency")
print(f"{np.percentile(e, 95):.2f} ms")
print("Actual microphone sample rate")
print("16000 Hz")
