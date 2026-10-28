import sys
import platform

print("--- ENVIRONMENT CHECK ---")
print(f"1. Python version: {sys.version}")

try:
    import torch
    print(f"2. PyTorch version: {torch.__version__}")
    cuda_av = torch.cuda.is_available()
    print(f"3. CUDA available: {cuda_av}")
    if cuda_av:
        print(f"4. GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("4. GPU name: None")
except ImportError:
    print("2. PyTorch: NOT INSTALLED")
    print("3. CUDA available: FALSE")
    print("4. GPU name: None")

try:
    import transformers
    print(f"5. Transformers version: {transformers.__version__}")
except ImportError:
    print("5. Transformers: NOT INSTALLED")

try:
    import datasets
    print(f"6. Datasets version: {datasets.__version__}")
except ImportError:
    print("6. Datasets: NOT INSTALLED")

try:
    import librosa
    print(f"7. Librosa version: {librosa.__version__}")
except ImportError:
    print("7. Librosa: NOT INSTALLED")

try:
    import soundfile as sf
    print(f"8. SoundFile version: {sf.__version__}")
except ImportError:
    print("8. SoundFile: NOT INSTALLED")
print("-------------------------")
