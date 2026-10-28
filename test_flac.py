import soundfile as sf
import os

for i in range(5):
    path = f'data/parquet_test/test_audio_{i}.wav'
    if os.path.exists(path):
        try:
            info = sf.info(path)
            print(f"File: {path}, SR: {info.samplerate}, Duration: {info.duration:.2f}s")
        except Exception as e:
            print(f"Failed to read {path}: {e}")
