import sys
import importlib
import os
import torch

print("========================================")
print("1. ENVIRONMENT CHECK")
print("========================================")
print(f"- Python executable: {sys.executable}")
print(f"- Python version: {sys.version.split()[0]}")
pkgs = {'torch': 'torch', 'transformers': 'transformers', 'soundfile': 'soundfile', 'pandas': 'pandas', 'numpy': 'numpy', 'scikit-learn': 'sklearn'}
missing = []
for name, mod_name in pkgs.items():
    try:
        mod = importlib.import_module(mod_name)
        print(f"- {name}: {mod.__version__}")
    except ImportError:
        missing.append(name)
        
if missing:
    print(f"Missing packages: {missing}")
    sys.exit(1)

print(f"- CUDA available: {torch.cuda.is_available()}")
print(f"- CUDA runtime: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"- GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("- GPU name: None")

import pandas as pd
from transformers import AutoProcessor
sys.path.append(os.path.abspath('.'))
import config
from src.real_audio_dataset import RealAudioDataset

print("\n========================================")
print("2. CREATE DATASET LOADER")
print("========================================")
train_dataset = RealAudioDataset(config.METADATA_PATH, 'train')
val_dataset = RealAudioDataset(config.METADATA_PATH, 'validation')
print("Dataset loader initialized successfully.")

print("\n========================================")
print("3. WAV2VEC2 PROCESSOR")
print("========================================")
processor = AutoProcessor.from_pretrained(config.MODEL_NAME)
print(f"Loaded processor: {config.MODEL_NAME}")

print("\n========================================")
print("4. PROCESS ONE REAL SAMPLE")
print("========================================")
waveform, label, metadata = train_dataset[0]
print(f"- original file: {metadata['original_audio_file_name']}")
print(f"- label: {label}")
print(f"- sample rate: {metadata['sample_rate']}")
print(f"- waveform shape: {waveform.shape}")
print(f"- waveform duration: {metadata['duration_seconds']}s")

inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
print(f"- processed input_values shape: {inputs.input_values.shape}")
print(f"- dtype: {inputs.input_values.dtype}")

print("\n========================================")
print("5. PROCESS A SMALL REAL BATCH")
print("========================================")
batch_idx = []
bonafide_count = 0
spoof_count = 0
for i in range(len(train_dataset)):
    _, l, _ = train_dataset[i]
    if l == 0 and bonafide_count < 2:
        batch_idx.append(i)
        bonafide_count += 1
    elif l == 1 and spoof_count < 2:
        batch_idx.append(i)
        spoof_count += 1
    if bonafide_count == 2 and spoof_count == 2:
        break

waveforms = []
labels = []
for i in batch_idx:
    w, l, _ = train_dataset[i]
    waveforms.append(w)
    labels.append(l)

batch_inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
print(f"Batch size: {len(waveforms)}")
print(f"Input tensor shape: {batch_inputs.input_values.shape}")
print(f"Labels: {labels}")
print(f"Attention mask shape: {batch_inputs.attention_mask.shape}")
print(f"dtype: {batch_inputs.input_values.dtype}")

print("\n========================================")
print("6. VERIFY LABELS")
print("========================================")
train_metadata = pd.read_csv(config.METADATA_PATH)
train_split = train_metadata[train_metadata['split'] == 'train']
val_split = train_metadata[train_metadata['split'] == 'validation']

t_b = len(train_split[train_split['label'] == 0])
t_s = len(train_split[train_split['label'] == 1])
v_b = len(val_split[val_split['label'] == 0])
v_s = len(val_split[val_split['label'] == 1])

print("TRAIN:")
print(f"bonafide = {t_b}")
print(f"spoof = {t_s}")
print("\nVALIDATION:")
print(f"bonafide = {v_b}")
print(f"spoof = {v_s}")

if t_b != 500 or t_s != 500 or v_b != 100 or v_s != 100:
    print("COUNTS DIFFER. STOPPING.")
    sys.exit(1)

print("\n========================================")
print("7. DATA LEAKAGE CHECK")
print("========================================")
dup_audio_paths = train_metadata.duplicated(subset=['audio_path']).sum()
dup_orig_files = train_metadata.duplicated(subset=['original_audio_file_name']).sum()

train_speakers = set(train_split['speaker_id'].unique()) - {'unknown'}
val_speakers = set(val_split['speaker_id'].unique()) - {'unknown'}
overlap = train_speakers.intersection(val_speakers)

print(f"Duplicate original filenames: {dup_orig_files}")
print(f"Duplicate audio paths: {dup_audio_paths}")
print(f"train speakers: {len(train_speakers)}")
print(f"validation speakers: {len(val_speakers)}")
print(f"overlap: {len(overlap)}")

if overlap:
    print("ISSUE: Speaker overlap between train and validation exists.")
    print("Overlap IDs:", list(overlap))
else:
    print("No speaker overlap detected.")
