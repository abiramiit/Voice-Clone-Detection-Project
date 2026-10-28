import sys
import os
import torch
import pandas as pd
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification

sys.path.append(os.path.abspath('.'))
import config
from src.real_audio_dataset import RealAudioDataset

print("========================================")
print("1. INIT MODEL")
print("========================================")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Allocated device: {torch.cuda.get_device_name(0)}")

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", 
    num_labels=2
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Params: {total_params}")
print(f"Trainable Params: {trainable_params}")
print(f"Model Device: {next(model.parameters()).device}")

print("\n========================================")
print("2. DATASET LOADER")
print("========================================")
train_dataset = RealAudioDataset(config.METADATA_PATH, 'train')
val_dataset = RealAudioDataset(config.METADATA_PATH, 'validation')

print("\n========================================")
print("3. ONE SAMPLE FORWARD PASS")
print("========================================")
waveform, label, meta = train_dataset[0]
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
pred = torch.argmax(logits, dim=-1)

print(f"Input Shape: {inputs['input_values'].shape}")
print(f"Actual Label: {label}")
print(f"Logits Shape: {logits.shape}")
print(f"Logits: {logits}")
print(f"Predicted Class: {pred.item()}")
print(f"Forward pass occurred on CUDA: {logits.device.type == 'cuda'}")

print("\n========================================")
print("4. BATCH FORWARD PASS")
print("========================================")
batch_idx = []
bf_c = 0
sf_c = 0

for i in range(len(train_dataset)):
    _, l, _ = train_dataset[i]
    if l == 0 and bf_c < 2:
        batch_idx.append(i)
        bf_c += 1
    elif l == 1 and sf_c < 2:
        batch_idx.append(i)
        sf_c += 1
    if bf_c == 2 and sf_c == 2:
        break

waveforms = []
labels = []
for i in batch_idx:
    w, l, _ = train_dataset[i]
    waveforms.append(w)
    labels.append(l)

batch_inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

try:
    with torch.no_grad():
        batch_outputs = model(**batch_inputs)
    b_logits = batch_outputs.logits
    print(f"Batch Size: {len(waveforms)}")
    print(f"Batch Input Shape: {batch_inputs['input_values'].shape}")
    print(f"Labels: {labels}")
    print(f"Batch Logits Shape: {b_logits.shape}")
    print(f"Output Device: {b_logits.device}")
    print("NO CUDA OOM ERROR")
except RuntimeError as e:
    if "out of memory" in str(e):
        print("CUDA OUT OF MEMORY ERROR")
    else:
        print(f"RUNTIME ERROR: {e}")

print("\n========================================")
print("COMPLETED STAGE 4 SMOKE TEST")
print("========================================")
