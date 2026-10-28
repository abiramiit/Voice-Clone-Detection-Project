import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('.'))
import config
from src.real_audio_dataset import RealAudioDataset

# Basic Seed Enforcer
torch.manual_seed(42)

print("========================================")
print("1. ENVIRONMENT INIT & DATALOADER")
print("========================================")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}")

MODEL_NAME = "facebook/wav2vec2-base"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
dataset = RealAudioDataset(config.METADATA_PATH, 'train')

def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

print("========================================")
print("2. MODEL INITIALIZATION & FREEZE CHECK")
print("========================================")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Explicity freeze feature extractor
model.freeze_feature_encoder()
model.to(device)

fe_trainable = sum(p.numel() for p in model.wav2vec2.feature_extractor.parameters() if p.requires_grad)
print(f"Feature extractor trainable parameters = {fe_trainable}")

print("\n========================================")
print("3. CONTROLLED OPTIMIZATION TEST")
print("========================================")
# Select balanced batch: [0, 1] labels equivalently
test_batch_items = []
count_0 = 0
count_1 = 0
for i in range(len(dataset)):
    lab = dataset[i][1]
    if lab == 0 and count_0 < 2:
        test_batch_items.append(dataset[i])
        count_0 += 1
    elif lab == 1 and count_1 < 2:
        test_batch_items.append(dataset[i])
        count_1 += 1
    if count_0 == 2 and count_1 == 2:
        break

t_inputs, t_labels = collate_fn(test_batch_items)
t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
t_labels = t_labels.to(device)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = torch.amp.GradScaler('cuda')

clf_orig = model.classifier.weight.detach().clone()
proj_orig = model.projector.weight.detach().clone()
trans_orig = model.wav2vec2.encoder.layers[-1].attention.k_proj.weight.detach().clone()
fe_orig = model.wav2vec2.feature_extractor.conv_layers[0].conv.weight.detach().clone()

with torch.amp.autocast('cuda'):
    t_outputs = model(**t_inputs, labels=t_labels)
    t_loss = t_outputs.loss

print(f"Logits (Pre-step): {t_outputs.logits.detach().cpu()}")
print(f"Loss: {t_loss.item()}")

scaler.scale(t_loss).backward()

# Check Gradients
def check_gradient(grad, name):
    if grad is None:
        print(f"{name}: frozen / no gradient")
        return True # finite by default if not required
    elif not torch.all(torch.isfinite(grad)):
        print(f"{name}: non-finite (NaN/Inf)")
        return False
    else:
        norm = grad.norm().item()
        print(f"{name}: finite, gradient norm = {norm:.6f}")
        return True

c_grad = check_gradient(model.classifier.weight.grad, "classifier")
p_grad = check_gradient(model.projector.weight.grad, "projector")
t_grad = check_gradient(model.wav2vec2.encoder.layers[-1].attention.k_proj.weight.grad, "transformer")
f_grad = check_gradient(model.wav2vec2.feature_extractor.conv_layers[0].conv.weight.grad, "feature extractor")

ok = c_grad and p_grad and t_grad and f_grad
if not ok:
    print("STOP: NaN or Inf gradients detected across trainable layers.")
else:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    clf_new = model.classifier.weight.detach()
    proj_new = model.projector.weight.detach()
    trans_new = model.wav2vec2.encoder.layers[-1].attention.k_proj.weight.detach()
    fe_new = model.wav2vec2.feature_extractor.conv_layers[0].conv.weight.detach()
    
    print(f"\nclassifier changed: {not torch.allclose(clf_orig, clf_new)}")
    print(f"projector changed: {not torch.allclose(proj_orig, proj_new)}")
    print(f"transformer changed: {not torch.allclose(trans_orig, trans_new)}")
    print(f"Feature extractor unchanged: {torch.allclose(fe_orig, fe_new)}")
    
    print("NO CUDA OOM")
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        t_outputs2 = model(**t_inputs)
    print(f"New Logits (Post-step): {t_outputs2.logits.detach().cpu()}")

print("\n========================================")
print("4. SHORT CONTROLLED TRAINING TEST (20 BATCHES)")
print("========================================")
if ok:
    model.gradient_checkpointing_enable()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    batches_run = 0
    total_loss = 0
    class_dist = {0: 0, 1: 0}
    succ_steps = 0
    skip_steps = 0
    nan_count = 0
    
    optimizer.zero_grad()
    
    for step, (b_inputs, b_labels) in enumerate(train_loader):
        if batches_run >= 20: break
        b_inputs = {k: v.to(device) for k, v in b_inputs.items()}
        b_labels = b_labels.to(device)
        class_dist[b_labels.item()] += 1
        
        with torch.amp.autocast('cuda'):
            outputs = model(**b_inputs, labels=b_labels)
            loss = outputs.loss / 8.0 # accumulation 8
            
        scaler.scale(loss).backward()
        total_loss += loss.item() * 8.0
        batches_run += 1
        
        # We manually step and verify scaling
        if (batches_run % 8 == 0) or (batches_run == 20):
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            
            if scale_after < scale_before:
                skip_steps += 1
                nan_count += 1
            else:
                succ_steps += 1
            optimizer.zero_grad()
            
    print(f"20-Batch Mean Loss: {total_loss / batches_run:.4f}")
    print(f"Class Distribution processed: 0: {class_dist[0]}, 1: {class_dist[1]}")
    print(f"Successful Optimizer Steps: {succ_steps}")
    print(f"Skipped Optimizer Steps: {skip_steps}")
    print(f"Non-finite Gradient Count detected: {nan_count}")
else:
    print("Skipping 20 batch train due to gradient failures in Controlled Audit")
