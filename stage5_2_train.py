import sys
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath('.'))
import config
from src.real_audio_dataset import RealAudioDataset

# Basic Seed Enforcer
torch.manual_seed(42)
np.random.seed(42)

start_time = time.time()
print("========================================")
print("1. ENVIRONMENT INIT & DATALOADER")
print("========================================")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Python executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

MODEL_NAME = "facebook/wav2vec2-base"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
train_dataset = RealAudioDataset(config.METADATA_PATH, 'train')
val_dataset = RealAudioDataset(config.METADATA_PATH, 'validation')

def count_classes(ds):
    c0 = 0
    c1 = 0
    for i in range(len(ds)):
        l = ds[i][1]
        if l == 0: c0 += 1
        elif l == 1: c1 += 1
    return c0, c1

tc0, tc1 = count_classes(train_dataset)
vc0, vc1 = count_classes(val_dataset)

print(f"Train count: {len(train_dataset)}")
print(f"Validation count: {len(val_dataset)}")
print(f"Class distribution (Train): Bonafide={tc0}, Spoof={tc1}")
print(f"Class distribution (Val): Bonafide={vc0}, Spoof={vc1}")

def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print("========================================")
print("2. MODEL INITIALIZATION")
print("========================================")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.config.id2label = {0: "bonafide", 1: "spoof"}
model.config.label2id = {"bonafide": 0, "spoof": 1}

# Explicity freeze feature extractor
model.freeze_feature_encoder()
model.to(device)
model.gradient_checkpointing_enable()

# Count params
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

LR = 3e-5
EPOCHS = 3
GRAD_ACCUM_STEPS = 8

print(f"Trainable parameter count: {trainable_params}")
print(f"Frozen parameter count: {frozen_params}")
print(f"Learning rate: {LR}")
print(f"Batch size: 1")
print(f"Accumulation steps: {GRAD_ACCUM_STEPS}")
print(f"Epochs: {EPOCHS}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda')

print("\n========================================")
print("3. TRAINING LOOP START")
print("========================================")
best_f1 = -1.0
best_val_loss = float('inf')
SAVE_DIR = os.path.join("models", "voice_clone_detector_stage5_2")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()
    
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / float(GRAD_ACCUM_STEPS)
            
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_train_loss += loss.item() * GRAD_ACCUM_STEPS
        
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation Eval
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(**inputs, labels=labels)
                v_loss = outputs.loss
                
            total_val_loss += v_loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_val_loss = total_val_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    pred_c0 = sum(1 for p in all_preds if p == 0)
    pred_c1 = sum(1 for p in all_preds if p == 1)
    
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Precision: {prec:.4f}")
    print(f"Validation Recall: {rec:.4f}")
    print(f"Validation F1: {f1:.4f}")
    print("Confusion Matrix:")
    print("                Predicted")
    print("              Bonafide Spoof")
    print(f"Actual Bonafide  {cm[0][0]:<7} {cm[0][1]}")
    print(f"       Spoof     {cm[1][0]:<7} {cm[1][1]}")
    print(f"Predicted class distribution: Bonafide={pred_c0}, Spoof={pred_c1}")
    print(f"Actual class distribution: Bonafide={vc0}, Spoof={vc1}")
    
    # Best model saving criteria:
    is_new_best = False
    if f1 > best_f1:
        is_new_best = True
    elif f1 == best_f1:
        if avg_val_loss < best_val_loss:
            is_new_best = True
            
    if is_new_best:
        best_f1 = f1
        best_val_loss = avg_val_loss
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)
        
        # Save config metric
        metrics_dict = {
            "epoch": epoch+1,
            "best_f1": float(f1),
            "best_val_loss": float(avg_val_loss),
            "precision": float(prec),
            "recall": float(rec),
            "accuracy": float(acc)
        }
        with open(os.path.join(SAVE_DIR, "training_config.json"), 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        print(f"--> Saved NEW best checkpoint at Epoch {epoch+1} (F1: {f1:.4f}, Val Loss: {avg_val_loss:.4f})")

print("\n========================================")
print("4. POST-TRAINING RELOAD INFERENCE")
print("========================================")
# Reload model from disk
test_model = Wav2Vec2ForSequenceClassification.from_pretrained(SAVE_DIR)
test_model.to(device)
test_model.eval()

t_total_val_loss = 0
t_all_preds = []
t_all_labels = []

contains_nan_inf = False

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = test_model(**inputs, labels=labels)
            
        logits = outputs.logits
        v_loss = outputs.loss
        if not torch.all(torch.isfinite(logits)):
            contains_nan_inf = True
        
        preds = torch.argmax(logits, dim=-1)
        t_all_preds.extend(preds.cpu().numpy())
        t_all_labels.extend(labels.cpu().numpy())
        t_total_val_loss += v_loss.item()
        
t_acc = accuracy_score(t_all_labels, t_all_preds)
t_prec = precision_score(t_all_labels, t_all_preds, zero_division=0)
t_rec = recall_score(t_all_labels, t_all_preds, zero_division=0)
t_f1 = f1_score(t_all_labels, t_all_preds, zero_division=0)
t_cm = confusion_matrix(t_all_labels, t_all_preds, labels=[0, 1])

pred_c0_t = sum(1 for p in t_all_preds if p == 0)
pred_c1_t = sum(1 for p in t_all_preds if p == 1)

print("INFERENCE RELOAD RESULTS:")
print(f"Accuracy: {t_acc:.4f} (Matches best? {abs(t_acc - metrics_dict['accuracy']) < 1e-4})")
print(f"Precision: {t_prec:.4f} (Matches best? {abs(t_prec - metrics_dict['precision']) < 1e-4})")
print(f"Recall: {t_rec:.4f} (Matches best? {abs(t_rec - metrics_dict['recall']) < 1e-4})")
print(f"F1: {t_f1:.4f} (Matches best? {abs(t_f1 - metrics_dict['best_f1']) < 1e-4})")
print("Reloaded Confusion Matrix:")
print("                Predicted")
print("              Bonafide Spoof")
print(f"Actual Bonafide  {t_cm[0][0]:<7} {t_cm[0][1]}")
print(f"       Spoof     {t_cm[1][0]:<7} {t_cm[1][1]}")
print(f"Final Prediction Dist: Bonafide={pred_c0_t}, Spoof={pred_c1_t}")
print(f"Any NaN/Inf outputs: {contains_nan_inf}")

end_time = time.time()
print(f"\nTotal Training Time: {(end_time - start_time)/60:.2f} mins")

if torch.cuda.is_available():
    max_mem = torch.cuda.max_memory_allocated(device)
    print(f"Maximum CUDA Memory Allocated: {max_mem / 1024**2:.2f} MB")
    
print("COMPLETED")
