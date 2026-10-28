import sys
import os
import time
import datetime
import shutil
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

# 0. Set seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Configurations
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LR = 5e-5
EPOCHS = 3
MODEL_NAME = "facebook/wav2vec2-base"

print("========================================")
print("1. ENVIRONMENT CHECK")
print("========================================")
print(f"- Python executable: {sys.executable}")
print(f"- PyTorch version: {torch.__version__}")
print(f"- CUDA availability: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"- CUDA runtime: {torch.version.cuda}")
    print(f"- GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("- GPU name: None")

print(f"- Batch size (per device): {BATCH_SIZE}")
print(f"- Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
print(f"- Learning rate: {LR}")
print(f"- Epochs: {EPOCHS}")
print(f"- Random seed: {seed}")

print("\n========================================")
print("2. DATASET LOADER")
print("========================================")
train_dataset = RealAudioDataset(config.METADATA_PATH, 'train')
val_dataset = RealAudioDataset(config.METADATA_PATH, 'validation')

print(f"TRAIN count: {len(train_dataset)}")
print(f"VALIDATION count: {len(val_dataset)}")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print("\n========================================")
print("3. MODEL INITIALIZATION")
print("========================================")
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Enable gradient checkpointing for memory savings
model.gradient_checkpointing_enable()

model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"- Trainable parameter count: {trainable_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda')

print("\n========================================")
print("4. TRAINING LOOP START")
print("========================================")
start_time = time.time()
best_f1 = -1.0

try:
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_train_loss += loss.item() * GRAD_ACCUM_STEPS
            
            # memory clear
            del inputs, labels, outputs
            torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
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
                    loss = outputs.loss
                    
                total_val_loss += loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                del inputs, labels, outputs
                torch.cuda.empty_cache()

        avg_val_loss = total_val_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
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
        
        # Checkpointing
        if f1 > best_f1:
            best_f1 = f1
            save_dir = os.path.join("models", "voice_clone_detector")
            
            # Simple backup mechanism
            if os.path.exists(save_dir) and epoch == 0:
                backup_dir = f"{save_dir}_backup_{int(time.time())}"
                shutil.move(save_dir, backup_dir)
                print(f"Created backup of existing model at: {backup_dir}")
                
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"Saved new best model to {save_dir} with F1: {f1:.4f}")

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n!!! CUDA OUT OF MEMORY ERROR !!!")
        print("Exact Error:")
        print(e)
        print("STOPPING TRAINING IMMEDATELY.")
    else:
        print(f"RUNTIME ERROR: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")

end_time = time.time()
print("\n========================================")
duration_mins = (end_time - start_time) / 60
print(f"Total Training Duration: {duration_mins:.2f} minutes")
print("========================================")
print("COMPLETED")
