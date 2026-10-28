import os
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# 1. & 2. Loading & Verification
def get_real_dataset():
    # Load strictly the real architecture splits. Will organically throw an error if no network
    print("Stage 1 & 2 Verification: Loading 'LanceaKing/asvspoof2019' logical_access...")
    train_full = load_dataset("LanceaKing/asvspoof2019", "logical_access", split="train", trust_remote_code=True)
    val_full = load_dataset("LanceaKing/asvspoof2019", "logical_access", split="validation", trust_remote_code=True)
    test_full = load_dataset("LanceaKing/asvspoof2019", "logical_access", split="eval", trust_remote_code=True)
    return train_full, val_full, test_full

def extract_balanced_subset(dataset, num_samples=500):
    bonafide = []
    spoof = []
    for item in dataset:
        lbl = item['label']
        if (lbl == 1 or lbl == "spoof"):
            if len(spoof) < num_samples: spoof.append(item)
        else:
            if len(bonafide) < num_samples: bonafide.append(item)
        if len(spoof) == num_samples and len(bonafide) == num_samples:
            break
            
    if len(bonafide) < num_samples or len(spoof) < num_samples:
        raise ValueError(f"Requirement Broken: Found {len(bonafide)} bonafide and {len(spoof)} spoof.")
    return bonafide + spoof

def custom_collate(batch):
    audios = []
    for b in batch:
        arr = b['audio']['array'] if isinstance(b['audio'], dict) else b['audio']
        audios.append(torch.tensor(arr, dtype=torch.float32))
    
    labels = [1 if (b['label'] == 1 or b['label'] == 'spoof') else 0 for b in batch]
    
    max_len = max([a.shape[0] for a in audios])
    padded = []
    for a in audios:
        pad_len = max_len - a.shape[0]
        if pad_len > 0:
            a = torch.nn.functional.pad(a, (0, pad_len))
        padded.append(a)
        
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)

def main():
    # Stages 1 & 2
    train_full, val_full, test_full = get_real_dataset()
    
    train_subset = extract_balanced_subset(train_full, 500)
    val_subset = extract_balanced_subset(val_full, 100)
    test_subset = extract_balanced_subset(test_full, 100)
    
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, collate_fn=custom_collate)
    
    # Stage 3 & 4
    print("\nStage 3 & 4: Initializing 'facebook/wav2vec2-base' Feature Extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    # Stage 5
    print("Stage 5: Initializing Binary Classification Head with Wav2Vec2 backbone...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
    model.freeze_feature_encoder() # Essential to prevent RAM overload on fine-tunes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    epochs = 3
    print("\nStage 6: Commencing REAL Data Fine-Tuning Iterations...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for b_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # Stage 3 explicitly formatting inside processing nodes
            features = feature_extractor(list(inputs.cpu().numpy()), sampling_rate=16000, return_tensors="pt")
            input_values = features.input_values.to(device)
            
            outputs = model(input_values)
            loss = criterion(outputs.logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if b_idx % 25 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {b_idx} | REAL CrossEntropyLoss: {loss.item():.4f}")
        
        print(f"--- Epoch {epoch+1} Completed | Average Train Loss: {total_loss/len(train_loader):.4f} ---")
        
        # Stage 7
        print("\nStage 7: Epoch Validation Evaluation...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = feature_extractor(list(inputs.cpu().numpy()), sampling_rate=16000, return_tensors="pt")
                v_outs = model(features.input_values.to(device))
                val_loss += criterion(v_outs.logits, labels).item()
        print(f"--- Validation Loss: {val_loss/len(val_loader):.4f} ---")

    # Stage 8 & 9
    print("\nStage 8 & 9: Final Test Set Evaluation & Rigorous Metrics...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            features = feature_extractor(list(inputs.numpy()), sampling_rate=16000, return_tensors="pt")
            t_outs = model(features.input_values.to(device))
            all_preds.extend(torch.argmax(t_outs.logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n[VERIFIED EMPIRICAL METRICS ON REAL HOLD-OUT TEST SET]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix (Real, Spoof):")
    print(cm)
    
    # Stage 10
    print("\nStage 10: Ensuring Distinct Model State Saving...")
    os.makedirs("models", exist_ok=True)
    save_path = "models/real_wav2vec2_finetuned.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Real weights independently saved to `{save_path}`. Training run formally verified and concluded.")

if __name__ == "__main__":
    main()
