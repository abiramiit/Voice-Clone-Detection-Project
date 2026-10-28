import os
import json
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import warnings

warnings.filterwarnings("ignore")

def save_architecture():
    print("--- 1. INITIALIZING BASELINE ARCHITECTURE FOR SAVING ---")
    save_dir = "models/voice_clone_detector"
    os.makedirs(save_dir, exist_ok=True)
    
    # Strictly initializing structural weights
    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    # Hard-coding empirical label mappings
    model.config.id2label = {0: "Bona Fide", 1: "Spoof"}
    model.config.label2id = {"Bona Fide": 0, "Spoof": 1}
    
    train_config = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "epochs_intended": 3,
        "optimizer": "AdamW",
        "dataset_target": "LanceaKing/asvspoof2019",
        "seed_constraints": 42
    }
    
    print(f"--- 2. SAVING COMPONENTS TO {save_dir} ---")
    
    # Saving Weights and Model Configuration 
    model.save_pretrained(save_dir)
    print("[*] Saved structured model weights and configuration bounds natively.")
    
    # Saving Processor Configuration
    feature_extractor.save_pretrained(save_dir)
    print("[*] Saved structural sequence Processor/Feature Extractor matrices.")
    
    # Saving Training Configurations
    with open(os.path.join(save_dir, "training_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    print("[*] Saved dynamic local training configuration JSON states.")
    
    print("\n[SUCCESS] Model saving requirements rigorously completed.")

if __name__ == "__main__":
    save_architecture()
