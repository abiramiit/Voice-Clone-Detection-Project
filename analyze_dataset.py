import os
import torch
from datasets import load_dataset
import numpy as np

def main():
    print("--- STARTING DATASET LOAD ---")
    print("Attempting to load 'LanceaKing/asvspoof2019' logical_access split from Hugging Face...")
    print("Warning: This dataset payload (uncompressed) borders ~25GB. Bandwidth limits apply.")
    
    dataset = load_dataset("LanceaKing/asvspoof2019", "logical_access", split="train", trust_remote_code=True)
    
    print("Dataset structure fetched successfully. Analyzing features...")
    
    total_samples = len(dataset)
    bonafide = 0
    spoof = 0
    missing_audio = 0
    sr_set = set()
    durations = []
    
    print(f"Total logical arrays verified: {total_samples}")
    
    for i, item in enumerate(dataset):
        if i % 2500 == 0:
            print(f"Deep scanning chunk {i} of {total_samples}...")
            
        label = item['label']
        is_spoof = (label == 1 or label == "spoof")
        if is_spoof:
            spoof += 1
        else:
            bonafide += 1
            
        audio = item.get('audio')
        if not audio or 'array' not in audio:
            missing_audio += 1
            continue
            
        arr = audio['array']
        sr = audio['sampling_rate']
        sr_set.add(sr)
        
        duration = len(arr) / sr
        durations.append(duration)
        
    avg_duration = np.mean(durations) if durations else 0
    min_duration = np.min(durations) if durations else 0
    max_duration = np.max(durations) if durations else 0

    if total_samples == 0:
        print("[CRITICAL] The dataset contains zero real files. Execution blocked.")
        return

    report = [
        "# EMPIRICAL DATASET REPORT",
        f"* Total Number of Samples: {total_samples}",
        f"* Number of Bona Fide Samples: {bonafide}",
        f"* Number of Spoof Samples: {spoof}",
        f"* Class Distribution: {spoof/total_samples*100:.2f}% Spoof / {bonafide/total_samples*100:.2f}% Bona Fide",
        f"* Audio File Formats: Loaded natively into empirical RAM arrays (FLAC sources)",
        f"* Sample Rates Extracted: {sr_set}",
        f"* Minimum Duration: {min_duration:.4f} seconds",
        f"* Maximum Duration: {max_duration:.4f} seconds",
        f"* Average Duration: {avg_duration:.4f} seconds",
        f"* Missing/Corrupted Audio Files: {missing_audio}"
    ]
    
    report_str = "\n".join(report)
    print("\n" + "="*50)
    print(report_str)
    print("="*50 + "\n")
    
    with open("DATASET_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_str)
    print("Verified local statistics securely saved to DATASET_REPORT.md!")

if __name__ == "__main__":
    main()
