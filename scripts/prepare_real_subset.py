import os
import csv
import soundfile as sf
import numpy as np
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

TARGETS = {
    'train': {'bonafide': 500, 'spoof': 500},
    'validation': {'bonafide': 100, 'spoof': 100},
    'test': {'bonafide': 100, 'spoof': 100}
}

def prepare_directories():
    base = "data/real_subset"
    for split in TARGETS.keys():
        for label in TARGETS[split].keys():
            os.makedirs(os.path.join(base, split, label), exist_ok=True)
    return base

def get_label_info(item):
    if 'label' in item:
        l = item['label']
        if isinstance(l, int):
            return "spoof" if l == 1 else "bonafide", l
        elif isinstance(l, str):
            return "spoof" if "spoof" in l.lower() else "bonafide", 1 if "spoof" in l.lower() else 0
    return None, None

def main():
    print("--- 1. INITIALIZING DATASET SUBSET EXTRACTION ---")
    base_dir = prepare_directories()
    csv_path = os.path.join(base_dir, "metadata.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "audio_path", "speaker_id", "original_audio_file_name", "label", "label_name"])

    counts = {
        'train': {'bonafide': 0, 'spoof': 0},
        'validation': {'bonafide': 0, 'spoof': 0},
        'test': {'bonafide': 0, 'spoof': 0}
    }
    
    durations = []
    sampling_rates = set()
    metrics = {'total_bytes': 0, 'corrupt_count': 0}
    
    print("Connecting to Bisher/ASVspoof_2019_LA via streaming...")
    dataset = {}
    try:
        dataset = load_dataset("Bisher/ASVspoof_2019_LA", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"[FATAL ERROR] Dataset connection completely failed: {e}")

    def process_split(split_name, dataset_split):
        print(f"\nProcessing mapping for split: {split_name}")
        for item in dataset_split:
            if counts[split_name]['bonafide'] >= TARGETS[split_name]['bonafide'] and counts[split_name]['spoof'] >= TARGETS[split_name]['spoof']:
                break
                
            label_str, label_int = get_label_info(item)
            if not label_str: continue
            
            if counts[split_name][label_str] >= TARGETS[split_name][label_str]:
                continue
                
            audio = item.get('audio')
            if not audio or 'array' not in audio:
                metrics['corrupt_count'] += 1
                continue
                
            arr = np.array(audio['array'])
            sr = audio['sampling_rate']
            
            if len(arr) == 0:
                metrics['corrupt_count'] += 1
                continue
                
            speaker_id = item.get('speaker_id', 'unknown')
            orig_name = item.get('file', f'audio_{counts[split_name][label_str]}')
            
            save_name = f"{speaker_id}_{counts[split_name][label_str]}.wav"
            save_path = os.path.join(base_dir, split_name, label_str, save_name)
            
            try:
                sf.write(save_path, arr, sr)
                size = os.path.getsize(save_path)
                metrics['total_bytes'] += size
                durations.append(len(arr) / sr)
                sampling_rates.add(sr)
                
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([split_name, save_path, speaker_id, orig_name, label_int, label_str])
                    
                counts[split_name][label_str] += 1
                
                if (counts[split_name][label_str]) % 100 == 0:
                    print(f" -> Grabbed {counts[split_name][label_str]} {label_str} samples for {split_name}")
            except Exception as e:
                print(f"Error saving file {save_path}: {e}")
                metrics['corrupt_count'] += 1

    for split_key in ['train', 'validation', 'test']:
        if split_key in dataset:
            process_split(split_key, dataset[split_key])
        elif split_key == 'validation' and 'val' in dataset:
            process_split('validation', dataset['val'])
        elif split_key == 'validation' and 'dev' in dataset:
            process_split('validation', dataset['dev'])
        elif split_key == 'test' and 'eval' in dataset:
            process_split('test', dataset['eval'])

    total_files = sum(counts['train'].values()) + sum(counts['validation'].values()) + sum(counts['test'].values())
    
    print("\n==================================")
    print("# REAL SUBSET VERIFICATION")
    print("==================================")
    print("\nTraining:")
    print(f"bonafide: {counts['train']['bonafide']}")
    print(f"spoof: {counts['train']['spoof']}")
    
    print("\nValidation:")
    print(f"bonafide: {counts['validation']['bonafide']}")
    print(f"spoof: {counts['validation']['spoof']}")
    
    print("\nTest:")
    print(f"bonafide: {counts['test']['bonafide']}")
    print(f"spoof: {counts['test']['spoof']}")
    
    print(f"\nTotal physical audio files: {total_files}")
    
    avg_d = np.mean(durations) if durations else 0
    min_d = np.min(durations) if durations else 0
    max_d = np.max(durations) if durations else 0
    mb_size = metrics['total_bytes'] / (1024 * 1024)
    
    print("\n--- STATISTICS ---")
    print(f"Total disk size: {mb_size:.2f} MB")
    print(f"Sample-rate distribution: {list(sampling_rates)}")
    print(f"Duration -> Min: {min_d:.2f}s | Max: {max_d:.2f}s | Avg: {avg_d:.2f}s")
    print(f"Corrupt file count: {metrics['corrupt_count']}")
    
    report_md = f"""# REAL ASVSPOOF 2019 LA SUBSET
    
* **Dataset Config:** `Bisher/ASVspoof_2019_LA` (Streamed physically bypassing ZIP blocks)
* **Total Extraction Files:** {total_files}

### Training
* Bonafide: {counts['train']['bonafide']}
* Spoof: {counts['train']['spoof']}

### Validation
* Bonafide: {counts['validation']['bonafide']}
* Spoof: {counts['validation']['spoof']}

### Test
* Bonafide: {counts['test']['bonafide']}
* Spoof: {counts['test']['spoof']}

### Array Metrics
* **Total Disk Threshold:** {mb_size:.2f} MB
* **Sample Rates Found:** {list(sampling_rates)}
* **Durations (Seconds):** Min {min_d:.2f}s | Max {max_d:.2f}s | Avg {avg_d:.2f}s
* **Corrupted Formats Rejected:** {metrics['corrupt_count']}
"""

    with open("REAL_DATASET_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    for split in ['train', 'validation', 'test']:
        for label in ['bonafide', 'spoof']:
            if counts[split][label] < TARGETS[split][label]:
                print(f"\n[HARD FAILURE] Target threshold not met for {split} {label}. Got {counts[split][label]} / {TARGETS[split][label]}.")
                print("STOP. Do not train.")
                return

    print("\nSuccess. Entire payload precisely mapped physically.")

if __name__ == "__main__":
    main()
