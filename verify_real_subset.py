import os
import csv
import soundfile as sf

def verify():
    out_dir = os.path.abspath("data/real_subset")
    metadata_path = os.path.join(out_dir, "metadata.csv")
    
    counts = {
        'train': {'bonafide': 0, 'spoof': 0},
        'validation': {'bonafide': 0, 'spoof': 0}
    }
    unique_orig_files = set()
    unique_speakers = set()
    corrupt_files = 0
    duplicate_files = 0
    
    sample_rates = {}
    durations = []
    total_bytes = 0
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row['split']
            label = row['label_name']
            orig_name = row['original_audio_file_name']
            speaker = row['speaker_id']
            audio_path = os.path.join(out_dir, row['audio_path'])
            
            if orig_name in unique_orig_files:
                duplicate_files += 1
            unique_orig_files.add(orig_name)
            if speaker and speaker != "unknown":
                unique_speakers.add(speaker)
                
            if not os.path.exists(audio_path):
                corrupt_files += 1
                continue
                
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                corrupt_files += 1
                continue
                
            total_bytes += file_size
            
            try:
                info = sf.info(audio_path)
                sr = info.samplerate
                dur = info.duration
                if sr not in sample_rates:
                    sample_rates[sr] = 0
                sample_rates[sr] += 1
                durations.append(dur)
                counts[split][label] += 1
            except Exception:
                corrupt_files += 1

    total = sum(counts['train'].values()) + sum(counts['validation'].values())
    
    avg_dur = sum(durations) / len(durations) if durations else 0
    min_dur = min(durations) if durations else 0
    max_dur = max(durations) if durations else 0
    
    report = []
    def p(text):
        print(text)
        report.append(text)
        
    p("========================================")
    p("REAL ASVSPOOF SUBSET VERIFICATION")
    p("========================================")
    p("")
    p("TRAIN:")
    p(f"  bonafide: {counts['train']['bonafide']}")
    p(f"  spoof: {counts['train']['spoof']}")
    p("")
    p("VALIDATION:")
    p(f"  bonafide: {counts['validation']['bonafide']}")
    p(f"  spoof: {counts['validation']['spoof']}")
    p("")
    p("TOTAL:")
    p(f"  {total}")
    p("")
    p("UNIQUE ORIGINAL FILES:")
    p(f"  {len(unique_orig_files)}")
    p("")
    p("UNIQUE SPEAKERS:")
    p(f"  {len(unique_speakers)}")
    p("")
    p("CORRUPT FILES:")
    p(f"  {corrupt_files}")
    p("")
    p("DUPLICATE FILES:")
    p(f"  {duplicate_files}")
    p("")
    p("SAMPLE RATE:")
    p(f"  {sample_rates}")
    p("")
    p("DURATION:")
    p(f"  minimum: {min_dur:.2f}s")
    p(f"  maximum: {max_dur:.2f}s")
    p(f"  average: {avg_dur:.2f}s")
    p("")
    p("DISK SIZE:")
    p(f"  {total_bytes / (1024*1024):.2f} MB")
    
    with open(os.path.join(out_dir, "..", "REAL_DATASET_REPORT.md"), "w", encoding='utf-8') as f:
        f.write("\n".join(report))

if __name__ == '__main__':
    verify()
