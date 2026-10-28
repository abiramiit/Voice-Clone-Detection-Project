import os
import csv
import soundfile as sf

def main():
    print("\n--- 2. VERIFYING EXTRACTED SUBSET INTEGRITY ---")
    base_dir = "data/real_subset"
    csv_path = os.path.join(base_dir, "metadata.csv")
    
    if not os.path.exists(csv_path):
        print(f"[FATAL] Metadata file missing at {csv_path}")
        return
        
    verified = 0
    failed = 0
    missing = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["audio_path"]
            if not os.path.exists(path):
                missing += 1
                continue
                
            try:
                data, sr = sf.read(path)
                if len(data) == 0:
                    failed += 1
                else:
                    verified += 1
            except Exception as e:
                failed += 1
                
    print(f"Total Physical Reads Evaluated.")
    print(f"* Verifiably Passed: {verified}")
    print(f"* Failed/Corrupt:  {failed}")
    print(f"* Missing completely:{missing}")
    
    if failed > 0 or missing > 0:
        print("[HARD FAILURE] Evaluation verified corrupted extraction integrity bounds.")
    elif verified == 1400:
        print("[SUCCESS] All 1400 items physically opened, decoded, verified valid dynamically natively.")
    else:
        print(f"[WARNING] Read count is {verified}, expected exactly 1400 target boundaries natively defined.")

if __name__ == "__main__":
    main()
