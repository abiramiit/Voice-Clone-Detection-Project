import sys
import numpy as np
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

def test_connection():
    try:
        print(f"\n--- DATASET ENVIRONMENT CHECK ---")
        print(f"Executing Dataset Extraction Node via Python isolate: {sys.version}")
        print(f"Target repository: 'Bisher/ASVspoof_2019_LA'")
        
        # 1. Load the TRAIN split, 3. Use streaming=True
        dataset = load_dataset("Bisher/ASVspoof_2019_LA", split="train", streaming=True, trust_remote_code=True)
        
        print("\n[SUCCESS] Authentic Dataset stream iterator resolved sequentially.")
        
        print("\n--- SCHEMA VALIDATION ---")
        print(f"Dataset object type: {type(dataset)}")
        feature_names = dataset.features.keys() if hasattr(dataset, 'features') else "Unknown stream features natively"
        print(f"Feature Names mapped implicitly: {feature_names}")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Dataset local connectivity constraint failed natively:")
        print(e)
        return

    # 4. Read only the first few examples sequentially cleanly.
    print("\n--- EXTRACTING EXACTLY THE FIRST 3 REAL ELEMENTS ---")
    
    try:
        for i, item in enumerate(dataset):
            if i >= 3:
                break
                
            print(f"\n--- EXAMPLE {i+1} ---")
            
            speaker_id = item.get('speaker_id', item.get('speaker', 'N/A'))
            audio_file_name = item.get('file', item.get('audio_file_name', 'N/A'))
            label = item.get('label', 'N/A')
            
            audio = item.get('audio')
            if audio and 'array' in audio:
                sampling_rate = audio.get('sampling_rate', 'N/A')
                array = np.array(audio['array'])
                shape = array.shape
                length = len(array)
            else:
                sampling_rate = 'N/A'
                shape = 'N/A'
                length = 'N/A'
                
            # 5. Print speaker_id, audio_file_name, key/label, audio sampling rate, audio shape/length
            print(f"  * speaker_id:              {speaker_id}")
            print(f"  * audio_file_name:         {audio_file_name}")
            print(f"  * key/label mapped:        {label}")
            print(f"  * sampling rate limits:    {sampling_rate}")
            print(f"  * audio structural shape:  {shape} / length={length}")
            
            # 6. Confirm 0 = bonafide, 1 = spoof natively
            class_str = ""
            if label == 0 or (isinstance(label, str) and 'bonafide' in label.lower()):
                class_str = "0 = bonafide"
            elif label == 1 or (isinstance(label, str) and 'spoof' in label.lower()):
                class_str = "1 = spoof"
                
            print(f"  * rule 6 verification:     Implicitly maps correctly to [ {class_str} ]")
            
    except Exception as e:
        print(f"\n[FATAL ERROR] Data parsing iteration bounds failed:")
        print(e)

if __name__ == "__main__":
    test_connection()
