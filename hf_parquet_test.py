import os
import sys
import tempfile
import pyarrow.parquet as pq
import wave
try:
    from huggingface_hub import HfFileSystem
except ImportError:
    print("huggingface_hub not found. Please install it.")
    sys.exit(1)

def run_test():
    repo_id = "datasets/Bisher/ASVspoof_2019_LA"
    fs = HfFileSystem()

    print(f"Inspecting repository: {repo_id}")
    try:
        parquet_files = [f for f in fs.ls(repo_id, detail=True) if f['name'].endswith('.parquet')]
        if not parquet_files:
            # Maybe inside a data/ subdir
            parquet_files = [f for f in fs.ls(f"{repo_id}/data", detail=True) if f['name'].endswith('.parquet')]
    except Exception as e:
        print(f"Error accessing repo: {e}")
        return

    train_files = [f for f in parquet_files if 'train' in f['name'].lower()]
    print(f"Total Parquet files: {len(parquet_files)}")
    print(f"Train Parquet files: {len(train_files)}")
    print("Train shard locations:")
    for f in train_files:
        print(f" - {f['name']} (Size: {f['size'] / (1024*1024):.2f} MB)")

    if not train_files:
        print("Could not find train parquet files.")
        return

    train_shard = train_files[0]
    print(f"\nTargeting first Train Shard: {train_shard['name']} for feasibility test.")
    
    total_downloaded = 0
    actual_audio_retrieved = False
    bonafide_count = 0
    spoof_count = 0
    rows_inspected = 0
    torchcodec_invoked = False
    
    try:
        print("Opening Parquet file over HTTP...")
        with fs.open(train_shard['name'], "rb") as f:
            pf = pq.ParquetFile(f)
            print(f"Parquet metadata loaded.")
            print(f"Number of Row Groups: {pf.num_row_groups}")
            print(f"Total Rows in file: {pf.metadata.num_rows}")
            
            # Read first row group
            print("\nReading first row group (PyArrow)...")
            table = pf.read_row_group(0)
            df = table.to_pandas()
            rows_inspected = len(df)
            print(f"Successfully read {rows_inspected} rows into pandas dataframe.")
            
            label_col = 'label' if 'label' in df.columns else None
            if not label_col:
                for col in df.columns:
                    if 'label' in str(col).lower():
                        label_col = col
            
            if label_col:
                val_counts = df[label_col].value_counts().to_dict()
                print(f"Labels in sampled data: {val_counts}")
                if 'bonafide' in df[label_col].unique() or 0 in df[label_col].unique() or '0' in df[label_col].unique():
                    bonafide_count = int(val_counts.get('bonafide', val_counts.get(0, val_counts.get('0', 0))))
                if 'spoof' in df[label_col].unique() or 1 in df[label_col].unique() or '1' in df[label_col].unique():
                    spoof_count = int(val_counts.get('spoof', val_counts.get(1, val_counts.get('1', 0))))

            audio_col = 'audio'
            if audio_col not in df.columns:
                print("Could not find 'audio' column.")
                return

            print("\nTesting audio data extraction...")
            out_dir = os.path.abspath("data/parquet_test")
            os.makedirs(out_dir, exist_ok=True)
            
            extracted_samples = 0
            for i in range(min(5, len(df))):
                audio_data = df[audio_col].iloc[i]
                audio_bytes = None
                
                if isinstance(audio_data, dict) and 'bytes' in audio_data:
                    audio_bytes = audio_data['bytes']
                elif hasattr(audio_data, 'get'):
                    audio_bytes = audio_data.get('bytes')
                
                # Check for PyArrow struct mapping handling
                if not audio_bytes and type(audio_data).__name__ == "dict":
                     if b'bytes' in audio_data: 
                        audio_bytes = audio_data[b'bytes']

                if audio_bytes:
                     actual_audio_retrieved = True
                     out_path = os.path.join(out_dir, f"test_audio_{i}.wav")
                     total_downloaded += len(audio_bytes)
                     with open(out_path, "wb") as out_f:
                         out_f.write(audio_bytes)
                     print(f"Saved {out_path} ({len(audio_bytes)} bytes)")
                     try:
                         with wave.open(out_path, "rb") as wf:
                             sr = wf.getframerate()
                             dur = wf.getnframes() / float(sr)
                             print(f"  -> Decode OK: File {i} | SR: {sr} | Duration: {dur:.2f}s")
                             extracted_samples += 1
                     except Exception as w_e:
                         print(f"  -> Decode FAILED with wave module: {w_e}")
                else:
                     print(f"Could not extract raw bytes for row {i}")
            
            print("\nSUMMARY:")
            print(f"Rows Inspected: {rows_inspected}")
            print(f"Bonafide Rows: {bonafide_count}")
            print(f"Spoof Rows: {spoof_count}")
            print(f"Actual Audio Retrieved: {actual_audio_retrieved}")
            print(f"TorchCodec Invoked: {torchcodec_invoked}")
            print(f"Bytes written locally for test samples: {total_downloaded}")
            
    except Exception as e:
        print(f"Error accessing Parquet file remotely: {e}")

if __name__ == '__main__':
    run_test()
