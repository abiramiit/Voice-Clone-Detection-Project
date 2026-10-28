import os
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
import numpy as np

class RealAudioDataset(Dataset):
    def __init__(self, metadata_path, split, transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Resolve audio path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        audio_path = os.path.join(base_dir, "data", "real_subset", row['audio_path'])
        
        # 1-2. Read label
        label = int(row['label'])
        
        # 3. Load audio using soundfile
        try:
            waveform, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"FAILED TO READ FILE: {audio_path}")
            raise e
            
        # 4. Verify sample rate
        if sample_rate != 16000:
            print(f"FAILED TO READ FILE - Invalid Sample Rate: {audio_path}")
            raise ValueError(f"Expected 16000 Hz, got {sample_rate} Hz")
            
        # 5. Convert audio to float32
        waveform = waveform.astype(np.float32)
        
        # 6. Convert stereo to mono if necessary
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)
            
        # 7. Preserve actual waveform (already done in float32 mono array)
        
        metadata = {
            'audio_path': audio_path,
            'speaker_id': row['speaker_id'],
            'original_audio_file_name': row['original_audio_file_name'],
            'sample_rate': sample_rate,
            'duration_seconds': row['duration_seconds']
        }
        
        if self.transform:
            waveform = self.transform(waveform)
            
        # 8. Return
        return waveform, label, metadata
