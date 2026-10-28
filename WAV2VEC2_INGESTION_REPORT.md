# Wav2Vec2 Ingestion Validation Report (GPU Environment)

## 1. Environment Verification
**Python Interpreter:** `E:\resume-project1\.venv_dataset\Scripts\python.exe`
- **Python version**: 3.12.0
- **torch**: 2.11.0+cu128
- **transformers**: 5.17.0
- **soundfile**: 0.14.0
- **pandas**: 3.0.5
- **numpy**: 2.5.3
- **scikit-learn**: 1.9.1

## 2. GPU & CUDA Details
- **CUDA available**: True
- **CUDA runtime**: 12.8
- **GPU name**: NVIDIA GeForce RTX 2050

## 3. Processor Loading
- Model: `facebook/wav2vec2-base`
- Initialization successful via `AutoProcessor`.

## 4. Single Real Sample Validation
- File Processed: `LA_T_1138215.flac`
- Label: `0` (bonafide)
- Sample Rate: `16000`
- Waveform Shape: `(55329,)`
- Duration: `3.4580625s`
- Processed Input Tensor Shape: `torch.Size([1, 55329])`
- Tensor Data Type: `torch.float32`

## 5. Batch Processing Validation
- Batch Size: `4` (2 bonafide, 2 spoof)
- Input Tensor Shape: `torch.Size([4, 70323])`
- Labels Output: `[0, 0, 1, 1]`
- Attention Mask Shape: `torch.Size([4, 70323])`
- Tensor Data Type: `torch.float32`
- *Observation*: Sequences correctly padded dynamically to the longest shape within the batch (`70323`).

## 6. File Counts & Splits
- **Train Split**:
  - Bonafide = `500`
  - Spoof = `500`
- **Validation Split**:
  - Bonafide = `100`
  - Spoof = `100`

## 7. Speaker Data Leakage Analysis
- Duplicate original filenames across dataset: `0`
- Duplicate audio paths: `0`
- Unique Training Speakers: `4`
- Unique Validation Speakers: `1`
- **Speaker Overlap (Train/Val)**: `0` (Zero Leakage)

**Conclusion**: Ingestion and environment validation successfully completed within the accurate GPU environment. Ready to proceed with next stages.
