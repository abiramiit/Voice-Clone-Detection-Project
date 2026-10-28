# Stage 4 Wav2Vec2 Classifier Smoke Test Report

## 1. Environment & CUDA Setup
- **Python version:** 3.12.0 (`.venv_dataset\Scripts\python.exe`)
- **GPU Name:** NVIDIA GeForce RTX 2050
- **CUDA Runtime:** 12.8
- **Model Device:** `cuda:0` (Forward pass occurred on CUDA)

## 2. Model Architecture Profile
- **Base Architecture:** `facebook/wav2vec2-base`
- **Output Head:** `Wav2Vec2ForSequenceClassification`
- **Classes:** 2 (`0 = bonafide`, `1 = spoof`)
- **Total Parameters:** `94,569,090`
- **Trainable Parameters:** `94,569,090`
*(Note: Expected newly initialized weights for the classification head log output was observed as 'MISSING' from the base checkpoint down to downstream linear layer weights as expected.)*

## 3. Real Audio Single Sample Forward Pass 
- **Input Tensor Shape:** `torch.Size([1, 55329])`
- **Actual Metadata Label:** `0` (bonafide)
- **Predicted Class (Untrained):** `0`
- **Logits Shape:** `torch.Size([1, 2])`
- **Raw Logits Value:** `tensor([[-0.0130, -0.0857]])`

## 4. Real Audio 4-Sample Batch Forward Pass
- **Batch Size:** `4` (2 bonafide, 2 spoof)
- **Batch Dataset Labels:** `[0, 0, 1, 1]`
- **Batched Processed Input Shape:** `torch.Size([4, 70323])`
- **Output Batched Logits Shape:** `torch.Size([4, 2])`
- **CUDA Load Verification:** Successfully ran without CUDA 'Out Of Memory' (OOM) errors.

**Conclusion:** The local environment properly loads the architecture and sequentially builds VRAM loads required for feature extraction sequences. A conservative mini-batch of 4 processed natively with 0 errors. Ready to proceed without memory exceptions.
