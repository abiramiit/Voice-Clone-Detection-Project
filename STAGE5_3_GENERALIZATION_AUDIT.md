# Stage 5.3 Generalization Audit Report

This report documents the deep structural distribution, provenance, and data leakage limits of the local fine-tuned subset to isolate why `Stage 5.2` reported an identical 1.000 F1 score.

## 1. File-Level Train/Validation Overlap
Every single audio representation (`1,200` unique components) in `data/real_subset/` was strictly validated via absolute `SHA-256` hashing calculations.
- **Identical contents between Train & Val**: `0`
- **Ghost/Duplicated Filenames**: `0`
*Observation*: The validation dataset possesses completely separate audio files than those passed heavily during training loops. No explicit file data leakage occurred.

## 2. Speaker Representation
- **Train Speaker Count**: 4
- **Validation Speaker Count**: 1
- **Train/Validation Speaker Overlap**: `0`
*Observation*: The Validation distribution utilized historically isolated speakers. The model achieved 100% boundary performance targeting a completely **unseen** speaker.

## 3. Label Dependencies
Both the training subset and the validation subset were rigorously matched via strict metadata counts preventing bias representation:
- **Train Constraints**: 500 Bonafide vs 500 Spoof
- **Validation Constraints**: 100 Bonafide vs 100 Spoof

## 4. File Path Structural Correlation
No directories, folders, or prefixes implicitly hint at a classification bias:
- **Bonafide Class Prefixes**: `LA_T` (500) and `LA_D` (100)
- **Spoof Class Prefixes**: `LA_T` (500) and `LA_D` (100)
*Observation*: Model generalization is isolated to pure audio inferences rather than structural OS filepath strings.

## 5. Audio Duration Structural Biases
- **Train Bonafide Mean Duration**: `3.25`s
- **Train Spoof Mean Duration**: `2.87`s
- **Validation Bonafide Mean Duration**: `3.97`s
- **Validation Spoof Mean Duration**: `2.86`s
*Observation*: A slight duration bias occurs wherein `Bonafide` labels possess natively longer waveforms (`~0.4s-1s` longer on average) against identically truncated spoof recordings. 

## 6. Prediction Mapping & Calibration Logits
All 200 validation audios were re-calibrated tracking strict classification bounds inside `reports/stage5_3_predictions.csv`:
- **Average Inference Calibration Confidence**: `0.999512`
- **Output Logit Examples**: Extremely uniform saturation. E.g., `Logits=(3.92, -3.39) -> Probs=(0.9995, 0.0006)` across various random sequences.
*Observation*: The model acts dynamically overconfident on subset classifications and logits lack soft recalibration gradients.

## 7. Data Leakage Verification
- **Model Checkpointing Leakage**: YES. The validation test outputs were explicitly used to dictate the selection parameter for `.save_pretrained(best_model)`. Overfitting to the validation space is implicitly likely.

## 8. Generalization Limitations
Do **NOT** assume the model possesses 100% absolute native intelligence. The following boundaries strictly apply to these metrics:
- Absolute metric evaluations were sampled on a remarkably small subset footprint (`200` samples).
- Independent speaker evaluation was successfully met, but **only scaling for 1 unseen speaker profile**.
- No subsequent **Blind Test Dataset** is available to test the Checkpoint weights without the tie-breaker leakage.
- The model exhibits heavily saturated predictions (`0.9995`) indicating it may fail abruptly instead of providing soft confidence boundaries on structurally different ambient domains.
