# Stage 6 Independent Test Evaluation Report

## 1. Missing Test Subset Evaluation
A rigorous check across `metadata.csv` covering all physically mapped 1,200 local subsets determined that **NO INDEPENDENT TEST SET IS AVAILABLE**.

- **Available Splits**: `Train` (1000 items), `Validation` (200 items).
- **Missing Resource**: A blind, physically available Real Test Subset conforming to the absolute ASVspoof logic pipeline.

## 2. Hard Stop Conditions Hit
Per strict diagnostic requirements:
- Validation data (`split=='validation'`) was **NOT** repackaged as Test evaluation due to implicit training leakage caused by previous iterative selection conditions.
- No synthetic components were dynamically manufactured to force a classification boundary. 

Validation data was used for checkpoint selection and is therefore not an untouched test set.

## 3. Terminal Statistics Output
The evaluation logic successfully collapsed safely generating:
```
========================================
STAGE 6 INDEPENDENT TEST EVALUATION
========================================
Test samples: N/A
Bonafide: N/A
Spoof: N/A

Train/Test hash overlap: N/A
Validation/Test hash overlap: N/A

Speaker overlap with train: N/A
Speaker overlap with validation: N/A

Accuracy: N/A
Precision: N/A
Recall: N/A
F1: N/A
ROC-AUC: N/A
EER: N/A
FPR: N/A
FNR: N/A
========================================
END STAGE 6
========================================
```
