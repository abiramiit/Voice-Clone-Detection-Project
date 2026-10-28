import pandas as pd
import os
import sys

sys.path.append(os.path.abspath('.'))
import config

df = pd.read_csv("data/real_subset/metadata.csv")

unique_splits = df['split'].unique().tolist()
print(f"Unique splits found in metadata.csv: {unique_splits}")

if 'test' not in unique_splits:
    print("\nNO INDEPENDENT TEST SET AVAILABLE.\n")
    print("========================================")
    print("STAGE 6 INDEPENDENT TEST EVALUATION")
    print("========================================")
    print("Test samples: N/A")
    print("Bonafide: N/A")
    print("Spoof: N/A")
    print("\nTrain/Test hash overlap: N/A")
    print("Validation/Test hash overlap: N/A")
    print("\nSpeaker overlap with train: N/A")
    print("Speaker overlap with validation: N/A")
    print("\nAccuracy: N/A")
    print("Precision: N/A")
    print("Recall: N/A")
    print("F1: N/A")
    print("ROC-AUC: N/A")
    print("EER: N/A")
    print("FPR: N/A")
    print("FNR: N/A")
    print("========================================")
    print("END STAGE 6")
    print("========================================")
    sys.exit(0)
else:
    print("Test set exists! Computing overlaps and predictions...")
