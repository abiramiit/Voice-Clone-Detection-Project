from datasets import load_dataset

dataset = load_dataset(
    "LanceaKing/asvspoof2019",
    "LA",
    split="train",
    trust_remote_code=True
)

print("DATASET:")
print(dataset)

print("\nNUMBER OF SAMPLES:")
print(len(dataset))

print("\nFEATURES:")
print(dataset.features)

print("\nFIRST SAMPLE:")
sample = dataset[0]
print(sample)

print("\nAUDIO INFORMATION:")
audio = sample["audio"]
print("Audio type:", type(audio))

if isinstance(audio, dict):
    print("Audio keys:", audio.keys())
    print("Sampling rate:", audio.get("sampling_rate"))
    print("Array type:", type(audio.get("array")))
    print("Array length:", len(audio.get("array", [])))

print("\nLABEL INFORMATION:")
labels = dataset["label"]
print("First 20 labels:", labels[:20])

from collections import Counter
print("Label counts:", Counter(labels))

print("\nCACHE FILES:")
print(dataset.cache_files)
