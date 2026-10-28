import os
import glob
import soundfile as sf
import sys

def validate():
    d = "data/human_eval/bonafide"
    if not os.path.exists(d):
        print("STAGE 21A STATUS: INSUFFICIENT REAL HUMAN DATA")
        return

    files = glob.glob(os.path.join(d, "*.wav"))
    if len(files) == 0:
        print("STAGE 21A STATUS: INSUFFICIENT REAL HUMAN DATA")
        return

    # Extract unique speakers
    # Assuming format: speaker01.wav or speaker01_02.wav
    speakers = set()
    total_duration = 0.0
    valid_files = 0
    corrupt = 0
    short_files = 0
    formats = set()
    sample_rates = set()
    channels = set()

    for f in files:
        base = os.path.basename(f).replace(".wav", "")
        # Get speaker ID: everything before the first underscore if multiple, else the whole string
        speaker_id = base.split("_")[0]
        speakers.add(speaker_id)
        
        try:
            info = sf.info(f)
            total_duration += info.duration
            valid_files += 1
            formats.add(info.format)
            sample_rates.add(info.samplerate)
            channels.add(info.channels)
            if info.duration < 20.0:
                short_files += 1
        except Exception:
            corrupt += 1

    print("Total files:", len(files))
    print("Unique speakers:", len(speakers), f"({list(speakers)})")
    print("Total duration:", round(total_duration, 2), "seconds")
    print("File formats:", list(formats))
    print("Sample rates:", list(sample_rates))
    print("Channels:", list(channels))
    print("Invalid/corrupt files:", corrupt)
    print("Files shorter than 20 seconds:", short_files)

    print("\n---")
    if len(speakers) < 5:
        print("STAGE 21A STATUS: INSUFFICIENT REAL HUMAN DATA")
    else:
        print("STAGE 21A STATUS: RECORDINGS READY FOR STAGE 21")

if __name__ == "__main__":
    validate()
