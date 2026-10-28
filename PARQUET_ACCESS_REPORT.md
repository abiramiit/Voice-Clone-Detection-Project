# ASVspoof 2019 LA Parquet Access Feasibility Test

## Overview
This report details the feasibility test for accessing the `Bisher/ASVspoof_2019_LA` dataset via Hugging Face's Parquet files, without utilizing the `datasets` module audio decoders, `TorchCodec`, or `FFmpeg`.

## Repository Information
- **Repository**: `datasets/Bisher/ASVspoof_2019_LA`
- **Total Parquet Files**: 3
- **Train Parquet Files**: 1

## Train Shard Information
- **Shard Location**: `datasets/Bisher/ASVspoof_2019_LA/data/train-00000-of-00001.parquet`
- **Shard Size**: 1.70 GB
- **Access Method**: HTTP Streaming using `huggingface_hub.HfFileSystem` and `pyarrow.parquet` `read_row_group()`.

## Download and Extraction Metrics
- **Row Groups**: 254
- **Total Rows (Train)**: 25,380
- **Rows Inspected**: 100 (Extracted by reading row group 0)
- **Bonafide Rows Found**: 0 (in the first row group)
- **Spoof Rows Found**: 100 (in the first row group)
- **Actual Audio Bytes Successfully Retrieved**: **Yes**
- **TorchCodec Invoked**: **No**
- **Bytes Downloaded (for local tests)**: ~310 KB (subset of ~6MB row group)

## Audio Verification
The audio bytes were successfully extracted locally into `data/parquet_test/`. The default `wave` module failed to parse the headers because the bytes are **FLAC** encoded, not standard WAV.
Using `soundfile` (which natively links to `libsndfile`), the files were perfectly decoded.

**Sample Metrics (data/parquet_test/):**
- `test_audio_0.wav`: **16000 Hz, 3.19s** (63,429 bytes)
- `test_audio_1.wav`: **16000 Hz, 2.38s** (46,470 bytes)
- `test_audio_2.wav`: **16000 Hz, 3.53s** (71,114 bytes)
- `test_audio_3.wav`: **16000 Hz, 2.14s** (63,854 bytes)
- `test_audio_4.wav`: **16000 Hz, 3.01s** (65,268 bytes)

## Conclusion
The feasibility test is **SUCCESSFUL**. The entire 7.54 GB dataset and the 7.12 GB zip file do not need to be downloaded. Python can stream exact row chunks (e.g., 100-row groups) via HTTP to access the `audio` -> `bytes` column directly into memory, which contains raw FLAC files readable by `soundfile`.
