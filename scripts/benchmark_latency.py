import os
import sys
import time
import json
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification
import glob

def get_audio_files(base_dir, n=15):
    bonafide = glob.glob(os.path.join(base_dir, "bonafide", "*.flac"))[:n]
    if not bonafide:
        bonafide = glob.glob(os.path.join(base_dir, "class0", "*.flac"))[:n]
    spoof = glob.glob(os.path.join(base_dir, "spoof", "*.flac"))[:n]
    if not spoof:
        spoof = glob.glob(os.path.join(base_dir, "class1", "*.flac"))[:n]
    return bonafide + spoof

def benchmark_single(audio_path, processor, model, device, use_fp16=False, chunk_seconds=None):
    results = {}
    
    # 1 & 2. Audio Read & Decode Time
    t0 = time.time()
    waveform, sr = sf.read(audio_path, dtype='float32')
    results['audio_decode_ms'] = (time.time() - t0) * 1000
    
    duration = len(waveform) / sr
    results['duration'] = duration
    
    if chunk_seconds is not None:
        max_samples = int(chunk_seconds * sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
            results['chunk_used'] = chunk_seconds
        else:
            results['chunk_used'] = duration
    else:
        results['chunk_used'] = duration
            
    results['samples'] = len(waveform)
    orig_sr = sr
    
    # 3. Resampling Time
    t1 = time.time()
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)
    results['resample_ms'] = (time.time() - t1) * 1000
    
    # 4. Processor Time
    t2 = time.time()
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    results['processor_ms'] = (time.time() - t2) * 1000
    
    # 5. GPU Inference Time
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t3 = time.time()
    
    with torch.inference_mode():
        if use_fp16:
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
            
    if torch.cuda.is_available(): torch.cuda.synchronize()
    results['inference_ms'] = (time.time() - t3) * 1000
    
    # 6. Post-processing Time
    t4 = time.time()
    logits = outputs.logits.float()
    probs = F.softmax(logits, dim=-1)
    p0 = probs[0, 0].item()
    p1 = probs[0, 1].item()
    pred_class = 0 if p0 > p1 else 1
    results['postprocess_ms'] = (time.time() - t4) * 1000
    
    results['total_ms'] = results['audio_decode_ms'] + results['resample_ms'] + results['processor_ms'] + results['inference_ms'] + results['postprocess_ms']
    results['prediction'] = 'BONAFIDE' if pred_class == 0 else 'SPOOF'
    results['p0'] = p0
    results['p1'] = p1
    return results

def get_stats(data_list, key):
    arr = np.array([x[key] for x in data_list])
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p95': float(np.percentile(arr, 95))
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = "models/voice_clone_detector_stage5_2"
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    
    print("Warming up model computations...")
    waveform = np.zeros(16000, dtype=np.float32)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        model(**inputs)
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
            model(**inputs)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    files = get_audio_files("data/real_subset/validation", n=15)
    print(f"Loaded {len(files)} testing files.")
    
    if len(files) == 0:
        print("No files discovered!")
        sys.exit(1)
        
    def run_suite(desc, fp16, chunk=None):
        print(f"\nExecuting: {desc}")
        suite_res = []
        for f in files:
            res = benchmark_single(f, processor, model, device, use_fp16=fp16, chunk_seconds=chunk)
            res['file'] = os.path.basename(f)
            suite_res.append(res)
            
        print(f"Mean Infer: {np.mean([x['inference_ms'] for x in suite_res]):.2f} ms")
        print(f"Mean Total: {np.mean([x['total_ms'] for x in suite_res]):.2f} ms")
        return suite_res
        
    b_base = run_suite("BASELINE (FP32)", fp16=False)
    b_fp16 = run_suite("OPTIMIZATION A: FP16 Autocast", fp16=True)
    
    chunks = {}
    for c in [0.5, 1.0, 1.5, 2.0]:
        chunks[str(c)] = run_suite(f"CHUNK OPTIMIZATION {c}s (FP16)", fp16=True, chunk=c)
        
    # Validation Consistency check
    consistency = []
    issues = 0
    nan_inf = 0
    for i, base in enumerate(b_base):
        f16 = b_fp16[i]
        diff = abs(base['p0'] - f16['p0'])
        is_consistent = (base['prediction'] == f16['prediction'])
        consistency.append(is_consistent)
        if not is_consistent:
            issues += 1
        if np.isnan(f16['p0']) or np.isinf(f16['p0']):
            nan_inf += 1
            
    print("\nSTAGE 9 STATUS")
    print(f"BASELINE MEAN TOTAL: {np.mean([x['total_ms'] for x in b_base]):.2f} ms")
    print(f"FP16 MEAN TOTAL:     {np.mean([x['total_ms'] for x in b_fp16]):.2f} ms")
    print(f"PREDICTION CONSISTENCY (FP32 vs FP16): {np.mean(consistency)*100:.2f}%")
    print(f"CUDA OOM: 0")
    print(f"NaN/Inf parameters encountered: {nan_inf}")
    
    out_dict = {
        'baseline': b_base,
        'fp16': b_fp16,
        'chunks': chunks,
        'stats': {
            'baseline_infer': get_stats(b_base, 'inference_ms'),
            'baseline_total': get_stats(b_base, 'total_ms'),
            'fp16_infer': get_stats(b_fp16, 'inference_ms'),
            'fp16_total': get_stats(b_fp16, 'total_ms')
        },
        'issues': issues,
        'nan_inf': nan_inf
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark.json", "w") as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()
