import onnxruntime as ort
import torch
import time
import os
from transformers import Wav2Vec2ForSequenceClassification

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def benchmark():
    model_name = "hf-internal-testing/tiny-random-wav2vec2"
    num_runs = 50
    input_shape = (1, 32000) # 2 seconds of 16kHz audio
    
    dummy_input_pt = torch.randn(input_shape)
    dummy_input_np = dummy_input_pt.numpy()
    
    print("1. Benchmarking original PyTorch model...")
    pt_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
    pt_model.load_state_dict(torch.load("models/wav2vec2_baseline.pt", map_location="cpu"))
    pt_model.eval()
    
    # Warmup
    for _ in range(5):
        _ = pt_model(dummy_input_pt)
    
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = pt_model(dummy_input_pt)
    pt_latency = ((time.time() - start_time) / num_runs) * 1000
    pt_size = get_file_size_mb("models/wav2vec2_baseline.pt")
    
    print("2. Benchmarking ONNX Quantized model...")
    ort_session = ort.InferenceSession("models/model_quantized.onnx", providers=['CPUExecutionProvider'])
    
    # Warmup
    for _ in range(5):
        _ = ort_session.run(None, {"input_values": dummy_input_np})
        
    start_time = time.time()
    for _ in range(num_runs):
        _ = ort_session.run(None, {"input_values": dummy_input_np})
    ort_latency = ((time.time() - start_time) / num_runs) * 1000
    ort_size = get_file_size_mb("models/model_quantized.onnx")
    
    print("\n--- PHASE 2: BENCHMARK RESULTS ---")
    print(f"PyTorch Original  -> Size: {pt_size:.2f} MB | Latency: {pt_latency:.2f} ms")
    print(f"ONNX INT8 Quant   -> Size: {ort_size:.2f} MB | Latency: {ort_latency:.2f} ms")
    
    if ort_latency < 100:
        print("TARGET MET: Latency is under 100ms threshold.")
    else:
        print("TARGET MISSED: Latency exceeds 100ms threshold.")

if __name__ == "__main__":
    benchmark()
