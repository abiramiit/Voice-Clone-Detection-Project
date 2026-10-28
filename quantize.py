import os
from optimum.onnxruntime import ORTModelForSequenceClassification
import onnxruntime.quantization as ortq

def export_and_quantize():
    print("Exporting PyTorch model to ONNX using Optimum...")
    model_id = "hf-internal-testing/tiny-random-wav2vec2"
    # Export directly to ONNX using Optimum's integration
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained("models/onnx")

    print("Quantizing ONNX model to INT8 (dynamic)...")
    ortq.quantize_dynamic(
        "models/onnx/model.onnx",
        "models/model_quantized.onnx",
        weight_type=ortq.QuantType.QInt8
    )
    print("Quantization complete! Saved to models/model_quantized.onnx")

if __name__ == "__main__":
    export_and_quantize()
