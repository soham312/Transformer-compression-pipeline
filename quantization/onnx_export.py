import os
import time
import json
import warnings
import torch
import torch.nn as nn
import numpy as np
import safetensors.torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, hamming_loss
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import load_and_tokenize_data
from distillation.student_model import StudentTransformer

def compute_metrics(logits, labels):
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)
    labels = np.array(labels)
    
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    
    ham_loss = hamming_loss(labels, preds)
    hamming_acc = 1.0 - ham_loss
    
    return {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1), "hamming_acc": float(hamming_acc)}

def get_dir_size(path):
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def benchmark_pytorch_model(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    latencies = []
    
    with torch.no_grad():
        # Warmup
        for i, batch in enumerate(loader):
            if i >= 10: break
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            
        # Benchmark
        for batch in loader:
            b_input_ids = batch["input_ids"]
            b_attn_mask = batch["attention_mask"]
            b_labels = batch["labels"]
            
            start_time = time.time()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            latencies.append(time.time() - start_time)
            
            all_logits.append(outputs.numpy())
            all_labels.append(b_labels.numpy())
            
    metrics = compute_metrics(np.vstack(all_logits), np.vstack(all_labels))
    latencies_ms = [(lat / loader.batch_size) * 1000 for lat in latencies]
    metrics["latency_mean_ms"] = float(np.mean(latencies_ms))
    metrics["latency_median_ms"] = float(np.median(latencies_ms))
    return metrics

def benchmark_onnx_model(session, loader):
    all_logits = []
    all_labels = []
    latencies = []
    
    input_name_ids = session.get_inputs()[0].name
    input_name_mask = session.get_inputs()[1].name
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= 10: break
        session.run(None, {
            input_name_ids: batch["input_ids"].numpy(),
            input_name_mask: batch["attention_mask"].numpy()
        })
        
    # Benchmark
    for batch in loader:
        b_input_ids = batch["input_ids"].numpy()
        b_attn_mask = batch["attention_mask"].numpy()
        b_labels = batch["labels"].numpy()
        
        start_time = time.time()
        outputs = session.run(None, {
            input_name_ids: b_input_ids,
            input_name_mask: b_attn_mask
        })
        latencies.append(time.time() - start_time)
        
        all_logits.append(outputs[0])
        all_labels.append(b_labels)
        
    metrics = compute_metrics(np.vstack(all_logits), np.vstack(all_labels))
    latencies_ms = [(lat / loader.batch_size) * 1000 for lat in latencies]
    metrics["latency_mean_ms"] = float(np.mean(latencies_ms))
    metrics["latency_median_ms"] = float(np.median(latencies_ms))
    return metrics

def load_student_config(model_dir):
    with open(os.path.join(model_dir, "student_config.json"), "r") as f:
        return json.load(f)

def run_export_and_benchmark():
    batch_size = 32
    max_length = 64
    base_dir = "model_checkpoints/student_distilled"
    onnx_dir = "model_checkpoints/student_onnx"
    os.makedirs(onnx_dir, exist_ok=True)
    
    print("Loading test dataset...")
    test_ds, _ = load_and_tokenize_data(model_name="bert-base-uncased", max_length=max_length, split="test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Set engine for PyTorch models
    if torch.backends.quantized.engine == 'none':
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
            
    print("1. Preparing Unquantized PyTorch baseline...")
    config = load_student_config(base_dir)
    pt_base = StudentTransformer(**config)
    pt_base.load_state_dict(safetensors.torch.load_file(os.path.join(base_dir, "model.safetensors")))
    pt_base.eval()
    base_size = get_dir_size(base_dir)
    
    print("2. Preparing Dynamic-quantized PyTorch student...")
    pt_dyn = torch.quantization.quantize_dynamic(pt_base, {nn.Linear}, dtype=torch.qint8)
    # PyTorch bug workaround
    def dummy_hook(*args): pass
    for layer in pt_dyn.encoder.layers:
        layer.register_forward_hook(dummy_hook)
    
    # Save to disk to measure true size
    dyn_dir = "model_checkpoints/student_dynamic_quant"
    dyn_size = get_dir_size(dyn_dir)
    
    print("3. Preparing Static-quantized PyTorch student...")
    pt_stat = StudentTransformer(**config)
    wrapper = torch.ao.quantization.QuantWrapper(pt_stat.classifier)
    pt_stat.classifier = wrapper
    pt_stat.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
    pt_stat.encoder.qconfig = None
    pt_stat.word_embeddings.qconfig = None
    pt_stat.position_embeddings.qconfig = None
    pt_stat.LayerNorm.qconfig = None
    torch.ao.quantization.prepare(pt_stat, inplace=True)
    torch.ao.quantization.convert(pt_stat, inplace=True)
    pt_stat.load_state_dict(torch.load("model_checkpoints/student_static_quant/quantized_model.pt"))
    pt_stat.eval()
    stat_size = get_dir_size("model_checkpoints/student_static_quant")
    
    print("4. Exporting to ONNX (Float)...")
    onnx_float_path = os.path.join(onnx_dir, "model_float.onnx")
    dummy_input_ids = torch.randint(0, 1000, (32, 64), dtype=torch.long)
    dummy_attention_mask = torch.ones((32, 64), dtype=torch.long)
    
    torch.onnx.export(
        pt_base,
        (dummy_input_ids, dummy_attention_mask),
        onnx_float_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    onnx_float_size = get_dir_size(onnx_float_path)
    
    print("5. Applying ONNX Runtime dynamic quantization...")
    onnx_int8_path = os.path.join(onnx_dir, "model_int8.onnx")
    quantize_dynamic(
        model_input=onnx_float_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8
    )
    onnx_int8_size = get_dir_size(onnx_int8_path)
    
    print("Verifying ONNX Float matches PyTorch...")
    ort_float_session = ort.InferenceSession(onnx_float_path, providers=['CPUExecutionProvider'])
    ort_int8_session = ort.InferenceSession(onnx_int8_path, providers=['CPUExecutionProvider'])
    
    pt_out = pt_base(dummy_input_ids, dummy_attention_mask).detach().numpy()
    ort_out = ort_float_session.run(None, {
        'input_ids': dummy_input_ids.numpy(),
        'attention_mask': dummy_attention_mask.numpy()
    })[0]
    
    assert np.allclose(pt_out, ort_out, atol=1e-5), "ONNX output does not match PyTorch output!"
    
    print("--- BENCHMARKING (CPU) ---")
    results = {}
    
    print("Benchmarking PyTorch Float...")
    res = benchmark_pytorch_model(pt_base, test_loader)
    res["size_mb"] = base_size
    results["pytorch_float"] = res
    print(f"  PyTorch Float: {res['latency_median_ms']:.2f} ms | Size: {base_size:.2f} MB | F1: {res['macro_f1']:.4f}")
    
    print("Benchmarking PyTorch Dynamic...")
    res = benchmark_pytorch_model(pt_dyn, test_loader)
    res["size_mb"] = dyn_size
    results["pytorch_dynamic"] = res
    print(f"  PyTorch Dynamic: {res['latency_median_ms']:.2f} ms | Size: {dyn_size:.2f} MB | F1: {res['macro_f1']:.4f}")
    
    print("Benchmarking PyTorch Static...")
    res = benchmark_pytorch_model(pt_stat, test_loader)
    res["size_mb"] = stat_size
    results["pytorch_static"] = res
    print(f"  PyTorch Static: {res['latency_median_ms']:.2f} ms | Size: {stat_size:.2f} MB | F1: {res['macro_f1']:.4f}")
    
    print("Benchmarking ONNX Float...")
    res = benchmark_onnx_model(ort_float_session, test_loader)
    res["size_mb"] = onnx_float_size
    results["onnx_float"] = res
    print(f"  ONNX Float: {res['latency_median_ms']:.2f} ms | Size: {onnx_float_size:.2f} MB | F1: {res['macro_f1']:.4f}")
    
    print("Benchmarking ONNX INT8 (Dynamic)...")
    res = benchmark_onnx_model(ort_int8_session, test_loader)
    res["size_mb"] = onnx_int8_size
    results["onnx_int8"] = res
    print(f"  ONNX INT8: {res['latency_median_ms']:.2f} ms | Size: {onnx_int8_size:.2f} MB | F1: {res['macro_f1']:.4f}")
    
    final_output = {
        "device": "cpu",
        "quantization_backend": torch.backends.quantized.engine,
        "results": results
    }
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/quantization_comparison.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Benchmarking complete. Results saved to eval/quantization_comparison.json")

if __name__ == "__main__":
    run_export_and_benchmark()
