import os
import time
import json
import warnings
import torch
import numpy as np
import safetensors.torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import torch.nn as nn
import shutil

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import load_and_tokenize_data
from distillation.student_model import StudentTransformer

def compute_metrics(logits, labels):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
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

def evaluate_model(model, loader, device="cpu"):
    """
    Evaluates the model and computes latency and metrics.
    Warmup pass is included before latency measurement.
    """
    model.eval()
    all_logits = []
    all_labels = []
    latencies = []
    
    with torch.no_grad():
        # Warmup (10 batches)
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            b_input_ids = batch["input_ids"].to(device)
            b_attn_mask = batch["attention_mask"].to(device)
            model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            
        # Actual measurement
        for batch in loader:
            b_input_ids = batch["input_ids"].to(device)
            b_attn_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)
            
            start_time = time.time()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            # Synchronize isn't strictly necessary on CPU but keeping simple time is fine
            latencies.append(time.time() - start_time)
            
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(b_labels.cpu().numpy())
            
    flat_logits = np.vstack(all_logits)
    flat_labels = np.vstack(all_labels)
    metrics = compute_metrics(flat_logits, flat_labels)
    
    # Calculate latency per sequence in ms
    # latencies list contains time per batch
    batch_size = loader.batch_size
    latencies_per_seq_ms = [(lat / batch_size) * 1000 for lat in latencies]
    
    metrics["latency_mean_ms"] = float(np.mean(latencies_per_seq_ms))
    metrics["latency_median_ms"] = float(np.median(latencies_per_seq_ms))
    
    return metrics

def run_quantization(
    model_dir="model_checkpoints/student_distilled",
    output_dir="model_checkpoints/student_dynamic_quant",
    metrics_file="eval/student_dynamic_quant_metrics.json",
    batch_size=32,
    max_length=64,
    num_samples=None
):
    print("Setting backend for dynamic quantization...")
    # set engine appropriately for CPU
    if torch.backends.quantized.engine == 'none':
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
    
    quant_engine = torch.backends.quantized.engine
    print(f"Active quantization backend: {quant_engine}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    print("Loading test dataset...")
    test_ds, tokenizer = load_and_tokenize_data(model_name="bert-base-uncased", max_length=max_length, split="test", num_samples=num_samples)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print("Loading unquantized distilled student...")
    config_path = os.path.join(model_dir, "student_config.json")
    with open(config_path, "r") as f:
        student_config = json.load(f)
        
    student = StudentTransformer(**student_config)
    student.load_state_dict(safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors")))
    student.to("cpu")
    student.eval()
    
    print("Evaluating unquantized student on CPU...")
    unquant_metrics = evaluate_model(student, test_loader, device="cpu")
    unquant_size = get_dir_size(model_dir)
    unquant_metrics["size_mb"] = unquant_size
    print(f"Unquantized - Size: {unquant_size:.2f} MB, Latency (median): {unquant_metrics['latency_median_ms']:.2f} ms, Macro F1: {unquant_metrics['macro_f1']:.4f}")
    
    print("Applying dynamic quantization...")
    quantized_student = torch.quantization.quantize_dynamic(
        student,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    # PyTorch bug workaround: quantized Linear layers cause an AttributeError inside TransformerEncoderLayer
    # when it checks weights for the sparsity fast path. Adding a dummy forward hook disables the fast path.
    def dummy_hook(*args): pass
    for layer in quantized_student.encoder.layers:
        layer.register_forward_hook(dummy_hook)
    
    # Save the quantized model using torch.save on the state_dict
    # Safetensors doesn't support qint8 tensors natively yet, and torch.save state_dict is robust for reloading
    # as long as we reload into a dynamically quantized skeleton.
    print("Saving quantized model...")
    save_path = os.path.join(output_dir, "quantized_model.pt")
    torch.save(quantized_student.state_dict(), save_path)
    
    with open(os.path.join(output_dir, "student_config.json"), "w") as f:
        json.dump(student_config, f, indent=4)
        
    tokenizer.save_pretrained(output_dir)
    
    print("Evaluating quantized student on CPU...")
    quant_metrics = evaluate_model(quantized_student, test_loader, device="cpu")
    quant_size = get_dir_size(output_dir)
    quant_metrics["size_mb"] = quant_size
    print(f"Quantized - Size: {quant_size:.2f} MB, Latency (median): {quant_metrics['latency_median_ms']:.2f} ms, Macro F1: {quant_metrics['macro_f1']:.4f}")
    
    # Calculate Deltas
    deltas = {
        "size_reduction_ratio": unquant_size / quant_size if quant_size > 0 else 0,
        "size_reduction_mb": unquant_size - quant_size,
        "latency_speedup_mean": unquant_metrics["latency_mean_ms"] / quant_metrics["latency_mean_ms"] if quant_metrics["latency_mean_ms"] > 0 else 0,
        "latency_speedup_median": unquant_metrics["latency_median_ms"] / quant_metrics["latency_median_ms"] if quant_metrics["latency_median_ms"] > 0 else 0,
        "macro_f1_delta": quant_metrics["macro_f1"] - unquant_metrics["macro_f1"],
        "micro_f1_delta": quant_metrics["micro_f1"] - unquant_metrics["micro_f1"],
        "hamming_acc_delta": quant_metrics["hamming_acc"] - unquant_metrics["hamming_acc"]
    }
    
    final_output = {
        "device": "cpu",
        "quantization_backend": quant_engine,
        "unquantized": unquant_metrics,
        "quantized": quant_metrics,
        "deltas": deltas
    }
    
    with open(metrics_file, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Dynamic quantization complete! Metrics saved to {metrics_file}.")
    return final_output

if __name__ == "__main__":
    run_quantization()
