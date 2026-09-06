import os
import time
import json
import warnings
import datetime
import numpy as np
import torch
import torch.nn as nn
import safetensors.torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import load_and_tokenize_data
from distillation.student_model import StudentTransformer

def compute_metrics_batch(logits_list, labels_list):
    logits = np.vstack(logits_list)
    labels = np.vstack(labels_list)
    
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    ham_loss = hamming_loss(labels, preds)
    hamming_acc = 1.0 - ham_loss
    exact_match = accuracy_score(labels, preds)
    
    return {
        "macro_f1": float(macro_f1), 
        "micro_f1": float(micro_f1), 
        "hamming_acc": float(hamming_acc),
        "exact_match_acc": float(exact_match)
    }

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

def count_parameters(model):
    if hasattr(model, "count_parameters"):
        return model.count_parameters()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_pytorch_model(model, loader, num_runs=5):
    model.eval()
    all_logits = []
    all_labels = []
    
    # 1. Warmup & collect metrics
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10: break
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            
        # Collect outputs once for correctness
        for batch in loader:
            b_input_ids = batch["input_ids"]
            b_attn_mask = batch["attention_mask"]
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            
            # Handle teacher model return tuple
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
                
            all_logits.append(logits.numpy())
            all_labels.append(batch["labels"].numpy())
            
    metrics = compute_metrics_batch(all_logits, all_labels)
    
    # 2. Timing passes
    all_latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            run_latencies = []
            for batch in loader:
                b_input_ids = batch["input_ids"]
                b_attn_mask = batch["attention_mask"]
                start = time.perf_counter()
                model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                end = time.perf_counter()
                run_latencies.append(end - start)
            
            # Convert to ms/seq
            run_latencies_ms_seq = [(lat / loader.batch_size) * 1000 for lat in run_latencies]
            all_latencies.extend(run_latencies_ms_seq)
            
    metrics["latency_mean_ms"] = float(np.mean(all_latencies))
    metrics["latency_median_ms"] = float(np.median(all_latencies))
    metrics["latency_std_ms"] = float(np.std(all_latencies))
    return metrics

def benchmark_onnx_model(session, loader, num_runs=5):
    input_name_ids = session.get_inputs()[0].name
    input_name_mask = session.get_inputs()[1].name
    
    all_logits = []
    all_labels = []
    
    # Warmup & collect metrics
    for i, batch in enumerate(loader):
        if i >= 10: break
        session.run(None, {
            input_name_ids: batch["input_ids"].numpy(),
            input_name_mask: batch["attention_mask"].numpy()
        })
        
    for batch in loader:
        outputs = session.run(None, {
            input_name_ids: batch["input_ids"].numpy(),
            input_name_mask: batch["attention_mask"].numpy()
        })
        all_logits.append(outputs[0])
        all_labels.append(batch["labels"].numpy())
        
    metrics = compute_metrics_batch(all_logits, all_labels)
    
    # Timing passes
    all_latencies = []
    for _ in range(num_runs):
        run_latencies = []
        for batch in loader:
            b_input_ids = batch["input_ids"].numpy()
            b_attn_mask = batch["attention_mask"].numpy()
            start = time.perf_counter()
            session.run(None, {
                input_name_ids: b_input_ids,
                input_name_mask: b_attn_mask
            })
            end = time.perf_counter()
            run_latencies.append(end - start)
            
        run_latencies_ms_seq = [(lat / loader.batch_size) * 1000 for lat in run_latencies]
        all_latencies.extend(run_latencies_ms_seq)
        
    metrics["latency_mean_ms"] = float(np.mean(all_latencies))
    metrics["latency_median_ms"] = float(np.median(all_latencies))
    metrics["latency_std_ms"] = float(np.std(all_latencies))
    return metrics

def load_student_config(model_dir):
    with open(os.path.join(model_dir, "student_config.json"), "r") as f:
        return json.load(f)

def run_benchmark():
    # 1. Setup global conditions
    thread_count = 4
    torch.set_num_threads(thread_count)
    
    if torch.backends.quantized.engine == 'none':
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
            
    quant_engine = torch.backends.quantized.engine
    
    batch_size = 32
    max_length = 64
    num_runs = 5
    
    # 2. Path checks
    teacher_dir = "model_checkpoints/teacher"
    distilled_dir = "model_checkpoints/student_distilled"
    scratch_dir = "model_checkpoints/student_scratch"
    
    print("Starting run_benchmark")
    
    for d in [teacher_dir, distilled_dir, scratch_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Missing required checkpoint directory: {d}")
            
    print("Loading test dataset...")
    test_ds, _ = load_and_tokenize_data("bert-base-uncased", max_length=max_length, split="test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print("Dataset loaded!")
    
    results = {}
    
    # 3. Teacher
    print("Benchmarking Variant 1: BERT-base teacher...")
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir)
    teacher.to("cpu")
    teacher.eval()
    t_metrics = benchmark_pytorch_model(teacher, test_loader, num_runs)
    t_metrics["size_mb"] = get_dir_size(teacher_dir)
    t_metrics["params"] = count_parameters(teacher)
    results["teacher"] = t_metrics
    
    # Compute relative baselines
    t_size = t_metrics["size_mb"]
    t_lat = t_metrics["latency_median_ms"]
    t_f1 = t_metrics["macro_f1"]
    
    # 4. Distilled Student
    print("Benchmarking Variant 2: Distilled student...")
    dist_config = load_student_config(distilled_dir)
    dist_student = StudentTransformer(**dist_config)
    dist_student.load_state_dict(safetensors.torch.load_file(os.path.join(distilled_dir, "model.safetensors")))
    dist_student.to("cpu")
    dist_student.eval()
    
    d_metrics = benchmark_pytorch_model(dist_student, test_loader, num_runs)
    d_metrics["size_mb"] = get_dir_size(distilled_dir)
    d_metrics["params"] = count_parameters(dist_student)
    results["student_distilled"] = d_metrics
    
    # 5. Scratch Student
    print("Benchmarking Variant 3: Scratch student...")
    scratch_config = load_student_config(scratch_dir)
    scratch_student = StudentTransformer(**scratch_config)
    scratch_student.load_state_dict(safetensors.torch.load_file(os.path.join(scratch_dir, "model.safetensors")))
    scratch_student.to("cpu")
    scratch_student.eval()
    
    s_metrics = benchmark_pytorch_model(scratch_student, test_loader, num_runs)
    s_metrics["size_mb"] = get_dir_size(scratch_dir)
    s_metrics["params"] = count_parameters(scratch_student)
    results["student_scratch"] = s_metrics
    
    # 6. Dynamically-quantized student
    print("Benchmarking Variant 4: Dynamically-quantized student...")
    pt_dyn = torch.quantization.quantize_dynamic(dist_student, {nn.Linear}, dtype=torch.qint8)
    def dummy_hook(*args): pass
    for layer in pt_dyn.encoder.layers:
        layer.register_forward_hook(dummy_hook)
        
    dyn_metrics = benchmark_pytorch_model(pt_dyn, test_loader, num_runs)
    # Estimate size in memory by saving temporarily
    tmp_path = "tmp_dyn.pt"
    torch.save(pt_dyn.state_dict(), tmp_path)
    dyn_metrics["size_mb"] = get_dir_size(tmp_path)
    os.remove(tmp_path)
    dyn_metrics["params"] = count_parameters(dist_student) # INT8 doesn't change parameter count
    results["student_dynamic_quant"] = dyn_metrics
    
    # 7. Statically-quantized student
    print("Benchmarking Variant 5: Statically-quantized student...")
    pt_stat = StudentTransformer(**dist_config)
    pt_stat.load_state_dict(dist_student.state_dict())
    
    wrapper = torch.ao.quantization.QuantWrapper(pt_stat.classifier)
    pt_stat.classifier = wrapper
    pt_stat.qconfig = torch.ao.quantization.get_default_qconfig(quant_engine)
    pt_stat.encoder.qconfig = None
    pt_stat.word_embeddings.qconfig = None
    pt_stat.position_embeddings.qconfig = None
    pt_stat.LayerNorm.qconfig = None
    
    torch.ao.quantization.prepare(pt_stat, inplace=True)
    # Calibrate on train split subset
    calib_ds, _ = load_and_tokenize_data("bert-base-uncased", max_length=max_length, split="train", num_samples=320)
    calib_loader = DataLoader(calib_ds, batch_size=32)
    with torch.no_grad():
        for b in calib_loader: pt_stat(b["input_ids"], b["attention_mask"])
    torch.ao.quantization.convert(pt_stat, inplace=True)
    
    stat_metrics = benchmark_pytorch_model(pt_stat, test_loader, num_runs)
    torch.save(pt_stat.state_dict(), "tmp_stat.pt")
    stat_metrics["size_mb"] = get_dir_size("tmp_stat.pt")
    os.remove("tmp_stat.pt")
    stat_metrics["params"] = count_parameters(dist_student)
    results["student_static_quant"] = stat_metrics
    
    # 8. ONNX INT8 student
    print("Benchmarking Variant 6: ONNX INT8 student...")
    onnx_float_path = "tmp_model_float.onnx"
    onnx_int8_path = "tmp_model_int8.onnx"
    
    dummy_input_ids = torch.randint(0, 1000, (32, 64), dtype=torch.long)
    dummy_attention_mask = torch.ones((32, 64), dtype=torch.long)
    
    torch.onnx.export(
        dist_student,
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
    
    quantize_dynamic(
        model_input=onnx_float_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8
    )
    
    ort_int8_session = ort.InferenceSession(onnx_int8_path, providers=['CPUExecutionProvider'])
    onnx_metrics = benchmark_onnx_model(ort_int8_session, test_loader, num_runs)
    onnx_metrics["size_mb"] = get_dir_size(onnx_int8_path)
    onnx_metrics["params"] = count_parameters(dist_student)
    results["student_onnx_int8"] = onnx_metrics
    
    os.remove(onnx_float_path)
    os.remove(onnx_int8_path)
    
    # Compute relative metrics for all
    for name, m in results.items():
        m["size_reduction_factor"] = float(t_size / m["size_mb"])
        m["speedup_factor"] = float(t_lat / m["latency_median_ms"])
        m["macro_f1_retained_pct"] = float((m["macro_f1"] / t_f1) * 100.0)
        
    metadata = {
        "device": "cpu",
        "thread_count": thread_count,
        "batch_size": batch_size,
        "max_length": max_length,
        "test_split_size": len(test_ds),
        "timing_repetitions": num_runs,
        "timing_method": "time.perf_counter() over batch loop, divided by batch_size, warmup 10 batches",
        "quantization_backend": quant_engine,
        "date": datetime.datetime.now().isoformat(),
        "caveats": {
            "static_quantization": "Applied only to the classifier layer. nn.TransformerEncoderLayer crashes during eager-mode calibration, leaving encoder and embeddings in FP32. Size and latency do not reflect a fully statically quantized model."
        }
    }
    
    final_output = {
        "metadata": metadata,
        "results": results
    }
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/benchmark_results.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    # Generate Markdown Table
    md_lines = []
    md_lines.append("# Comprehensive Multi-Model Benchmark Results")
    md_lines.append("")
    md_lines.append(f"**Device:** {metadata['device'].upper()} | **Threads:** {metadata['thread_count']} | **Batch Size:** {metadata['batch_size']}")
    md_lines.append("")
    md_lines.append("| Variant | Size (MB) | Size Reduction | Median Latency (ms) | Speedup | Macro F1 | F1 Retained | Micro F1 | Exact Match |")
    md_lines.append("|---------|-----------|----------------|---------------------|---------|----------|-------------|----------|-------------|")
    
    variants_order = [
        ("Teacher (BERT-base)", "teacher"),
        ("Distilled Student", "student_distilled"),
        ("Scratch Student", "student_scratch"),
        ("Dynamic INT8 (PyTorch)", "student_dynamic_quant"),
        ("Static INT8 (PyTorch)*", "student_static_quant"),
        ("Dynamic INT8 (ONNX)", "student_onnx_int8"),
    ]
    
    for disp_name, key in variants_order:
        m = results[key]
        md_lines.append(f"| {disp_name} | {m['size_mb']:.2f} | {m['size_reduction_factor']:.2f}x | {m['latency_median_ms']:.2f} | {m['speedup_factor']:.2f}x | {m['macro_f1']:.4f} | {m['macro_f1_retained_pct']:.1f}% | {m['micro_f1']:.4f} | {m['exact_match_acc']:.4f} |")
        
    md_lines.append("")
    md_lines.append("* *Note on Static INT8 (PyTorch): Static quantization was applied to the classifier only due to eager-mode bugs in PyTorch's TransformerEncoderLayer. Encoder and embeddings remain FP32.*")
    
    md_table = "\n".join(md_lines)
    with open("eval/benchmark_table.md", "w") as f:
        f.write(md_table)
        
    print("\n" + md_table)
    print("\nBenchmark complete. Results saved to eval/benchmark_results.json and eval/benchmark_table.md")

if __name__ == "__main__":
    run_benchmark()
