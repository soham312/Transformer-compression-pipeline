import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import safetensors.torch
import onnxruntime as ort
import sys
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import load_and_tokenize_data
from distillation.student_model import StudentTransformer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

def get_predictions(model_variant, test_loader, onnx=False):
    all_logits = []
    all_labels = []
    
    if not onnx:
        model_variant.eval()
        
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        if onnx:
            inputs = {
                model_variant.get_inputs()[0].name: input_ids.numpy(),
                model_variant.get_inputs()[1].name: attention_mask.numpy()
            }
            logits = model_variant.run(None, inputs)[0]
            all_logits.append(logits)
        else:
            with torch.no_grad():
                outputs = model_variant(input_ids, attention_mask=attention_mask)
                if hasattr(outputs, "logits"): # Teacher
                    outputs = outputs.logits
            all_logits.append(outputs.numpy())
            
        all_labels.append(labels.numpy())
        
    return np.vstack(all_logits), np.vstack(all_labels)

def compute_metrics(logits, labels):
    preds = (logits > 0).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "hamming_acc": 1.0 - hamming_loss(labels, preds)
    }

def run_bootstrap_significance(n_iterations=1000):
    teacher_dir = "model_checkpoints/teacher"
    distilled_dir = "model_checkpoints/student_distilled"
    scratch_dir = "model_checkpoints/student_scratch"
    onnx_dir = "model_checkpoints/student_onnx"
    
    max_length = 64
    batch_size = 32
    
    print("Loading test dataset...")
    test_ds, _ = load_and_tokenize_data("bert-base-uncased", max_length=max_length, split="test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    predictions = {}
    print("Caching predictions...")
    
    # 1. Teacher
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir)
    predictions["Teacher"], labels = get_predictions(teacher, test_loader)
    
    # 2. Distilled
    with open(os.path.join(distilled_dir, "student_config.json")) as f:
        config = json.load(f)
    dist_student = StudentTransformer(**config)
    dist_student.load_state_dict(safetensors.torch.load_file(os.path.join(distilled_dir, "model.safetensors")))
    predictions["Distilled"], _ = get_predictions(dist_student, test_loader)
    
    # 3. Scratch
    with open(os.path.join(scratch_dir, "student_config.json")) as f:
        s_config = json.load(f)
    scratch_student = StudentTransformer(**s_config)
    scratch_student.load_state_dict(safetensors.torch.load_file(os.path.join(scratch_dir, "model.safetensors")))
    predictions["Scratch"], _ = get_predictions(scratch_student, test_loader)
    
    # Set quantization engine for both dynamic and static
    if torch.backends.quantized.engine == 'none':
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
            
    # 4. Dynamic INT8
    pt_dyn = torch.quantization.quantize_dynamic(dist_student, {nn.Linear}, dtype=torch.qint8)
    def dummy_hook(*args): pass
    for layer in pt_dyn.encoder.layers:
        layer.register_forward_hook(dummy_hook)
    predictions["Dynamic_INT8_PyTorch"], _ = get_predictions(pt_dyn, test_loader)
    
    # 5. Static INT8 (classifier only)
    pt_stat = StudentTransformer(**config)
    pt_stat.load_state_dict(dist_student.state_dict())
    pt_stat.classifier = torch.ao.quantization.QuantWrapper(pt_stat.classifier)
    pt_stat.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
    pt_stat.encoder.qconfig = None
    pt_stat.word_embeddings.qconfig = None
    pt_stat.position_embeddings.qconfig = None
    pt_stat.LayerNorm.qconfig = None
    torch.ao.quantization.prepare(pt_stat, inplace=True)
    # Calibrate
    pt_stat.eval()
    for batch in test_loader:
        pt_stat(batch['input_ids'], attention_mask=batch['attention_mask'])
        break # One batch is enough for testing
    torch.ao.quantization.convert(pt_stat, inplace=True)
    predictions["Static_INT8_PyTorch"], _ = get_predictions(pt_stat, test_loader)
    
    # 6. ONNX INT8
    ort_session = ort.InferenceSession(os.path.join(onnx_dir, "model_int8.onnx"), providers=['CPUExecutionProvider'])
    predictions["Dynamic_INT8_ONNX"], _ = get_predictions(ort_session, test_loader, onnx=True)
    
    # --- Bootstrap ---
    print(f"Running bootstrap resampling ({n_iterations} iterations)...")
    np.random.seed(42)
    n_samples = labels.shape[0]
    
    bootstrap_results = {k: {"macro_f1": [], "micro_f1": [], "hamming_acc": []} for k in predictions.keys()}
    paired_distilled_scratch = []
    paired_quant_unquant = [] # ONNX INT8 vs Distilled FP32
    
    for i in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        resampled_labels = labels[indices]
        
        for variant, logits in predictions.items():
            resampled_logits = logits[indices]
            metrics = compute_metrics(resampled_logits, resampled_labels)
            for m in metrics:
                bootstrap_results[variant][m].append(metrics[m])
                
        # Paired comparisons
        dist_logits = predictions["Distilled"][indices]
        scratch_logits = predictions["Scratch"][indices]
        onnx_logits = predictions["Dynamic_INT8_ONNX"][indices]
        
        m_dist = compute_metrics(dist_logits, resampled_labels)["macro_f1"]
        m_scratch = compute_metrics(scratch_logits, resampled_labels)["macro_f1"]
        m_onnx = compute_metrics(onnx_logits, resampled_labels)["macro_f1"]
        
        paired_distilled_scratch.append(m_dist - m_scratch)
        paired_quant_unquant.append(m_onnx - m_dist)
        
    # --- Reporting ---
    print("\n--- Bootstrap 95% Confidence Intervals ---")
    for variant in predictions.keys():
        print(f"\n{variant}:")
        for m in ["macro_f1", "micro_f1", "hamming_acc"]:
            arr = np.array(bootstrap_results[variant][m])
            mean = np.mean(arr)
            std = np.std(arr)
            ci_lower = np.percentile(arr, 2.5)
            ci_upper = np.percentile(arr, 97.5)
            print(f"  {m}: {mean:.4f} ± {std:.4f}  [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
            
    print("\n--- Paired Bootstrap Differences (Macro F1) ---")
    
    arr_ds = np.array(paired_distilled_scratch)
    ds_mean = np.mean(arr_ds)
    ds_ci_lower = np.percentile(arr_ds, 2.5)
    ds_ci_upper = np.percentile(arr_ds, 97.5)
    ds_win_rate = np.mean(arr_ds > 0)
    print(f"\nDistilled minus Scratch:")
    print(f"  Mean diff: {ds_mean:.4f}")
    print(f"  95% CI: [{ds_ci_lower:.4f}, {ds_ci_upper:.4f}]")
    print(f"  Distilled win rate: {ds_win_rate:.1%}")
    if ds_ci_lower <= 0 <= ds_ci_upper:
        print("  => The 95% CI straddles zero. The difference is NOT statistically significant.")
    else:
        print("  => The 95% CI excludes zero. The difference IS statistically significant.")
        
    arr_qu = np.array(paired_quant_unquant)
    qu_mean = np.mean(arr_qu)
    qu_ci_lower = np.percentile(arr_qu, 2.5)
    qu_ci_upper = np.percentile(arr_qu, 97.5)
    print(f"\nQuantized (ONNX INT8) minus Unquantized (Distilled FP32):")
    print(f"  Mean diff: {qu_mean:.4f}")
    print(f"  95% CI: [{qu_ci_lower:.4f}, {qu_ci_upper:.4f}]")
    if qu_ci_lower <= 0 <= qu_ci_upper:
        print("  => The 95% CI straddles zero. Quantization's accuracy cost is indistinguishable from zero noise.")
    else:
        print("  => The 95% CI excludes zero. There is a statistically significant impact.")
        
if __name__ == "__main__":
    run_bootstrap_significance()
