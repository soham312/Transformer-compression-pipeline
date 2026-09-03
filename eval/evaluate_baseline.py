import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, 
    hamming_loss, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import load_and_tokenize_data

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def compute_overall_metrics(probs, labels):
    preds = (probs > 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    exact_match = accuracy_score(labels, preds)
    ham_loss = hamming_loss(labels, preds)
    hamming_acc = 1.0 - ham_loss
    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "exact_match_accuracy": float(exact_match),
        "hamming_accuracy": float(hamming_acc)
    }

def compute_per_class_metrics(probs, labels, label_names):
    preds = (probs > 0.5).astype(int)
    num_classes = labels.shape[1]
    per_class_metrics = {}
    
    for i in range(num_classes):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        y_pred = preds[:, i]
        label_name = label_names[i]
        
        if len(np.unique(y_true)) > 1:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            brier = brier_score_loss(y_true, y_prob)
        else:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            brier = brier_score_loss(y_true, y_prob) if len(y_true) > 0 else 0.0
            
        per_class_metrics[label_name] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "brier_score": float(brier)
        }
    return per_class_metrics

def generate_reliability_diagram(probs, labels, label_names, output_path):
    num_classes = labels.shape[1]
    fig, axes = plt.subplots(7, 4, figsize=(20, 30))
    axes = axes.flatten()
    
    for i in range(num_classes):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        label_name = label_names[i]
        
        ax = axes[i]
        if len(np.unique(y_true)) > 1:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            ax.plot(prob_pred, prob_true, marker='o', label=label_name)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_title(label_name)
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_baseline():
    device = get_device()
    print(f"Using device: {device}")
    
    model_dir = "model_checkpoints/teacher"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found at {model_dir}. Please complete Stage 3.")
        
    print("Loading test dataset...")
    # Load dataset
    test_ds, tokenizer = load_and_tokenize_data(model_name=model_dir, max_length=64, split="test")
    
    # Get label names
    raw_ds = load_dataset("google-research-datasets/go_emotions", split="train")
    label_names = raw_ds.features['labels'].feature.names
    
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    all_logits = []
    all_labels = []
    latencies = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch["input_ids"].to(device)
            b_attn_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].float().numpy()
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            start_time = time.time()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            latencies.append(time.time() - start_time)
            
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(b_labels)
            
    avg_latency_ms = (sum(latencies) / len(test_ds)) * 1000
    model_size_mb = get_dir_size(model_dir)
    
    flat_logits = np.vstack(all_logits)
    probs = 1.0 / (1.0 + np.exp(-flat_logits))
    flat_labels = np.vstack(all_labels).astype(int)
    
    overall_metrics = compute_overall_metrics(probs, flat_labels)
    overall_metrics["model_size_mb"] = float(model_size_mb)
    overall_metrics["avg_latency_ms_per_seq"] = float(avg_latency_ms)
    
    per_class_metrics = compute_per_class_metrics(probs, flat_labels, label_names)
    
    print("Generating reliability diagram...")
    os.makedirs("eval", exist_ok=True)
    generate_reliability_diagram(probs, flat_labels, label_names, "eval/calibration_reliability.png")
    
    baseline_results = {
        "overall": overall_metrics,
        "per_class": per_class_metrics
    }
    
    with open("eval/teacher_baseline.json", "w") as f:
        json.dump(baseline_results, f, indent=4)
        
    print("Baseline evaluation complete. Results saved to eval/teacher_baseline.json")

if __name__ == "__main__":
    evaluate_baseline()
