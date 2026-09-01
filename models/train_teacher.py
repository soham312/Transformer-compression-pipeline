"""
Teacher Model Training Pipeline (BERT-base)
Fine-tunes a pretrained BERT model on the GoEmotions dataset.
"""
import os
import time
import json
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss

# Filter out sklearn warnings for ill-defined metrics on small test batches
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import load_and_tokenize_data

def get_device():
    """
    Returns MPS device if available (Apple Silicon), else CUDA if available, else CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def compute_metrics(logits, labels):
    """
    Computes Macro F1, Exact Match Accuracy, and Macro AUC.
    """
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = np.array(labels)
    
    # Macro F1
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    
    # Exact match accuracy
    acc = accuracy_score(labels, preds)
    
    # Macro AUC
    try:
        auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        # Happens in testing if a class has no positive examples in the batch
        auc = 0.5
        
    return {"f1": f1, "accuracy": acc, "auc": auc}

def get_dataloader(dataset, batch_size=16, shuffle=False):
    # The dataset already has set_format("torch") applied in load_and_tokenize_data
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_teacher(
    model_name="bert-base-uncased",
    epochs=3,
    batch_size=16,
    lr=2e-5,
    max_length=64, # Optimized sequence length found in EDA
    patience=2,
    output_dir="model_checkpoints/teacher",
    num_samples=None,  # Helpful for fast testing
    plot_dir="dashboard"
):
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs("eval", exist_ok=True)
    
    print("Loading datasets...")
    train_ds, tokenizer = load_and_tokenize_data(model_name, max_length=max_length, split="train", num_samples=num_samples)
    val_ds, _ = load_and_tokenize_data(model_name, max_length=max_length, split="validation", num_samples=num_samples)
    
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Initializing {model_name} for multi-label classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=28, 
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    
    # Linear warmup and decay scheduling
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # Loss function for multi-label classification
    loss_fn = BCEWithLogitsLoss()
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    
    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(train_loader):
            b_input_ids = batch["input_ids"].to(device)
            b_attn_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            logits = outputs.logits
            
            loss = loss_fn(logits, b_labels)
            total_train_loss += loss.item()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch["input_ids"].to(device)
                b_attn_mask = batch["attention_mask"].to(device)
                b_labels = batch["labels"].float().to(device)
                outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask)
                logits = outputs.logits
                loss = loss_fn(logits, b_labels)
                total_val_loss += loss.item()
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(b_labels.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        flat_logits = np.vstack(all_logits)
        flat_labels = np.vstack(all_labels)
        metrics = compute_metrics(flat_logits, flat_labels)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val F1: {metrics['f1']:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val AUC: {metrics['auc']:.4f}")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(metrics["f1"])
        history["val_auc"].append(metrics["auc"])
        
        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("Validation loss improved. Saving best model checkpoint...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
                
    # Calculate model size on disk
    def get_dir_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return total / (1024 * 1024) # Return size in MB
        
    model_size_mb = get_dir_size(output_dir)
    print(f"\nFinal Model Size: {model_size_mb:.2f} MB")
    
    # Measure Latency (on validation set)
    print("Measuring inference latency on validation set...")
    model.eval()
    latencies = []
    
    # Load the best saved model for evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    best_model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids = batch["input_ids"].to(device)
            b_attn_mask = batch["attention_mask"].to(device)
            
            # Synchronize before starting timer
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            start_time = time.time()
            best_model(input_ids=b_input_ids, attention_mask=b_attn_mask)
            
            # Synchronize after inference to get accurate time
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            latencies.append(time.time() - start_time)
            
    # Calculate average latency per sequence
    avg_latency_ms = (sum(latencies) / len(val_ds)) * 1000
    print(f"Average Inference Latency: {avg_latency_ms:.2f} ms per sequence")
    
    final_metrics = {
        "best_val_loss": best_val_loss,
        "final_f1": history["val_f1"][-1],
        "final_auc": history["val_auc"][-1],
        "final_accuracy": metrics["accuracy"],
        "model_size_mb": model_size_mb,
        "avg_latency_ms_per_seq": avg_latency_ms
    }
    
    with open("eval/teacher_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Val Loss", marker='o')
    plt.title("Teacher Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["val_f1"], label="Macro F1", marker='s')
    plt.plot(history["val_auc"], label="Macro AUC", marker='^')
    plt.title("Teacher Model Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "teacher_training_curves.png"))
    plt.close()
    
    print("Training pipeline complete! Baseline metrics saved to eval/teacher_metrics.json.")
    return final_metrics

if __name__ == "__main__":
    train_teacher()
