import os
import time
import json
import warnings
import torch
import numpy as np
import safetensors.torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import load_and_tokenize_data
from distillation.student_model import StudentTransformer
from distillation.loss import DistillationLoss

# --- Hyperparameters ---
NUM_EPOCHS = 15
BATCH_SIZE = 32
LR = 5e-5
ALPHA = 0.5
TEMPERATURE = 4.0
WARMUP_RATIO = 0.1
PATIENCE = 3
MAX_LENGTH = 64
NUM_SAMPLES = None
# -----------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def compute_metrics(logits, labels):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = np.array(labels)
    
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    
    ham_loss = hamming_loss(labels, preds)
    hamming_acc = 1.0 - ham_loss
    
    return {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1), "hamming_acc": float(hamming_acc)}

def precompute_teacher_logits(dataset, teacher, device, batch_size=64):
    """
    Runs the dataset through the frozen teacher once and caches logits in memory.
    Returns a TensorDataset containing (input_ids, attention_mask, labels, teacher_logits).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_input_ids = []
    all_attn_mask = []
    all_labels = []
    all_logits = []
    
    print("Precomputing teacher logits...")
    with torch.no_grad():
        for batch in loader:
            b_input_ids = batch["input_ids"]
            b_attn_mask = batch["attention_mask"]
            b_labels = batch["labels"]
            
            outputs = teacher(input_ids=b_input_ids.to(device), attention_mask=b_attn_mask.to(device))
            
            all_input_ids.append(b_input_ids)
            all_attn_mask.append(b_attn_mask)
            all_labels.append(b_labels)
            all_logits.append(outputs.logits.cpu())
            
    return TensorDataset(
        torch.cat(all_input_ids, dim=0),
        torch.cat(all_attn_mask, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_logits, dim=0)
    )

def train_student(
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    alpha=ALPHA,
    temperature=TEMPERATURE,
    warmup_ratio=WARMUP_RATIO,
    patience=PATIENCE,
    max_length=MAX_LENGTH,
    num_samples=NUM_SAMPLES,
    teacher_dir="model_checkpoints/teacher",
    output_dir="model_checkpoints/student_distilled",
    metrics_file="eval/student_distilled_metrics.json",
    seed=42
):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    print("Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    
    train_ds, _ = load_and_tokenize_data(model_name=teacher_dir, max_length=max_length, split="train", num_samples=num_samples)
    val_ds, _ = load_and_tokenize_data(model_name=teacher_dir, max_length=max_length, split="validation", num_samples=num_samples)
    
    print("Loading teacher model...")
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    print("Precomputing train and validation logits...")
    train_tensor_ds = precompute_teacher_logits(train_ds, teacher, device)
    val_tensor_ds = precompute_teacher_logits(val_ds, teacher, device)
    
    # We can remove the teacher from memory now if we want, but keeping it is fine.
    # del teacher
    # if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    train_loader = DataLoader(train_tensor_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_ds, batch_size=batch_size, shuffle=False)
        
    print("Initializing student model...")
    # Define architecture arguments explicitly to save them
    student_config = {
        "vocab_size": 30522,
        "max_position_embeddings": 512,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "num_labels": 28,
        "dropout_prob": 0.1
    }
    student = StudentTransformer(**student_config)
    student.to(device)
    
    optimizer = AdamW(student.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(warmup_ratio * total_steps), 
        num_training_steps=total_steps
    )
    
    loss_fn = DistillationLoss(alpha=alpha, temperature=temperature)
    
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": [], "val_hamming_acc": []}
    best_metrics = {}
    
    print("Starting distillation training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        student.train()
        total_train_loss = 0
        
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            b_labels = batch[2].float().to(device)
            t_logits = batch[3].to(device)
                
            optimizer.zero_grad()
            
            s_logits = student(input_ids=b_input_ids, attention_mask=b_attn_mask)
            
            loss = loss_fn(s_logits, t_logits, b_labels)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        student.eval()
        total_val_loss = 0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch[0].to(device)
                b_attn_mask = batch[1].to(device)
                b_labels = batch[2].float().to(device)
                t_logits = batch[3].to(device)
                
                s_logits = student(input_ids=b_input_ids, attention_mask=b_attn_mask)
                
                loss = loss_fn(s_logits, t_logits, b_labels)
                total_val_loss += loss.item()
                
                all_logits.append(s_logits.cpu().numpy())
                all_labels.append(b_labels.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        flat_logits = np.vstack(all_logits)
        flat_labels = np.vstack(all_labels)
        metrics = compute_metrics(flat_logits, flat_labels)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Macro F1: {metrics['macro_f1']:.4f} | Val Micro F1: {metrics['micro_f1']:.4f} | Val Hamming Acc: {metrics['hamming_acc']:.4f}")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(metrics['macro_f1'])
        history["val_micro_f1"].append(metrics['micro_f1'])
        history["val_hamming_acc"].append(metrics['hamming_acc'])
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_metrics = {
                "best_val_loss": float(best_val_loss),
                "best_macro_f1": float(metrics['macro_f1']),
                "best_micro_f1": float(metrics['micro_f1']),
                "best_hamming_acc": float(metrics['hamming_acc'])
            }
            print("Validation loss improved. Saving best student model...")
            safetensors.torch.save_file(
                student.state_dict(), 
                os.path.join(output_dir, "model.safetensors")
            )
            # Save student config
            with open(os.path.join(output_dir, "student_config.json"), "w") as f:
                json.dump(student_config, f, indent=4)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break
                
    def get_dir_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return total / (1024 * 1024)
        
    model_size_mb = get_dir_size(output_dir)
    print(f"\nFinal Student Model Size: {model_size_mb:.2f} MB")
    
    print("Measuring inference latency on validation set (Student)...")
    student.eval()
    latencies = []
    
    # Load best checkpoint and config
    with open(os.path.join(output_dir, "student_config.json"), "r") as f:
        loaded_config = json.load(f)
    best_student = StudentTransformer(**loaded_config)
    best_student.load_state_dict(safetensors.torch.load_file(os.path.join(output_dir, "model.safetensors")))
    best_student.to(device)
    best_student.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids = batch[0].to(device)
            b_attn_mask = batch[1].to(device)
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            start_time = time.time()
            best_student(input_ids=b_input_ids, attention_mask=b_attn_mask)
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                
            latencies.append(time.time() - start_time)
            
    avg_latency_ms = (sum(latencies) / len(val_ds)) * 1000
    print(f"Average Inference Latency (Student): {avg_latency_ms:.2f} ms per sequence")
    
    final_metrics = {
        "best_val_loss": best_metrics.get("best_val_loss", float("inf")),
        "final_macro_f1": best_metrics.get("best_macro_f1", 0.0),
        "final_micro_f1": best_metrics.get("best_micro_f1", 0.0),
        "final_hamming_acc": best_metrics.get("best_hamming_acc", 0.0),
        "model_size_mb": float(model_size_mb),
        "avg_latency_ms_per_seq": float(avg_latency_ms),
        "hyperparameters": {
            "alpha": float(alpha),
            "temperature": float(temperature),
            "lr": float(lr),
            "epochs_run": int(len(history["train_loss"])),
            "batch_size": int(batch_size),
            "seed": int(seed)
        },
        "history": history
    }
    
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Distillation pipeline complete! Metrics saved to {metrics_file}.")
    return final_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="model_checkpoints/student_distilled")
    parser.add_argument("--metrics_file", type=str, default="eval/student_distilled_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_student(output_dir=args.output_dir, metrics_file=args.metrics_file, seed=args.seed)
