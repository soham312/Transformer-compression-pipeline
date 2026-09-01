import os
import sys
import pytest
import torch

# Add the project root to sys.path so we can import from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train_teacher import train_teacher, get_device

def test_device_fallback():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["mps", "cuda", "cpu"]

def test_train_teacher_pipeline(tmp_path):
    output_dir = os.path.join(tmp_path, "teacher")
    plot_dir = os.path.join(tmp_path, "dashboard")
    
    # Run a tiny training loop (1 epoch, 8 samples, using tiny BERT to keep test time low)
    metrics = train_teacher(
        model_name="prajjwal1/bert-tiny", # Extremely fast tiny BERT model
        epochs=1,
        batch_size=4,
        max_length=16,
        output_dir=output_dir,
        num_samples=8,
        plot_dir=plot_dir
    )
    
    # Verify model assets were saved
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "config.json"))
    
    # Check if either pytorch_model.bin or model.safetensors exists
    has_weights = os.path.exists(os.path.join(output_dir, "pytorch_model.bin")) or \
                  os.path.exists(os.path.join(output_dir, "model.safetensors"))
    assert has_weights, "Model weights were not saved."
    
    # Verify outputs and plots
    assert os.path.exists(os.path.join(plot_dir, "teacher_training_curves.png"))
    assert os.path.exists("eval/teacher_metrics.json")
    
    # Verify metrics dictionary structure
    assert "best_val_loss" in metrics
    assert "final_f1" in metrics
    assert "final_auc" in metrics
    assert "model_size_mb" in metrics
    assert "avg_latency_ms_per_seq" in metrics
    
    assert metrics["model_size_mb"] > 0
    assert metrics["avg_latency_ms_per_seq"] > 0
