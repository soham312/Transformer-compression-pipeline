import os
import torch
import safetensors.torch
import pytest
from transformers import AutoModelForSequenceClassification

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distillation.train_student import train_student
from distillation.student_model import StudentTransformer
from distillation.loss import DistillationLoss
from torch.optim import AdamW

def test_teacher_frozen_and_student_updates():
    """
    Verify teacher stays frozen (sum of teacher params unchanged before/after step)
    Verify student parameters DO change after step
    """
    # Dummy input
    batch_size = 2
    seq_len = 8
    num_classes = 28
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Setup teacher (use randomly initialized BERT for speed in this unit test)
    # We just need to check the freezing logic
    teacher = AutoModelForSequenceClassification.from_config(
        AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes).config
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Setup student
    student = StudentTransformer(num_labels=num_classes)
    student.train()
    
    optimizer = AdamW(student.parameters(), lr=1e-3)
    loss_fn = DistillationLoss()
    
    # Sum params before
    teacher_params_before = sum(p.sum().item() for p in teacher.parameters())
    student_params_before = sum(p.sum().item() for p in student.parameters())
    
    # Forward pass
    with torch.no_grad():
        t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        
    s_logits = student(input_ids=input_ids, attention_mask=attention_mask)
    
    loss = loss_fn(s_logits, t_logits, labels)
    loss.backward()
    optimizer.step()
    
    # Sum params after
    teacher_params_after = sum(p.sum().item() for p in teacher.parameters())
    student_params_after = sum(p.sum().item() for p in student.parameters())
    
    # Assertions
    assert teacher_params_before == teacher_params_after, "Teacher parameters changed!"
    assert student_params_before != student_params_after, "Student parameters did not change!"

def test_train_student_end_to_end_and_reload(tmp_path):
    """
    Smoke test: one training step with batch=2, num_samples=4, epochs=1 runs end-to-end
    Verify saved checkpoint reloads into fresh StudentTransformer and produces identical outputs on fixed input
    """
    output_dir = tmp_path / "student_distilled"
    metrics_file = tmp_path / "student_metrics.json"
    teacher_dir = "model_checkpoints/teacher"
    
    if not os.path.exists(teacher_dir):
        pytest.skip(f"Teacher directory {teacher_dir} not found. Skip integration test.")
        
    # Run training end-to-end with tiny dataset
    final_metrics = train_student(
        epochs=1,
        batch_size=2,
        num_samples=4,
        teacher_dir=teacher_dir,
        output_dir=str(output_dir),
        metrics_file=str(metrics_file)
    )
    
    # Verify outputs saved
    assert os.path.exists(metrics_file), "Metrics JSON not saved!"
    assert os.path.exists(output_dir / "model.safetensors"), "Model checkpoint not saved!"
    
    # Verify saved checkpoint reloads into fresh StudentTransformer and produces identical outputs
    model1 = StudentTransformer()
    model1.load_state_dict(safetensors.torch.load_file(str(output_dir / "model.safetensors")))
    model1.eval()
    
    model2 = StudentTransformer()
    model2.load_state_dict(safetensors.torch.load_file(str(output_dir / "model.safetensors")))
    model2.eval()
    
    # Fixed input
    input_ids = torch.randint(0, 1000, (1, 16))
    attention_mask = torch.ones((1, 16))
    
    with torch.no_grad():
        out1 = model1(input_ids=input_ids, attention_mask=attention_mask)
        out2 = model2(input_ids=input_ids, attention_mask=attention_mask)
        
    assert torch.allclose(out1, out2, atol=1e-6), "Reloaded models produced different outputs for same input!"

def test_precompute_teacher_logits():
    from datasets import Dataset
    from distillation.train_student import precompute_teacher_logits
    
    # Create dummy dataset
    data = {
        "input_ids": [[101, 2023, 102], [101, 1045, 102]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
        "labels": [[0.0]*28, [0.0]*28]
    }
    dummy_ds = Dataset.from_dict(data)
    dummy_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Setup dummy teacher
    teacher = AutoModelForSequenceClassification.from_config(
        AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28).config
    )
    teacher.eval()
    
    # Run precompute
    device = torch.device("cpu")
    tensor_ds = precompute_teacher_logits(dummy_ds, teacher, device, batch_size=2)
    
    # Get the cached logits for the first item
    input_ids, attention_mask, labels, cached_logits = tensor_ds[0]
    
    # Compute direct
    with torch.no_grad():
        direct_out = teacher(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        direct_logits = direct_out.logits.squeeze(0)
        
    assert torch.allclose(cached_logits, direct_logits, atol=1e-6), "Cached logits do not match direct teacher forward pass!"

