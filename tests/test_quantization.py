import os
import torch
import torch.nn as nn
import safetensors.torch
import pytest
import json
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distillation.student_model import StudentTransformer
from quantization.dynamic_quantize import run_quantization

def test_quantization_smoke_and_shapes(tmp_path):
    if torch.backends.quantized.engine == 'none':
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
            
    """
    Verify the quantized model produces output of the correct shape on a dummy batch.
    Verify outputs correlate strongly with unquantized outputs.
    Verify the quantized model is measurably smaller on disk.
    Verify reloading works and produces same outputs.
    """
    output_dir = tmp_path / "student_dynamic_quant"
    metrics_file = tmp_path / "quant_metrics.json"
    
    # We need an unquantized model to quantize!
    # Let's create a fake checkpoint dir in tmp_path
    unquant_dir = tmp_path / "student_distilled_fake"
    os.makedirs(unquant_dir)
    
    student_config = {
        "vocab_size": 1000,
        "max_position_embeddings": 128,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "num_labels": 28,
        "dropout_prob": 0.1
    }
    with open(unquant_dir / "student_config.json", "w") as f:
        json.dump(student_config, f)
        
    student = StudentTransformer(**student_config)
    safetensors.torch.save_file(student.state_dict(), unquant_dir / "model.safetensors")
    
    # Generate some dummy data
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones((2, 16))
    
    # Get unquantized outputs
    student.eval()
    with torch.no_grad():
        unquant_out = student(input_ids=input_ids, attention_mask=attention_mask)
        
    # We can use the run_quantization function which will benchmark on the actual test set,
    # but that might be slow even with num_samples. Actually num_samples=8 is very fast.
    # To test quantization correlation directly on our controlled model:
    quantized_student = torch.quantization.quantize_dynamic(
        student, {nn.Linear}, dtype=torch.qint8
    )
    def dummy_hook(*args): pass
    for layer in quantized_student.encoder.layers:
        layer.register_forward_hook(dummy_hook)
    
    # Verify shape
    with torch.no_grad():
        quant_out = quantized_student(input_ids=input_ids, attention_mask=attention_mask)
        
    assert quant_out.shape == (2, 28), "Output shape is incorrect"
    
    # Verify correlation > 0.95
    unquant_np = unquant_out.numpy().flatten()
    quant_np = quant_out.numpy().flatten()
    
    correlation = np.corrcoef(unquant_np, quant_np)[0, 1]
    assert correlation > 0.95, f"Quantization changed outputs too much, correlation: {correlation}"
    
    # Save both and compare size
    unquant_path = unquant_dir / "model.safetensors"
    quant_path = output_dir / "quantized_model.pt"
    os.makedirs(output_dir)
    
    torch.save(quantized_student.state_dict(), quant_path)
    
    unquant_size = os.path.getsize(unquant_path)
    quant_size = os.path.getsize(quant_path)
    
    # Quantized model must be at least 10% smaller (actually should be much smaller because Linear layers are INT8)
    assert quant_size < unquant_size * 0.9, f"Quantized size ({quant_size}) not significantly smaller than unquantized ({unquant_size})"
    
    # Verify reloading
    reloaded_student = StudentTransformer(**student_config)
    # We must quantize it dynamically first to create the right skeleton, then load state_dict
    reloaded_student = torch.quantization.quantize_dynamic(
        reloaded_student, {nn.Linear}, dtype=torch.qint8
    )
    for layer in reloaded_student.encoder.layers:
        layer.register_forward_hook(dummy_hook)
    reloaded_student.load_state_dict(torch.load(quant_path))
    reloaded_student.eval()
    
    with torch.no_grad():
        reloaded_out = reloaded_student(input_ids=input_ids, attention_mask=attention_mask)
        
    assert torch.allclose(quant_out, reloaded_out, atol=1e-5), "Reloaded quantized model produced different outputs"

def test_run_quantization_integration(tmp_path):
    """
    Run the actual script but with a tiny dataset to ensure it runs end-to-end.
    """
    output_dir = tmp_path / "quant_output"
    metrics_file = tmp_path / "metrics.json"
    
    # Only run if there is a real distilled student to test against
    real_model_dir = "model_checkpoints/student_distilled"
    if not os.path.exists(real_model_dir):
        pytest.skip(f"No model found in {real_model_dir}")
        
    results = run_quantization(
        model_dir=real_model_dir,
        output_dir=str(output_dir),
        metrics_file=str(metrics_file),
        batch_size=4,
        max_length=16,
        num_samples=8
    )
    
    assert "deltas" in results
    assert results["deltas"]["size_reduction_ratio"] > 1.0
    assert os.path.exists(metrics_file)
    assert os.path.exists(output_dir / "quantized_model.pt")
