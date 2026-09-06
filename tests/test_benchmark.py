import os
import json
import pytest
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distillation.student_model import StudentTransformer
from eval.benchmark_all import benchmark_pytorch_model, compute_metrics_batch

def test_benchmark_metrics():
    # Test ratio correctness
    t_size = 100.0
    t_lat = 50.0
    t_f1 = 0.50
    
    m = {"size_mb": 10.0, "latency_median_ms": 5.0, "macro_f1": 0.45}
    m["size_reduction_factor"] = t_size / m["size_mb"]
    m["speedup_factor"] = t_lat / m["latency_median_ms"]
    m["macro_f1_retained_pct"] = (m["macro_f1"] / t_f1) * 100.0
    
    assert m["size_reduction_factor"] == 10.0
    assert m["speedup_factor"] == 10.0
    assert m["macro_f1_retained_pct"] == 90.0
    
    # Test compute metrics required keys
    logits = [np.zeros((2, 28))]
    labels = [np.zeros((2, 28))]
    metrics = compute_metrics_batch(logits, labels)
    
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert "hamming_acc" in metrics
    assert "exact_match_acc" in metrics

def test_missing_checkpoints():
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from eval.benchmark_all import run_benchmark
    
    original_exists = os.path.exists
    def mock_exists(path):
        if "teacher" in path: return False
        return True
        
    os.path.exists = mock_exists
    try:
        with pytest.raises(FileNotFoundError, match="Missing required checkpoint directory: model_checkpoints/teacher"):
            run_benchmark()
    finally:
        os.path.exists = original_exists

def test_variants_output_shape(tmp_path):
    # Verify all six variants load and produce logits of shape (batch, 28) on a dummy batch
    # without touching eval/ or model_checkpoints/ writing
    import safetensors.torch
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    # We will load from real paths since we are testing loading
    teacher_dir = "model_checkpoints/teacher"
    distilled_dir = "model_checkpoints/student_distilled"
    scratch_dir = "model_checkpoints/student_scratch"
    onnx_dir = "model_checkpoints/student_onnx"
    
    if not os.path.exists(teacher_dir) or not os.path.exists(distilled_dir) or not os.path.exists(scratch_dir) or not os.path.exists(onnx_dir):
        pytest.skip("Checkpoints not available to test loading")
        
    batch_size = 32
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 64), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, 64), dtype=torch.long)
    
    # 1. Teacher
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir)
    teacher.eval()
    with torch.no_grad():
        out1 = teacher(dummy_input_ids, dummy_attention_mask).logits
    assert out1.shape == (batch_size, 28)
    
    # 2. Distilled Student
    with open(os.path.join(distilled_dir, "student_config.json")) as f:
        config = json.load(f)
    dist_student = StudentTransformer(**config)
    dist_student.load_state_dict(safetensors.torch.load_file(os.path.join(distilled_dir, "model.safetensors")))
    dist_student.eval()
    with torch.no_grad():
        out2 = dist_student(dummy_input_ids, dummy_attention_mask)
    assert out2.shape == (batch_size, 28)
    
    # 3. Scratch Student
    with open(os.path.join(scratch_dir, "student_config.json")) as f:
        s_config = json.load(f)
    scratch_student = StudentTransformer(**s_config)
    scratch_student.load_state_dict(safetensors.torch.load_file(os.path.join(scratch_dir, "model.safetensors")))
    scratch_student.eval()
    with torch.no_grad():
        out3 = scratch_student(dummy_input_ids, dummy_attention_mask)
    assert out3.shape == (batch_size, 28)
    
    # 4. Dynamic Student
    pt_dyn = torch.quantization.quantize_dynamic(dist_student, {nn.Linear}, dtype=torch.qint8)
    def dummy_hook(*args): pass
    for layer in pt_dyn.encoder.layers:
        layer.register_forward_hook(dummy_hook)
    with torch.no_grad():
        out4 = pt_dyn(dummy_input_ids, dummy_attention_mask)
    assert out4.shape == (batch_size, 28)
    
    # 5. Static Student (classifier only)
    if torch.backends.quantized.engine == 'none':
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
            
    pt_stat = StudentTransformer(**config)
    pt_stat.load_state_dict(dist_student.state_dict())
    wrapper = torch.ao.quantization.QuantWrapper(pt_stat.classifier)
    pt_stat.classifier = wrapper
    pt_stat.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
    pt_stat.encoder.qconfig = None
    pt_stat.word_embeddings.qconfig = None
    pt_stat.position_embeddings.qconfig = None
    pt_stat.LayerNorm.qconfig = None
    torch.ao.quantization.prepare(pt_stat, inplace=True)
    with torch.no_grad(): pt_stat(dummy_input_ids, dummy_attention_mask)
    torch.ao.quantization.convert(pt_stat, inplace=True)
    with torch.no_grad():
        out5 = pt_stat(dummy_input_ids, dummy_attention_mask)
    assert out5.shape == (batch_size, 28)
    
    # 6. ONNX INT8
    ort_session = ort.InferenceSession(os.path.join(onnx_dir, "model_int8.onnx"), providers=['CPUExecutionProvider'])
    out6 = ort_session.run(None, {
        ort_session.get_inputs()[0].name: dummy_input_ids.numpy(),
        ort_session.get_inputs()[1].name: dummy_attention_mask.numpy()
    })[0]
    assert out6.shape == (batch_size, 28)

