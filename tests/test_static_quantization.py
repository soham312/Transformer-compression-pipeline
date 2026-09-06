import os
import torch
import torch.nn as nn
import safetensors.torch
import pytest
import json
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distillation.student_model import StudentTransformer

def test_static_quant_and_onnx_export(tmp_path):
    """
    - Static-quantized model produces correct output shape on a dummy batch
    - ONNX float model output matches PyTorch output within tolerance
    - ONNX INT8 model output correlates strongly with PyTorch output
    - Both ONNX models load in an InferenceSession without error
    """
    if torch.backends.quantized.engine == 'none':
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
            
    # Setup dummy student
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
    
    student = StudentTransformer(**student_config)
    student.eval()
    
    input_ids = torch.randint(0, 1000, (2, 16), dtype=torch.long)
    attention_mask = torch.ones((2, 16), dtype=torch.long)
    
    with torch.no_grad():
        pt_out = student(input_ids, attention_mask)
        
    # 1. Test Static Quantization (only classifier as per our script)
    pt_stat = StudentTransformer(**student_config)
    pt_stat.load_state_dict(student.state_dict())
    
    wrapper = torch.ao.quantization.QuantWrapper(pt_stat.classifier)
    pt_stat.classifier = wrapper
    pt_stat.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
    pt_stat.encoder.qconfig = None
    pt_stat.word_embeddings.qconfig = None
    pt_stat.position_embeddings.qconfig = None
    pt_stat.LayerNorm.qconfig = None
    
    torch.ao.quantization.prepare(pt_stat, inplace=True)
    pt_stat(input_ids, attention_mask)  # Calibrate
    torch.ao.quantization.convert(pt_stat, inplace=True)
    
    with torch.no_grad():
        stat_out = pt_stat(input_ids, attention_mask)
        
    assert stat_out.shape == (2, 28)
    
    # 2. Test ONNX Export & Float Match
    onnx_float_path = str(tmp_path / "model_float.onnx")
    torch.onnx.export(
        student,
        (input_ids, attention_mask),
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
    
    ort_float_session = ort.InferenceSession(onnx_float_path, providers=['CPUExecutionProvider'])
    ort_float_out = ort_float_session.run(None, {
        'input_ids': input_ids.numpy(),
        'attention_mask': attention_mask.numpy()
    })[0]
    
    # ONNX Float should match exactly or very closely
    assert np.allclose(pt_out.numpy(), ort_float_out, atol=1e-5), "ONNX Float doesn't match PyTorch Float"
    
    # 3. Test ONNX Dynamic Quantization Match (Correlation)
    onnx_int8_path = str(tmp_path / "model_int8.onnx")
    quantize_dynamic(
        model_input=onnx_float_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QInt8
    )
    
    ort_int8_session = ort.InferenceSession(onnx_int8_path, providers=['CPUExecutionProvider'])
    ort_int8_out = ort_int8_session.run(None, {
        'input_ids': input_ids.numpy(),
        'attention_mask': attention_mask.numpy()
    })[0]
    
    pt_flat = pt_out.numpy().flatten()
    ort_int8_flat = ort_int8_out.flatten()
    
    correlation = np.corrcoef(pt_flat, ort_int8_flat)[0, 1]
    # State threshold: 0.95 correlation is a strong indicator that the model isn't completely scrambled
    # Given we are applying dynamic quantization, the values might drift slightly but distribution remains very similar.
    assert correlation > 0.95, f"ONNX INT8 quantization ruined outputs, correlation: {correlation}"
