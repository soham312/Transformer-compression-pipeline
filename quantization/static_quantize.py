import os
import json
import warnings
import torch
import torch.nn as nn
import safetensors.torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import load_and_tokenize_data
from distillation.student_model import StudentTransformer

def run_static_quantization(
    model_dir="model_checkpoints/student_distilled",
    output_dir="model_checkpoints/student_static_quant",
    batch_size=32,
    max_length=64,
    calibration_batches=100
):
    print("Setting backend for static quantization...")
    if torch.backends.quantized.engine == 'none':
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        elif 'fbgemm' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
    
    quant_engine = torch.backends.quantized.engine
    print(f"Active quantization backend: {quant_engine}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading unquantized distilled student...")
    config_path = os.path.join(model_dir, "student_config.json")
    with open(config_path, "r") as f:
        student_config = json.load(f)
        
    student = StudentTransformer(**student_config)
    student.load_state_dict(safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors")))
    student.to("cpu")
    student.eval()
    
    print("Loading train dataset for calibration...")
    # Load just enough data for calibration (e.g., calibration_batches * batch_size)
    num_samples = calibration_batches * batch_size
    train_ds, tokenizer = load_and_tokenize_data(model_name="bert-base-uncased", max_length=max_length, split="train", num_samples=num_samples)
    calib_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # -------------------------------------------------------------
    # Static Quantization Configuration
    # -------------------------------------------------------------
    # Note: PyTorch eager-mode static quantization for nn.TransformerEncoderLayer
    # is highly bugged (crashes during mask broadcasting). To achieve a robust
    # static quantization locally, we statically quantize the classifier only
    # and leave the embedding and encoder layers in float32.
    
    wrapper = torch.ao.quantization.QuantWrapper(student.classifier)
    student.classifier = wrapper
    
    student.qconfig = torch.ao.quantization.get_default_qconfig(quant_engine)
    
    # Keep encoder, embeddings, and LayerNorm float
    student.encoder.qconfig = None
    student.word_embeddings.qconfig = None
    student.position_embeddings.qconfig = None
    student.LayerNorm.qconfig = None
    
    print("Preparing model for static quantization...")
    torch.ao.quantization.prepare(student, inplace=True)
    
    print(f"Calibrating on {len(calib_loader)} batches...")
    with torch.no_grad():
        for batch in calib_loader:
            b_input_ids = batch["input_ids"]
            b_attn_mask = batch["attention_mask"]
            student(b_input_ids, b_attn_mask)
            
    print("Converting model to quantized form...")
    torch.ao.quantization.convert(student, inplace=True)
    
    print("Saving static quantized model...")
    save_path = os.path.join(output_dir, "quantized_model.pt")
    torch.save(student.state_dict(), save_path)
    
    with open(os.path.join(output_dir, "student_config.json"), "w") as f:
        json.dump(student_config, f, indent=4)
        
    tokenizer.save_pretrained(output_dir)
    print("Static quantization export complete!")
    return save_path

if __name__ == "__main__":
    run_static_quantization()
