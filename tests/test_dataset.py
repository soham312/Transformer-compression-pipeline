import pytest
import torch
import sys
import os

# Add the project root to sys.path so we can import from data
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import load_and_tokenize_data

def test_load_and_tokenize_data():
    # Load a tiny subset to ensure tests run quickly
    dataset, tokenizer = load_and_tokenize_data(
        model_name="bert-base-uncased",
        max_length=32,
        split="train",
        num_samples=10
    )
    
    # Check dataset length
    assert len(dataset) == 10
    
    # Check if necessary columns exist
    expected_columns = ["input_ids", "attention_mask", "labels"]
    for col in expected_columns:
        assert col in dataset.features
        
    # Check tensor shapes and types
    sample = dataset[0]
    
    # input_ids and attention_mask should be 1D tensors of size 32
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert sample["input_ids"].shape == (32,)
    
    assert isinstance(sample["attention_mask"], torch.Tensor)
    assert sample["attention_mask"].shape == (32,)
    
    # labels should be a 1D tensor of size 28 (GoEmotions has 28 classes)
    assert isinstance(sample["labels"], torch.Tensor)
    assert sample["labels"].shape == (28,)
    # The labels can be integers or floats (we will cast to float for BCEWithLogitsLoss later)
    assert sample["labels"].dtype in [torch.float32, torch.float64, torch.int64]
    
    # Check tokenizer configuration
    assert tokenizer.name_or_path == "bert-base-uncased"
    assert tokenizer.model_max_length >= 32
