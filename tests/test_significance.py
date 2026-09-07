import os
import sys
import json
import numpy as np
import pytest
import builtins
import torch
import scipy.stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distillation.student_model import StudentTransformer
from eval.aggregate_seeds import aggregate_seeds

def test_bootstrap_synthetic():
    np.random.seed(42)
    # Synthetic normal distribution: mean=0, std=1
    data = np.random.normal(0, 1, 1000)
    
    boot_means = []
    for _ in range(1000):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
        
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    # Standard error of mean = 1 / sqrt(1000) ~= 0.0316
    # 95% CI is roughly 0 ± 1.96 * 0.0316 = [-0.062, 0.062]
    assert -0.1 < ci_lower < -0.01
    assert 0.01 < ci_upper < 0.1

def test_paired_bootstrap_zero_diff():
    np.random.seed(42)
    preds_A = np.random.randint(0, 2, (100, 28))
    preds_B = np.copy(preds_A) # Identical
    labels = np.random.randint(0, 2, (100, 28))
    
    from sklearn.metrics import f1_score
    diffs = []
    for _ in range(10):
        idx = np.random.choice(100, 100, replace=True)
        f1_A = f1_score(labels[idx], preds_A[idx], average="macro", zero_division=0)
        f1_B = f1_score(labels[idx], preds_B[idx], average="macro", zero_division=0)
        diffs.append(f1_A - f1_B)
        
    assert np.allclose(diffs, 0.0)

def test_seed_setting():
    from distillation.student_model import StudentTransformer
    
    def get_model(seed):
        torch.manual_seed(seed)
        model = StudentTransformer(
            vocab_size=30522,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=512,
            num_labels=28
        )
        return model.state_dict()['word_embeddings.weight']
        
    w1 = get_model(42)
    w2 = get_model(42)
    w3 = get_model(123)
    
    assert torch.allclose(w1, w2)
    assert not torch.allclose(w1, w3)

def test_aggregation_missing_files(tmp_path, monkeypatch):
    import glob
    def mock_glob(pattern):
        if "distilled" in pattern:
            return [str(tmp_path / "d1.json")]
        elif "scratch" in pattern:
            return []
        return []
        
    monkeypatch.setattr(glob, "glob", mock_glob)
    
    with open(tmp_path / "d1.json", "w") as f:
        json.dump({"final_macro_f1": 0.3}, f)
        
    # Mock writing out files to tmp_path
    original_open = builtins.open
    def mock_open(path, mode='r', **kwargs):
        if str(path).startswith("eval/"):
            return original_open(tmp_path / os.path.basename(path), mode, **kwargs)
        return original_open(path, mode, **kwargs)
    monkeypatch.setattr("builtins.open", mock_open)
    
    aggregate_seeds()
    
    assert os.path.exists(tmp_path / "multiseed_results.json")
    with open(tmp_path / "multiseed_results.json", "r") as f:
        res = json.load(f)
        assert res["n_distilled"] == 1
        assert res["n_scratch"] == 0
        assert "ttest" not in res or not res["ttest"]

def test_welchs_ttest():
    A = [0.1, 0.2, 0.3, 0.4, 0.5]
    B = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    import scipy.stats as stats
    t_stat_scipy, p_val_scipy = stats.ttest_ind(A, B, equal_var=False)
    
    n_A, n_B = len(A), len(B)
    mean_A, mean_B = np.mean(A), np.mean(B)
    var_A, var_B = np.var(A, ddof=1), np.var(B, ddof=1)
    
    # Welch's t-test manual
    t_stat_manual = (mean_A - mean_B) / np.sqrt(var_A/n_A + var_B/n_B)
    
    assert np.isclose(t_stat_scipy, t_stat_manual)
