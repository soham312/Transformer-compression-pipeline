import numpy as np
import pytest
from eval.evaluate_baseline import compute_overall_metrics, compute_per_class_metrics

def test_compute_overall_metrics():
    # 3 samples, 2 classes
    probs = np.array([
        [0.9, 0.1],  # Pred: [1, 0]
        [0.4, 0.8],  # Pred: [0, 1]
        [0.2, 0.2]   # Pred: [0, 0]
    ])
    labels = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    
    metrics = compute_overall_metrics(probs, labels)
    
    assert np.isclose(metrics["exact_match_accuracy"], 1.0)
    assert np.isclose(metrics["hamming_accuracy"], 1.0)
    assert np.isclose(metrics["macro_f1"], 1.0)
    assert np.isclose(metrics["micro_f1"], 1.0)

def test_compute_overall_metrics_partial_match():
    # 3 samples, 2 classes
    probs = np.array([
        [0.9, 0.9],  # Pred: [1, 1]
        [0.4, 0.8],  # Pred: [0, 1]
        [0.2, 0.2]   # Pred: [0, 0]
    ])
    labels = np.array([
        [1, 0],
        [0, 1],
        [1, 0]
    ])
    
    metrics = compute_overall_metrics(probs, labels)
    
    # Exact match: only sample 2 is exact match ([0, 1] == [0, 1]) -> 1/3
    assert np.isclose(metrics["exact_match_accuracy"], 1/3)
    
    # Hamming accuracy:
    # Preds: [1, 1], [0, 1], [0, 0]
    # True:  [1, 0], [0, 1], [1, 0]
    # Sample 1: 1 match, 1 mismatch
    # Sample 2: 2 match
    # Sample 3: 1 match (class 1 is 0), 1 mismatch (class 0 is 0 instead of 1)
    # Total correct elements = 1 + 2 + 1 = 4 out of 6 -> 4/6 = 2/3
    assert np.isclose(metrics["hamming_accuracy"], 2/3)

def test_compute_per_class_metrics_edge_cases():
    # Edge case: class with no true positives
    probs = np.array([
        [0.9, 0.1],
        [0.4, 0.2]
    ])
    labels = np.array([
        [1, 0],
        [1, 0]
    ])
    label_names = ["class_0", "class_1"]
    
    metrics = compute_per_class_metrics(probs, labels, label_names)
    
    assert "class_1" in metrics
    c1 = metrics["class_1"]
    assert c1["tp"] == 0
    assert c1["fn"] == 0
    assert c1["fp"] == 0
    assert c1["tn"] == 2
    assert c1["precision"] == 0.0
    assert c1["recall"] == 0.0
    assert c1["f1"] == 0.0
    
    # verify class_0
    c0 = metrics["class_0"]
    assert c0["tp"] == 1
    assert c0["fp"] == 0
    assert c0["fn"] == 1
    assert c0["tn"] == 0

def test_compute_per_class_metrics_json_structure():
    probs = np.array([[0.8]])
    labels = np.array([[1]])
    label_names = ["class_0"]
    
    metrics = compute_per_class_metrics(probs, labels, label_names)
    
    # Ensure types are Python natives suitable for JSON serialization
    c0 = metrics["class_0"]
    assert isinstance(c0["tp"], int)
    assert isinstance(c0["fp"], int)
    assert isinstance(c0["fn"], int)
    assert isinstance(c0["tn"], int)
    assert isinstance(c0["precision"], float)
    assert isinstance(c0["recall"], float)
    assert isinstance(c0["f1"], float)
    assert isinstance(c0["brier_score"], float)
