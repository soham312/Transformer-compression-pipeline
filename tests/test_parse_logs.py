import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eval.parse_training_logs import parse_logs

def test_parse_logs():
    sample_log = """
Training distilled student for seed 42...
Loading tokenizer and datasets...
Epoch 1/15
Train Loss: 0.6296 | Val Loss: 0.1898
Val Macro F1: 0.0000 | Val Micro F1: 0.0000 | Val Hamming Acc: 0.9580
Validation loss improved. Saving best student model...

Epoch 2/15
Train Loss: 0.1889 | Val Loss: 0.1853
Val Macro F1: 0.0000 | Val Micro F1: 0.0000 | Val Hamming Acc: 0.9580

Epoch 3/15
Train Loss: 0.1271 | Val Loss: 0.0892
Val Macro F1: 0.1446 | Val Micro F1: 0.4416 | Val Hamming Acc: 0.9661
Validation loss improved. Saving best student model...
Early stopping triggered.

Training scratch student for seed 42...
Epoch 1/15
Train Loss: 0.5000 | Val Loss: 0.4000
Val Macro F1: 0.1000 | Val Micro F1: 0.1000 | Val Hamming Acc: 0.9000
Validation loss improved. Saving best student model...
    """
    
    histories = parse_logs(sample_log)
    
    # Check distilled 42
    assert "distilled" in histories
    assert "scratch" in histories
    assert "42" in histories["distilled"]
    assert "42" in histories["scratch"]
    
    dist_42 = histories["distilled"]["42"]
    assert dist_42["train_loss"] == [0.6296, 0.1889, 0.1271]
    assert dist_42["val_loss"] == [0.1898, 0.1853, 0.0892]
    assert dist_42["val_macro_f1"] == [0.0, 0.0, 0.1446]
    assert dist_42["selected_epoch"] == 3
    
    scratch_42 = histories["scratch"]["42"]
    assert scratch_42["train_loss"] == [0.5000]
    assert scratch_42["val_loss"] == [0.4000]
    assert scratch_42["selected_epoch"] == 1
