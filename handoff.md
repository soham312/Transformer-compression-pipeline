# Project Handoff: Transformer Fine-Tuning & Compression Pipeline

This document summarizes the progress made so far on the Transformer compression pipeline.

## Overview
We are building a pipeline to compress a Transformer model (BERT-base-uncased) fine-tuned on the `GoEmotions` dataset. The goal is to dramatically reduce the model size (target < 100MB) and inference latency (target < 1.5ms) while retaining at least 95% of the teacher's macro F1 score (target > 0.38).

## Completed Stages

### Stage 1: Scaffold & Data Loading
- Set up the repository structure.
- Selected the `google-research-datasets/go_emotions` dataset (28 fine-grained emotion classes, multi-label classification).
- Built the data loading and tokenization pipeline in `data/dataset.py`.
- **Note:** `torchvision.io` was stubbed in `dataset.py` to prevent `ModuleNotFoundError` when loading datasets without `torchvision` installed.

### Stage 2: Exploratory Data Analysis (EDA)
- Performed EDA (reports and charts available in the `dashboard/` directory).
- Identified severe class imbalance (e.g., `neutral` is highly frequent, `grief` is extremely rare).
- Optimized the maximum sequence length to `max_length=64` to speed up training, since most Reddit comments are short.

### Stage 3: Teacher Fine-Tuning
- Created `models/train_teacher.py` for fine-tuning `bert-base-uncased`.
- Added support for Apple Silicon (`mps`) backend, falling back to CUDA or CPU.
- The teacher model was trained and the best checkpoint was saved to `model_checkpoints/teacher/`.
- Generated validation metrics in `eval/teacher_metrics.json`.

### Stage 4: Baseline evaluation report
- Full test-set eval of the fine-tuned teacher (held-out `test` split of GoEmotions)
- Files: `eval/evaluate_baseline.py`, `eval/teacher_baseline.json`, `eval/calibration_reliability.png`, `tests/test_eval.py`
- Also fixed a torchvision import bug in `data/dataset.py`
- Baseline metrics (test set):
  - **Macro F1:** 0.4110, **Micro F1:** 0.5815
  - **Hamming acc:** 0.9708, **Exact-match acc:** 0.4590
  - **Model size:** 418.43 MB, **avg latency:** 5.43 ms/seq on M3 Pro MPS
- Per-class precision/recall/F1/TP/FP/FN/TN/Brier for all 28 emotions in the JSON
- Rare classes (grief, pride, relief, embarrassment, nervousness) have F1 = 0.0 — matches original Demszky et al. 2020 BERT baseline behavior
- All 8 tests pass (`test_dataset`, `test_eda`, `test_eval` ×4, `test_train_teacher` ×2)

### Stage 5: Distillation loss & custom student architecture
- Implemented `distillation/loss.py`: A custom `DistillationLoss` combining standard BCE hard-label loss and scaled binary KL divergence (temperature-scaled) to handle the multi-label nature of GoEmotions.
- Designed `distillation/student_model.py`: A `StudentTransformer` built from PyTorch primitives targeting ~12M parameters (4 layers, 256 hidden dimension, 4 attention heads). Initialized from scratch.
- Added comprehensive unit tests in `tests/test_distillation.py` covering model architecture, parameter count, and loss function correctness (including manual KL divergence calculations).
- All 17 unit tests now pass locally.

### Stage 6: Distilled student trained (COMPLETE)
- Trained on Colab T4 using cached teacher logits (precomputed once, not per-epoch).
- Budget: 15 epochs, early stopping fired at epoch 10 (patience=3); best checkpoint is epoch 7.
- Final metrics (validation split, corresponding to the saved best-val-loss checkpoint):
  - **best_val_loss:** 0.0729
  - **macro F1:** 0.3277
  - **micro F1:** 0.5266
  - **Hamming accuracy:** 0.9685
  - **model size:** 43.07 MB
  - **avg latency:** 0.16 ms/seq (measured on T4 — note: teacher's 5.43 ms/seq was measured on M3 Pro MPS, so these are not directly comparable; Stage 10 must re-measure all variants on the same device)
- Hyperparameters: `alpha=0.5`, `temperature=4.0`, `lr=5e-5`, `batch_size=32`, `max_length=64`.
- Compression achieved: 418.43 MB → 43.07 MB (9.7× smaller), retaining ~82% of teacher macro F1 (0.4002 → 0.3277).
- Files: `distillation/train_student.py`, `tests/test_train_student.py`, `eval/student_distilled_metrics.json`, `model_checkpoints/student_distilled/` (weights + `student_config.json`, also backed up to Google Drive).
- **Note:** An earlier 5-epoch run reached only macro F1 0.1876 — the student was clearly undertrained at that budget. That run is archived in Drive as `student_distilled_5ep` / `student_distilled_metrics_5ep.json` for comparison in the README.
- **Important for Stage 7:** The from-scratch control student MUST use the identical budget (15 epochs, patience=3, lr=5e-5, batch=32) and the same `StudentTransformer` architecture, or the distillation-vs-scratch comparison is invalid.
- **Fixes/Lessons:** `models/train_teacher.py` now takes a configurable `metrics_file` parameter, and both `tests/test_train_teacher.py` and `tests/test_train_student.py` write to pytest `tmp_path` — earlier, smoke tests were silently overwriting real training results in `eval/`.

### Stage 7: Control group (student trained from scratch) (COMPLETE)
- Trained on Colab T4 without any teacher or distillation loss, using plain `BCEWithLogitsLoss`.
- Architecture and budget verified absolutely identical to the distilled student (`StudentTransformer` config, 15 epochs, patience=3, lr=5e-5, batch_size=32, max_length=64).
- Budget results: 15-epoch limit, early stopping fired at epoch 8, best checkpoint selected at epoch 5.
- Final metrics (validation split, best checkpoint):
  - **macro F1:** 0.3233
  - **micro F1:** 0.5234
  - **Hamming accuracy:** 0.9674
  - **model size:** 43.07 MB
  - **avg latency:** 0.15 ms/seq on T4
- **Key Finding:** The distilled student (0.3277) and the from-scratch control student (0.3233) differ by only 0.0044 macro F1 — a gap that falls within expected noise for a single seed. Under matched architecture and budget, distillation showed no measurable benefit over standard hard-label training.
- **Candidate Explanations:**
  1. The teacher is weak (0.4002 macro F1 vs the ~0.46 published baseline), leaving little advantage to transfer.
  2. Multi-label sigmoid outputs inherently carry less cross-class "dark knowledge" than the softmax distributions assumed in standard Hinton-style distillation.
- **Note:** The scratch model reached a higher peak macro F1 (0.3757 at epoch 7) than the distilled model ever did, but that checkpoint was discarded because validation loss had already started rising. A loss/F1 divergence was observed in both student models.
- **Stage 11 Imperative:** The significance testing in Stage 11 is now load-bearing. It will determine whether the 0.0044 F1 gap is a real (but small) signal or purely statistical noise across seeds.

### Stage 8: Dynamic quantization (COMPLETE)
- Run locally on M3 Pro CPU.
- Files: `quantization/dynamic_quantize.py`, `tests/test_quantization.py`, `eval/student_dynamic_quant_metrics.json`.
- Applied `quantize_dynamic` (`qint8`, `nn.Linear`) to the distilled student, using the `qnnpack` backend.
- **Results:**
  - **Size:** 43.07 MB → 37.29 MB (only ~13% reduction)
  - **Median latency:** 0.84 ms/seq → 1.32 ms/seq (**57% REGRESSION**)
  - **Macro F1:** 0.3269 → 0.3272 (virtually unchanged)
- **Causes for poor performance:**
  1. The model's footprint is dominated by the 30,522 × 256 embedding table (~31 MB), which dynamic quantization does not touch.
  2. The per-forward-pass activation range computation costs more time than the INT8 matmul saves on these small Linear layers.
- **Verdict:** Dynamic quantization is not worth it on this architecture.

### Stage 9: Static quantization + ONNX export (COMPLETE)
- Run locally on M3 Pro CPU.
- Files: `quantization/static_quantize.py`, `quantization/onnx_export.py`, `tests/test_static_quantization.py`, `eval/quantization_comparison.json`.
- Unified CPU benchmark (batch 32, seq_len 64, test split):

| Variant | Size | Median latency | Macro F1 |
|---------|------|----------------|----------|
| PyTorch Float | 43.07 MB | 0.85 ms | 0.3269 |
| PyTorch Dynamic INT8 | 37.29 MB | 1.16 ms | 0.3272 |
| PyTorch Static INT8 | 43.29 MB | 0.69 ms | 0.3244 |
| ONNX Float | 42.52 MB | 1.25 ms | 0.3269 |
| ONNX Dynamic INT8 | 10.85 MB | 1.19 ms | 0.3281 |

- **Key finding:** ONNX INT8 achieves a **4× size reduction** at no accuracy cost. This is because ONNX Runtime's quantizer successfully compresses `nn.Embedding`, whereas PyTorch's eager mode quantizer only targets `nn.Linear`. Since embeddings constitute ~70% of this model, the framework choice mattered significantly more than the quantization technique itself.
- **Important caveat:** PyTorch static quantization was applied to the `classifier` layer ONLY. The `nn.TransformerEncoderLayer` crashes during eager-mode calibration (it casts boolean padding masks to float, breaking `masked_fill_`), so the encoder and embeddings had to stay in FP32. Its 0.69 ms latency is therefore not a completely fair test of full-model static quantization — and its 43.29 MB size (slightly larger than baseline due to added scale parameters) confirms almost nothing was quantized.
- **Also note:** Dynamic-quantized latency measured 1.32 ms in Stage 8 and 1.16 ms in Stage 9 on the same model and device — roughly 14% run-to-run variance in CPU timing. Stage 10 and 11 will need enough repetitions to distinguish real differences from this noise.

## Current State
- Stages 1-9 are complete. We have successfully fine-tuned the teacher, distilled and control-trained the students, and benchmarked PyTorch vs. ONNX quantization strategies.
- No git operations have been performed for the recent stages, as the user manually reviews and handles version control.

## Next Steps
- Proceed to the remaining stages (Stage 10 unified testing and Stage 11 statistical significance checks).
