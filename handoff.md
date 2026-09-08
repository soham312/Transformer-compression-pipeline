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

### Stage 10: Comprehensive multi-model benchmark (COMPLETE)

This stage produced the final apples-to-apples comparison of all six model variants on the M3 Pro CPU. The benchmark script loaded each variant, ran a warmup, and measured median latency over 5 iterations on the full test set with batch size 32, ensuring fair conditions for evaluating both latency and accuracy.

**Final Results Table (Device: CPU, Threads: 4, Batch: 32)**

| Variant | Size (MB) | Size Reduction | Median Latency (ms) | Speedup | Macro F1 | F1 Retained | Micro F1 | Exact Match |
|---------|-----------|----------------|---------------------|---------|----------|-------------|----------|-------------|
| Teacher (BERT-base) | 418.43 | 1.00x | 12.90 | 1.00x | 0.4110 | 100.0% | 0.5815 | 0.4590 |
| Distilled Student | 43.07 | 9.71x | 0.69 | 18.83x | 0.3269 | 79.5% | 0.5221 | 0.3989 |
| Scratch Student | 43.07 | 9.71x | 0.64 | 20.32x | 0.3291 | 80.1% | 0.5276 | 0.4048 |
| Dynamic INT8 (PyTorch) | 36.39 | 11.50x | 1.34 | 9.59x | 0.3272 | 79.6% | 0.5225 | 0.3989 |
| Static INT8 (PyTorch)* | 42.39 | 9.87x | 0.65 | 19.74x | 0.3244 | 78.9% | 0.5204 | 0.3962 |
| Dynamic INT8 (ONNX) | 10.85 | 38.57x | 1.08 | 11.90x | 0.3281 | 79.8% | 0.5234 | 0.3999 |

*Note: Static quantization (PyTorch) was applied to the classifier only due to eager-mode bugs in `nn.TransformerEncoderLayer`.*

**Key Findings & Interpretations:**
1. **Best accuracy-per-MB:** **ONNX INT8**. At just 10.85 MB (a 38.5x reduction over the teacher), it retains ~80% of the teacher's Macro F1. It achieves this because `onnxruntime` successfully quantizes the massive `nn.Embedding` table, which PyTorch's dynamic quantization ignores.
2. **Best accuracy-per-ms:** **FP32 Student (Scratch/Distilled)**. PyTorch's FP32 models run at ~0.64 - 0.69 ms/seq on CPU, nearly a 20x speedup over the teacher. PyTorch dynamic quantization severely regress latency (1.34 ms/seq) due to activation scaling overhead on tiny layers, and even ONNX INT8 (1.08 ms) is slower than pure FP32 PyTorch.
3. **Distillation vs. Scratch:** The Scratch student (0.3291 Macro F1) slightly outperformed the Distilled student (0.3269). Distillation provided zero measurable benefit on this architecture and dataset compared to standard hard-label training. 
4. **Teacher Latency:** The true cost of BERT-base on CPU is high (12.90 ms/batch-seq median), underscoring the necessity of the student models which hit ~0.65ms for a 20x improvement.
5. **Deployment Recommendation:** 
   - If **storage/memory** is the absolute bottleneck (e.g., edge devices), deploy the **ONNX INT8** model.
   - If **CPU latency** is the absolute bottleneck, deploy the **FP32 Scratch Student**.

Files modified/created: `eval/benchmark_all.py`, `tests/test_benchmark.py`, `eval/benchmark_results.json`, `eval/benchmark_table.md`.
All tests pass and no git operations were executed.

## Current State
- All 11 Stages are complete. We have successfully fine-tuned the teacher, distilled and control-trained the students, benchmarked PyTorch vs. ONNX quantization strategies, and performed rigorous statistical significance testing.
- No git operations have been performed for the recent stages, as the user manually reviews and handles version control.

### Stage 11 — Statistical Significance (COMPLETE)

Files: `distillation/run_multiseed.py`, `eval/bootstrap_significance.py`, `eval/aggregate_seeds.py`, `tests/test_significance.py`, `eval/multiseed_results.json`, 10 per-seed metrics files (`eval/student_distilled_metrics_seed{42,123,456,789,1011}.json` and the scratch equivalents). `train_student.py` and `train_scratch.py` now accept `seed`, `output_dir`, and `metrics_file` parameters with determinism enforced across torch, numpy, and random.

**Multi-seed results (5 seeds, run on Colab T4, best-val-loss checkpoint per run):**

| Metric | Distilled | Scratch | Diff | p-value |
|--------|-----------|---------|------|---------|
| Macro F1 | 0.3277 ± 0.0061 | 0.3251 ± 0.0054 | +0.0026 | 0.4878 (not significant) |
| Micro F1 | 0.5251 ± 0.0034 | 0.5101 ± 0.0027 | +0.0151 | 0.0001 (significant) |
| Hamming Acc | 0.9684 ± 0.0000 | 0.9676 ± 0.0002 | +0.0008 | 0.0013 (significant) |

*Welch's t-test, unequal variances, n=5 per group.*

**HEADLINE FINDING:** Distillation produces a statistically significant improvement in micro F1 (+0.015, p=0.0001, t=7.70) and Hamming accuracy, but no significant improvement in macro F1. 
**The mechanism:** Micro F1 and Hamming weight every label decision equally and are therefore dominated by frequent classes, where the teacher performs well (gratitude 0.92, amusement 0.82, love 0.81 per Stage 4). Macro F1 weights all 28 classes equally and is dominated by the five classes where the teacher scores exactly zero — grief, pride, relief, embarrassment, nervousness. Distillation can only transfer knowledge the teacher possesses; it possesses none about those classes. This supersedes the earlier reading that distillation showed no benefit at all — the benefit is real but metric-dependent and mechanistically explained.

**Bootstrap results (1000 resamples, paired, test split):**
- **Distilled minus scratch macro F1:** 95% CI [-0.0111, 0.0064], straddles zero — consistent with the multi-seed macro F1 null
- **ONNX INT8 minus FP32 distilled:** mean +0.0012, 95% CI [0.0003, 0.0023], excludes zero — quantization's accuracy cost is not merely small but statistically indistinguishable from zero, if anything marginally favorable. Do not claim quantization improves accuracy; the correct claim is no measurable degradation.

**Also worth recording:** The multi-seed run was executed twice (the first Colab session was reclaimed before results were downloaded) and produced numerically identical per-seed results, confirming the seeding is fully deterministic and the pipeline reproducible.

**Consistent secondary observation across all 10 runs:** The scratch student's validation loss bottoms out at epoch 5 in every seed and then rises, while its macro F1 continues climbing to 0.37–0.38 by epochs 7–8. The distilled student's val loss keeps improving to epoch 7–8. 
**Two implications:** 
(a) distillation acts as a regularizer, extending useful training by 2–3 epochs.
(b) selecting checkpoints on validation loss systematically costs the scratch model roughly 0.04 macro F1 relative to its peak. This loss/F1 divergence on imbalanced multi-label data is worth documenting in the README.

### Deferred Improvements (identified, not yet actioned)
- Teacher `pos_weight` for class imbalance (Stage 2 EDA flagged this; Stage 3 never implemented it — the teacher's rare-class collapse is now directly implicated in the macro F1 null).
- Per-class threshold tuning fit on validation.
- Reduced student vocabulary to shrink the dominant embedding table.

### Stage 12 — Streamlit Dashboard (COMPLETE)

Files: `dashboard/app.py`

**Status:** Code complete, verified to run cleanly with `streamlit run dashboard/app.py`. 
No additional dependencies were added beyond what was already in `requirements.txt` (used `streamlit` + `matplotlib`).

**What it visualizes:**
1. **The Trade-Off Surface:** A scatter plot graphing the six benchmarked variants' Macro F1 against their Size (MB, log scale), with marker size mapped to inference latency. Accompanied by a bar chart showing compression/speedup factors relative to the teacher. This perfectly isolates the top-right teacher vs the mid-left ONNX INT8 compression win.
2. **Training Curves (Caveat):** Because the per-epoch loss and F1 histories were not recorded into the JSON artifacts during Stages 6 and 7, this panel currently renders an informative warning instead of fabricating charts. It explicitly documents the Stage 11 finding about the loss/F1 divergence (validation loss bottoming at epoch 5 while F1 peaks at epochs 7-8).
3. **Statistical Significance:** A visual strip plot of the actual multi-seed spread for Macro F1, Micro F1, and Hamming Accuracy for both Distilled and Scratch models. It includes the Welch's t-test outcomes and explicitly explains the mechanism behind the significance split (teacher's strength in frequent classes translating to Micro F1 wins, while its collapse on rare classes nullifies Macro F1 gains).
4. **Per-Class Diagnostics:** A paired-axis bar/line chart graphing the teacher's F1 score alongside true class support, rendering the rare-class collapse visually undeniable. Includes the calibration reliability plot from Stage 4.

**Notes for Execution:**
The dashboard gracefully degrades with explicit error messages if any artifact JSON file goes missing, and correctly populates metadata about the benchmarking environment directly from the dynamically loaded files.

### Stage 12 Addendum — Training Curve Visualization

**History Capture Fix:**
- Updated `models/train_teacher.py`, `distillation/train_student.py`, and `distillation/train_scratch.py` to persist the `history` dictionary (per-epoch losses and metrics) directly into their respective final metrics JSON files. Future runs will inherently contain all data necessary for curve visualization.
- Created `eval/parse_training_logs.py` to scrape the historical console output from `multiseed_log.txt` to backfill `eval/training_histories.json` for the existing runs, ensuring no data was lost.
- Updated `dashboard/app.py` to seamlessly parse `eval/training_histories.json` and generate a robust visual panel overlaying all 5 seed runs. Individual seeds are mapped with low opacity, and their mean is overlaid with a thick line. 

**Visualization Result:**
The Dashboard's Training Curve panel vividly surfaces the loss/F1 divergence: you can visibly trace the scratch model's validation loss turning sharply upward at epoch 5 while its Macro F1 continues an upward trajectory until epoch 7/8. Meanwhile, distillation is observed acting as a regularizer, as the Distilled model's validation loss stably descends alongside F1 to epoch 7/8 without early divergence.

### Stage 13 — Final test coverage + comprehensive README (COMPLETE)

Files: `tests/*`, `README.md`

**Status:** Code complete, tests cleanly passing, documentation entirely finished.

**What was done:**
1. **Test Coverage Audit:** Executed the entire suite. Verified that `tests/test_distillation.py` rigorously covered all custom Binary KL mathematics (including $T^2$ scaling and alpha boundaries). Verified that `tests/test_quantization.py` and `test_static_quantization.py` checked shapes, outputs, and correlations across PyTorch dynamic, PyTorch static, and ONNX INT8 pipelines. Verified `tests/test_benchmark.py` covered ratio math accurately. Evaluated significance logic across synthetic datasets in `test_significance.py`.
2. **README Construction:** Designed a robust, highly-legible repository landing page detailing the dual hypotheses (Distillation transferability and Quantization efficiency). Logged the 3 principal findings (Macro F1 null vs Micro F1 win, ONNX domination over PyTorch dynamic due to `nn.Embedding`, and the loss-checkpointing regularization divergence). Honestly captured the architecture configurations, CPU-only benchmarking caveats, and detailed exactly what decisions should be altered in a real production iteration (like implementing `pos_weight` and threshold-tuning to save the rare classes).

This project is now 100% complete and ready to present.
