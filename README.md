# Transformer Compression Pipeline

> **Does knowledge distillation transfer dark knowledge on a severely imbalanced multi-label task, and how small can we quantize the result?**

This repository explores the limits of knowledge distillation and INT8 quantization applied to a custom from-scratch Transformer on the 28-class GoEmotions dataset. 

## 1. Headline Results

### The Trade-Off Surface (CPU Inference)

| Model Variant | Size (MB) | Macro F1 | Micro F1 | Hamming Acc | Median Latency (ms/seq) | Compression vs Teacher | Speedup vs Teacher |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher (BERT-base)** | 418.43 | 0.4110 | 0.5815 | 0.9708 | 12.90 | 1.0x | 1.00x |
| **Distilled Student** | 43.07 | 0.3269 | 0.5221 | 0.9685 | 0.69 | 9.7x | 18.83x |
| **Scratch Control** | 43.07 | 0.3291 | 0.5276 | 0.9680 | 0.64 | 9.7x | 20.32x |
| **PyTorch Dynamic INT8** | 36.39 | 0.3272 | 0.5225 | 0.9685 | 1.34 | 11.5x | 9.59x |
| **PyTorch Static INT8*** | 42.39 | 0.3244 | 0.5204 | 0.9685 | 0.65 | 9.9x | 19.74x |
| **ONNX INT8** | **10.85** | **0.3281** | 0.5234 | 0.9686 | **1.08** | **38.6x** | **11.90x** |

*(Benchmarked on M3 Pro CPU, batch size 32, max_length 64. Note: PyTorch static quantization was only applied to the classifier layer because `nn.TransformerEncoderLayer` crashes during eager-mode calibration. Teacher latency showed high variance across repetitions (median 12.90 ms, std 29.23 ms); speedup factors derived from it should be read as approximate.)*

Check out the interactive dashboard for visual analysis of the multi-seed variance, trade-off scatter plot, and training curves: `streamlit run dashboard/app.py`.

## 2. What this project set out to test

This project is structured around testing two specific hypotheses, backed by a strict scientific control group and multi-seed statistical significance testing:
1. **Does knowledge distillation transfer information beyond hard labels**, over and above what the same architecture learns on its own? 
2. **Does INT8 quantization preserve accuracy while meaningfully compressing the model?**

A control group (training from scratch on hard labels without the teacher) and formal Welch's t-tests were built specifically so these questions could be answered rigorously rather than assumed.

## 3. Dataset and Task

The task is classifying text into the **GoEmotions** dataset, which consists of 28 fine-grained emotion labels.
This is significantly harder than binary sentiment analysis for three reasons:
- **Multi-label:** ~16.4% of the samples carry multiple valid labels.
- **Severe Class Imbalance:** The "neutral" class dominates the dataset, while classes like "grief" are extraordinarily rare.
- **Short Texts:** The average sequence length is only ~19 tokens, offering very little context per sample.

## 4. Architecture and Method

* **Teacher:** A fine-tuned `BERT-base-uncased` model. It is 418 MB and achieves a 0.4110 macro F1 on the held-out test split.
* **Student:** A custom 12M-parameter Transformer (4 layers, 256 hidden size, 4 attention heads, 1024 intermediate size, mean pooling). Critically, it is **trained from random initialization** — not from a pretrained checkpoint like DistilBERT. This choice guarantees that any performance gain relative to the control group is strictly attributable to the teacher's distillation signal, as no pretrained knowledge leaks in.
* **Distillation Loss:** In standard distillation (Hinton et al. 2015), temperature scaling is applied to softmax logits to reveal the "dark knowledge" (the relative probabilities of incorrect classes). However, because GoEmotions is a multi-label task where classes are not mutually exclusive, softmax is mathematically incorrect. We adapted the loss by applying per-class binary KL divergence over independent sigmoids, scaling the logits by temperature $T$, and scaling the final KL loss by $T^2$ to ensure the gradient magnitudes match the hard-label BCE loss.
* **Control Group:** An identical student architecture, trained under an identical budget (15 epochs, patience 3, learning rate 5e-5, batch 32) using plain BCE on the hard labels. The *only* difference is the absence of the distillation loss.

## 5. Results

### Finding 1: Distillation helps on frequent classes, but not rare ones.
Across 5 seeds, compared to the scratch control, the distilled student achieved:
- **Micro F1:** +0.0151 (p=0.0001, highly significant)
- **Hamming Accuracy:** +0.0008 (p=0.0013, significant)
- **Macro F1:** +0.0026 (p=0.4878, **not significant**)

**The Mechanism:** Micro F1 and Hamming accuracy weight every label decision equally, meaning they are dominated by the most frequent classes in the dataset. The teacher is very strong on these frequent classes (e.g., gratitude 0.92, amusement 0.82, love 0.81), so distillation successfully transfers this knowledge. However, Macro F1 weights all 28 classes equally. The teacher scores exactly zero F1 on the five rarest classes (grief, pride, relief, embarrassment, nervousness). Distillation can only transfer what the teacher knows; it knows nothing about those rare classes, so the macro F1 metric sees no significant benefit.

### Finding 2: Framework choice completely dominated the quantization technique.
Applying INT8 dynamic quantization to the same model yielded drastically different results depending on the framework:
- **PyTorch Dynamic Quantization** gave a mere ~15.5% size reduction (43.07 MB -> 36.39 MB) and nearly a **2x latency regression** relative to the unquantized student (0.69 -> 1.34 ms/seq). PyTorch's `quantize_dynamic` only targets `nn.Linear` layers, ignoring the massive 30,522 x 256 `nn.Embedding` table which accounts for ~70% of the model's parameters. Furthermore, the runtime overhead of calculating activation ranges per-forward-pass heavily outweighed the INT8 matmul speedups on these small layers.
- **ONNX Runtime** quantized the embeddings as well, delivering a massive **38.6x compression** (418.43 MB -> 10.85 MB) relative to the teacher, and an 11.9x speedup. 

A paired bootstrap over 1000 resamples on the test set showed the ONNX INT8 accuracy difference was +0.0012, with a 95% CI of [0.0003, 0.0023]. This means there is **no measurable degradation** in accuracy from quantization.

### Finding 3: Validation loss and macro F1 disagree about when to stop.
Replicated across all 10 multi-seed runs, a clear divergence emerged: the scratch student's validation loss bottoms out at **epoch 5** and then begins to rise (overfitting), while its validation macro F1 continues climbing until **epochs 7–8**. 
Conversely, the distilled student's validation loss safely keeps improving alongside its F1 until epochs 7–8. 
This observation is consistent with a regularization effect, as we observed extended validation-loss improvement but did not test the mechanism directly. The consequence for the scratch model is severe: selecting checkpoints strictly based on validation loss systematically costs the scratch model roughly 0.04 macro F1 relative to its true peak.

## 6. Reproducing this project

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Execution Order
1. **EDA:** `python data/eda.py` (Note: Stage 2 scripts are in the repo)
2. **Teacher Training (GPU recommended):** `python models/train_teacher.py`
3. **Teacher Baseline & Calibration:** `python eval/evaluate_baseline.py`
4. **Student Training (GPU recommended):** `python distillation/train_student.py`
5. **Control Training (GPU recommended):** `python distillation/train_scratch.py`
6. **Quantization (CPU only):** `python quantization/dynamic_quantize.py`, `python quantization/static_quantize.py`, `python quantization/onnx_export.py`
7. **Multi-Model Benchmark (CPU only):** `python eval/benchmark_all.py`
8. **Statistical Significance:** `python distillation/run_multiseed.py` followed by `python eval/bootstrap_significance.py` and `python eval/aggregate_seeds.py`
9. **Dashboard:** `streamlit run dashboard/app.py`

*Note: All benchmark latencies in Stage 10/11 were measured strictly on CPU with a fixed thread count for comparability. PyTorch and ONNX INT8 quantized models cannot run on MPS or CUDA. Earlier mixed-device latency measurements in the git history are superseded by the `benchmark_results.json`.*

## 7. Design Decisions & Limitations

**Why BERT-base as a teacher and a from-scratch student?**
Using a from-scratch 12M parameter student, rather than a pretrained checkpoint like DistilBERT, isolates the distillation signal. If the student started with pretrained weights, it would be impossible to untangle which accuracy gains came from the teacher's dark knowledge versus the original pretraining corpus.

**Why per-class binary KL rather than softmax KL?**
GoEmotions is a multi-label task. Applying softmax assumes classes are mutually exclusive, forcing probabilities to sum to 1. This would aggressively penalize valid secondary labels. Modeling the classes as independent Bernoulli distributions via independent sigmoids and binary KL is mathematically correct for multi-label.

**Why a control group?**
Without the scratch student, we would observe the distilled model's metrics and assume distillation drove all of its performance. The control group proved that for macro F1, the vast majority of that performance was simply the architecture learning from the hard labels—making the distillation effect on macro F1 statistically indistinguishable from noise. However, the control group also proved that distillation *did* provide a highly significant boost to micro F1 (p=0.0001) and Hamming accuracy, confirming that knowledge transfer occurred, but only for frequent classes.

**Metrics**
Exact-match accuracy requires all 28 classes to be predicted perfectly, making it nearly useless for granular evaluation on this dataset. Hamming accuracy and Micro F1 correctly evaluate per-label decisions, while Macro F1 treats all classes equally to expose the rare-class collapse.

**Limitations & What I'd do next**
1. **Rare-Class Collapse:** The teacher's 0.4110 macro F1 sits below the ~0.46 published GoEmotions BERT baseline (Demszky et al. 2020). This is because class imbalance was identified in Stage 2 EDA, but was never addressed in Stage 3 training via `pos_weight` or focal loss. This rare-class collapse is directly implicated in Finding 1's macro F1 null. Fixing this via `pos_weight` is the lever most likely to convert the macro F1 null into a win.
2. **Threshold Tuning:** Thresholds were left at a fixed 0.5. Stage 4 calibration plots demonstrated that rare classes are systematically under-called and would benefit from per-class threshold tuning optimized on the validation set.
3. **Vocabulary Reduction:** The 30,522-token vocabulary embedding table accounts for ~70% of the student's parameters. Reducing the vocabulary to only the tokens actually present in the GoEmotions dataset would massively shrink the model footprint before quantization is even applied.
4. **Seed Count:** n=5 seeds provides relatively wide confidence intervals. More seeds would increase statistical power.

## 8. Repository Structure

* `models/` - Teacher model definition and training scripts.
* `distillation/` - Student model architecture, custom binary KL loss, training scripts (student + control), and multiseed orchestration.
* `quantization/` - Dynamic INT8 (PyTorch), Static INT8 (PyTorch), and ONNX INT8 export scripts.
* `eval/` - Evaluation scripts, benchmarking harness, statistical bootstrapping, t-tests, and log parsing.
* `tests/` - Comprehensive `pytest` suite ensuring numerical stability, shape correctness, and significance logic.
* `dashboard/` - Streamlit application for interactive visualization of the trade-off surface and training histories.
* `model_checkpoints/` - Saved model weights (gitignored).
