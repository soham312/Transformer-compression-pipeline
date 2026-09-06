# Comprehensive Multi-Model Benchmark Results

**Device:** CPU | **Threads:** 4 | **Batch Size:** 32

| Variant | Size (MB) | Size Reduction | Median Latency (ms) | Speedup | Macro F1 | F1 Retained | Micro F1 | Exact Match |
|---------|-----------|----------------|---------------------|---------|----------|-------------|----------|-------------|
| Teacher (BERT-base) | 418.43 | 1.00x | 12.90 | 1.00x | 0.4110 | 100.0% | 0.5815 | 0.4590 |
| Distilled Student | 43.07 | 9.71x | 0.69 | 18.83x | 0.3269 | 79.5% | 0.5221 | 0.3989 |
| Scratch Student | 43.07 | 9.71x | 0.64 | 20.32x | 0.3291 | 80.1% | 0.5276 | 0.4048 |
| Dynamic INT8 (PyTorch) | 36.39 | 11.50x | 1.34 | 9.59x | 0.3272 | 79.6% | 0.5225 | 0.3989 |
| Static INT8 (PyTorch)* | 42.39 | 9.87x | 0.65 | 19.74x | 0.3244 | 78.9% | 0.5204 | 0.3962 |
| Dynamic INT8 (ONNX) | 10.85 | 38.57x | 1.08 | 11.90x | 0.3281 | 79.8% | 0.5234 | 0.3999 |

* *Note on Static INT8 (PyTorch): Static quantization was applied to the classifier only due to eager-mode bugs in PyTorch's TransformerEncoderLayer. Encoder and embeddings remain FP32.*