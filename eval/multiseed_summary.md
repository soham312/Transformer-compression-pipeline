# Multi-Seed Aggregation Results

**Distilled runs (n=5)**
- Macro F1: 0.3277 ± 0.0061
- Micro F1: 0.5251 ± 0.0034
- Hamming Acc: 0.9684 ± 0.0000

**Scratch runs (n=5)**
- Macro F1: 0.3251 ± 0.0054
- Micro F1: 0.5101 ± 0.0027
- Hamming Acc: 0.9676 ± 0.0002

## Statistical Significance (Welch's t-test)
Comparing Distilled vs Scratch on Macro F1 (n=5 vs n=5):
- Mean Difference (Distilled - Scratch): 0.0026 ± 0.0036
- t-statistic: 0.7279
- p-value: 0.4878
=> The difference is NOT statistically significant (p >= 0.05).

Comparing Distilled vs Scratch on Micro F1 (n=5 vs n=5):
- Mean Difference (Distilled - Scratch): 0.0151 ± 0.0020
- t-statistic: 7.6968
- p-value: 0.0001
=> The difference IS statistically significant (p < 0.05).

Comparing Distilled vs Scratch on Hamming Acc (n=5 vs n=5):
- Mean Difference (Distilled - Scratch): 0.0008 ± 0.0001
- t-statistic: 7.6135
- p-value: 0.0013
=> The difference IS statistically significant (p < 0.05).

