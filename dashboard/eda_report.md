# GoEmotions Exploratory Data Analysis

## 1. Class Imbalance
The dataset is highly imbalanced. The most frequent class 'neutral' appears 14219 times, while the least frequent class 'grief' appears only 77 times.
**Training Implications:** We will likely need to use techniques like class weights in our BCEWithLogitsLoss, focal loss, or strategic resampling to prevent the model from ignoring rare classes.

## 2. Text Lengths
- Average word count: 12.8
- Average token count: 19.2
- 99th percentile token length: 38.0

**Training Implications:** Most texts are very short (e.g., Reddit comments). We can safely set the model's `max_length` to something small (like 64 or 128) instead of the BERT default of 512. This will significantly speed up both teacher fine-tuning and student distillation.

## 3. Label Noise / Multi-label Complexity
Number of examples with multiple labels: 7102 out of 43410 (16.4%).

**Training Implications:** The multi-label nature is prominent. Some labels might be inherently ambiguous or highly correlated (e.g., 'joy' and 'excitement'). The teacher model must output well-calibrated probabilities for the student to learn these nuanced soft labels during distillation.
