"""
Exploratory Data Analysis for GoEmotions dataset.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

def perform_eda(output_dir="dashboard", split="train", subsample=None):
    """
    Performs Exploratory Data Analysis on the GoEmotions dataset.
    Generates plots for class distribution and text/token lengths,
    and documents data quality issues.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("go_emotions", split=split)
    if subsample:
        dataset = dataset.select(range(min(subsample, len(dataset))))
    
    # 1. Class Distribution Analysis
    label_names = dataset.features["labels"].feature.names
    num_labels = len(label_names)
    
    label_counts = np.zeros(num_labels)
    for labels in dataset["labels"]:
        for label_idx in labels:
            label_counts[label_idx] += 1
            
    # Plot class distribution
    plt.figure(figsize=(14, 8))
    sorted_indices = np.argsort(label_counts)[::-1]
    sorted_labels = [label_names[i] for i in sorted_indices]
    sorted_counts = label_counts[sorted_indices]
    
    plt.bar(range(num_labels), sorted_counts)
    plt.xticks(range(num_labels), sorted_labels, rotation=45, ha='right')
    plt.title("Class Distribution in GoEmotions")
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()
    
    # 2. Text Length Distribution (Words vs Tokens)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    word_lengths = []
    
    for text in dataset["text"]:
        # Simple whitespace split for word count
        word_lengths.append(len(text.split()))
        
    # Tokenize the entire dataset for token lengths
    def get_token_lengths(examples):
        tokens = tokenizer(examples["text"], truncation=False)
        return {"token_length": [len(t) for t in tokens["input_ids"]]}
        
    tokenized_dataset = dataset.map(get_token_lengths, batched=True)
    token_lengths = tokenized_dataset["token_length"]
    
    # Plot length distributions
    plt.figure(figsize=(10, 5))
    plt.hist(word_lengths, bins=50, alpha=0.5, label='Words (Whitespace split)')
    plt.hist(token_lengths, bins=50, alpha=0.5, label='Tokens (BERT tokenizer)')
    plt.title("Text Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "length_distribution.png"))
    plt.close()
    
    # 3. Document Data Quality Issues
    num_multi_label = sum(1 for labels in dataset["labels"] if len(labels) > 1)
    percent_multi_label = (num_multi_label / len(dataset)) * 100
    
    report_path = os.path.join(output_dir, "eda_report.md")
    with open(report_path, "w") as f:
        f.write("# GoEmotions Exploratory Data Analysis\n\n")
        f.write("## 1. Class Imbalance\n")
        f.write("The dataset is highly imbalanced. ")
        f.write(f"The most frequent class '{sorted_labels[0]}' appears {int(sorted_counts[0])} times, ")
        f.write(f"while the least frequent class '{sorted_labels[-1]}' appears only {int(sorted_counts[-1])} times.\n")
        f.write("**Training Implications:** We will likely need to use techniques like class weights in our ")
        f.write("BCEWithLogitsLoss, focal loss, or strategic resampling to prevent the model from ignoring rare classes.\n\n")
        
        f.write("## 2. Text Lengths\n")
        f.write(f"- Average word count: {np.mean(word_lengths):.1f}\n")
        f.write(f"- Average token count: {np.mean(token_lengths):.1f}\n")
        f.write(f"- 99th percentile token length: {np.percentile(token_lengths, 99):.1f}\n\n")
        f.write("**Training Implications:** Most texts are very short (e.g., Reddit comments). ")
        f.write("We can safely set the model's `max_length` to something small (like 64 or 128) instead of ")
        f.write("the BERT default of 512. This will significantly speed up both teacher fine-tuning and student distillation.\n\n")
        
        f.write("## 3. Label Noise / Multi-label Complexity\n")
        f.write(f"Number of examples with multiple labels: {num_multi_label} out of {len(dataset)} ")
        f.write(f"({percent_multi_label:.1f}%).\n\n")
        f.write("**Training Implications:** The multi-label nature is prominent. Some labels might be inherently ")
        f.write("ambiguous or highly correlated (e.g., 'joy' and 'excitement'). The teacher model must output ")
        f.write("well-calibrated probabilities for the student to learn these nuanced soft labels during distillation.\n")
        
    return {
        "most_frequent_class": sorted_labels[0],
        "least_frequent_class": sorted_labels[-1],
        "mean_token_length": np.mean(token_lengths),
        "percent_multi_label": percent_multi_label,
        "total_samples": len(dataset)
    }

if __name__ == "__main__":
    print("Running EDA on GoEmotions dataset...")
    results = perform_eda()
    print("EDA Complete. Results saved to the 'dashboard' directory.")
    print("Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")
