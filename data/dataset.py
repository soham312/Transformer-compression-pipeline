"""
Data loading and tokenization pipeline for the Transformer Fine-Tuning & Compression project.

Task Selection: GoEmotions (Multi-label Emotion Classification)
Dataset: 'go_emotions' from Hugging Face Datasets.

Justification:
Unlike basic binary or 3-class sentiment analysis (positive/negative/neutral), 
GoEmotions involves 28 fine-grained emotion categories (e.g., 'admiration', 'amusement', 'anger', 'annoyance').
This task is significantly harder due to:
1. Multi-label nature: A single text can express multiple emotions simultaneously.
2. Fine-grained classes: Differentiating between closely related emotions (e.g., 'joy' vs 'amusement') 
   requires the model to understand deeper semantic nuances.
3. Class imbalance: Some emotions are much more frequent than others, making it harder for the model 
   to learn representations for the rare classes.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    import torchvision.io as _tv_io
    if not hasattr(_tv_io, "VideoReader"):
        class _StubVideoReader:  # unused stub; only needed so `datasets` internals can import it
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Video reading isn't used in this project.")
        _tv_io.VideoReader = _StubVideoReader
except ImportError:
    pass

def load_and_tokenize_data(model_name="bert-base-uncased", max_length=128, split="train", num_samples=None):
    """
    Loads the go_emotions dataset and tokenizes the text.
    
    Args:
        model_name (str): Name of the pretrained model tokenizer to use.
        max_length (int): Maximum sequence length for tokenization.
        split (str): Dataset split to load ('train', 'validation', 'test').
        num_samples (int, optional): Number of samples to take (useful for testing/dev).
        
    Returns:
        tuple: (Hugging Face Dataset with tokenized inputs and multi-hot labels, tokenizer)
    """
    # Load dataset
    dataset = load_dataset("google-research-datasets/go_emotions", split=split)
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_and_encode(examples):
        # Tokenize text
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        # Convert list of label indices to multi-hot encoded vectors
        num_labels = 28
        multi_hot_labels = []
        for labels_list in examples["labels"]:
            # Create a vector of zeros
            label_vec = [0.0] * num_labels
            for label_idx in labels_list:
                label_vec[label_idx] = 1.0
            multi_hot_labels.append(label_vec)
            
        tokenized["labels"] = multi_hot_labels
        return tokenized
        
    # Apply tokenization and encoding
    tokenized_dataset = dataset.map(tokenize_and_encode, batched=True, remove_columns=["id", "text"])
    
    # Set format to PyTorch tensors for relevant columns
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    return tokenized_dataset, tokenizer
