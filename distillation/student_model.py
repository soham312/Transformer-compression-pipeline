import torch
import torch.nn as nn
import math

class StudentTransformer(nn.Module):
    """
    Custom lightweight student transformer model for knowledge distillation.
    
    Targeting roughly:
    - 4 layers
    - 256 hidden dimension
    - 4 attention heads
    - ~11-15M parameters
    """
    def __init__(self, 
                 vocab_size=30522, 
                 max_position_embeddings=512, 
                 hidden_size=256, 
                 num_hidden_layers=4, 
                 num_attention_heads=4,
                 intermediate_size=1024,
                 num_labels=28,
                 dropout_prob=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 1. Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
        # 2. Transformer Encoder Stack
        # We use batch_first=True so input shape is (batch_size, sequence_length, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        
        # 3. Classification Head
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize weights randomly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Random weight initialization similar to BERT."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
            attention_mask: Tensor of shape (batch_size, sequence_length)
        """
        seq_length = input_ids.size(1)
        
        # Create position IDs (0, 1, 2, ..., seq_length-1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Compute embeddings
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # TransformerEncoder expects src_key_padding_mask where True means "ignore this position"
        # HuggingFace attention_mask is 1 for "keep", 0 for "ignore".
        # So we invert the mask for PyTorch's TransformerEncoder.
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
            
        # Pass through the transformer encoder stack
        encoder_outputs = self.encoder(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Pooling: We use Mean Pooling (average over non-masked tokens)
        # Using mean pooling generally works well for text classification when [CLS] isn't specifically trained
        if attention_mask is not None:
            # Expand attention_mask to match hidden states: (batch, seq_len, 1)
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_outputs.size()).float()
            sum_embeddings = torch.sum(encoder_outputs * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = encoder_outputs.mean(dim=1)
            
        pooled_output = self.dropout(pooled_output)
        
        # Compute logits for 28 classes
        logits = self.classifier(pooled_output)
        
        return logits
        
    def count_parameters(self):
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
