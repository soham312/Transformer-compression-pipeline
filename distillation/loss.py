import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Distillation Loss for Multi-Label Classification.
    
    This loss combines two objectives:
    1. Hard-label loss: Standard Binary Cross-Entropy (BCE) with logits, comparing 
       the student's predictions against the ground truth labels.
    2. Soft-label loss: Kullback-Leibler (KL) divergence between the student's 
       and teacher's logits, with temperature softening.

    Multi-label considerations:
    Standard knowledge distillation applies a softmax over classes, treating the 
    problem as a single-label multinomial distribution. Since GoEmotions is a 
    multi-label dataset (an input can have multiple emotions simultaneously), we 
    must instead treat each class as an independent binary distribution [p, 1-p].
    We compute the binary KL divergence for each class separately and average them.
    
    Args:
        alpha (float): Weight for the soft-label loss. The hard-label loss gets weight (1 - alpha).
        temperature (float): Temperature scaling factor (T) applied to logits before computing
                             probabilities. Higher T "softens" the probability distribution,
                             revealing the teacher's dark knowledge (relative probabilities of 
                             non-target classes).
    """
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        # We use BCEWithLogitsLoss because GoEmotions is multi-label.
        # It handles the sigmoid and cross-entropy in a single numerically stable step.
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Computes the combined distillation loss.
        
        Args:
            student_logits: Tensor of shape (batch_size, num_classes)
            teacher_logits: Tensor of shape (batch_size, num_classes)
            true_labels: FloatTensor of shape (batch_size, num_classes) with 0/1 labels
            
        Returns:
            Scalar loss tensor.
        """
        # 1. Hard-label loss (Standard BCE)
        # Compares student outputs directly to the ground truth targets.
        hard_loss = self.bce_loss(student_logits, true_labels)

        # 2. Soft-label loss (Binary KL Divergence with Temperature)
        # We scale both logits by the temperature T.
        s_logits_t = student_logits / self.temperature
        t_logits_t = teacher_logits / self.temperature

        # For a binary distribution [p, 1-p], the KL divergence KL(P || Q) is:
        # P(1) * log(P(1)/Q(1)) + P(0) * log(P(0)/Q(0))
        # 
        # Using logarithms for numerical stability:
        # P(1) = sigmoid(T_logits), P(0) = 1 - sigmoid(T_logits) = sigmoid(-T_logits)
        # log Q(1) = logsigmoid(S_logits), log Q(0) = logsigmoid(-S_logits)
        
        # Teacher probabilities (P)
        t_probs_pos = torch.sigmoid(t_logits_t)
        t_probs_neg = torch.sigmoid(-t_logits_t) # equivalent to 1 - t_probs_pos
        
        # Teacher log probabilities (log P)
        t_logprobs_pos = F.logsigmoid(t_logits_t)
        t_logprobs_neg = F.logsigmoid(-t_logits_t)
        
        # Student log probabilities (log Q)
        s_logprobs_pos = F.logsigmoid(s_logits_t)
        s_logprobs_neg = F.logsigmoid(-s_logits_t)

        # Compute KL components for positive and negative cases
        # kl_pos = P(1) * (log P(1) - log Q(1))
        kl_pos = t_probs_pos * (t_logprobs_pos - s_logprobs_pos)
        # kl_neg = P(0) * (log P(0) - log Q(0))
        kl_neg = t_probs_neg * (t_logprobs_neg - s_logprobs_neg)
        
        # Sum positive and negative parts to get binary KL divergence per class
        kl_div = kl_pos + kl_neg
        
        # Average the KL divergence across all classes and batch elements
        soft_loss = kl_div.mean()

        # Scale the soft loss by T^2
        # Why? Because scaling logits by 1/T scales the gradients of the soft loss by 1/T^2.
        # Multiplying by T^2 ensures the soft and hard losses contribute gradients of similar 
        # magnitudes, as established in Hinton et al. 2015.
        soft_loss = soft_loss * (self.temperature ** 2)

        # 3. Combine losses
        # Weighted combination of soft and hard objectives.
        loss = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        
        return loss
