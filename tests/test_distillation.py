import torch
import torch.nn.functional as F
import math
from distillation.loss import DistillationLoss
from distillation.student_model import StudentTransformer

def test_distillation_loss_identical_logits():
    """When student logits equal teacher logits, KL divergence should be zero."""
    student_logits = torch.randn(4, 28)
    teacher_logits = student_logits.clone()
    true_labels = torch.randint(0, 2, (4, 28)).float()
    
    # Use alpha=1.0 to isolate the soft-label loss (KL div)
    loss_fn = DistillationLoss(alpha=1.0, temperature=4.0)
    loss = loss_fn(student_logits, teacher_logits, true_labels)
    
    # It should be extremely close to zero
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_distillation_loss_alpha_zero():
    """When alpha=0, loss should equal exactly the hard-label BCE loss."""
    student_logits = torch.randn(4, 28)
    teacher_logits = torch.randn(4, 28)
    true_labels = torch.randint(0, 2, (4, 28)).float()
    
    loss_fn = DistillationLoss(alpha=0.0, temperature=4.0)
    loss = loss_fn(student_logits, teacher_logits, true_labels)
    
    # Calculate standard BCE manually
    bce = F.binary_cross_entropy_with_logits(student_logits, true_labels, reduction='mean')
    
    assert torch.isclose(loss, bce)

def test_distillation_loss_alpha_one():
    """When alpha=1, loss should equal exactly the scaled KL divergence."""
    student_logits = torch.randn(4, 28)
    teacher_logits = torch.randn(4, 28)
    true_labels = torch.randint(0, 2, (4, 28)).float()
    
    loss_fn_alpha1 = DistillationLoss(alpha=1.0, temperature=4.0)
    loss_1 = loss_fn_alpha1(student_logits, teacher_logits, true_labels)
    
    loss_fn_alpha05 = DistillationLoss(alpha=0.5, temperature=4.0)
    loss_05 = loss_fn_alpha05(student_logits, teacher_logits, true_labels)
    
    bce = F.binary_cross_entropy_with_logits(student_logits, true_labels, reduction='mean')
    
    # loss_05 should be 0.5 * loss_1 + 0.5 * bce
    expected_combined = 0.5 * loss_1 + 0.5 * bce
    assert torch.isclose(loss_05, expected_combined)

def test_distillation_loss_temperature_effect():
    """Test that higher temperature softens the distribution and reduces raw KL before scaling."""
    student_logits = torch.randn(4, 28)
    teacher_logits = torch.randn(4, 28)
    true_labels = torch.randint(0, 2, (4, 28)).float()
    
    loss_fn_t1 = DistillationLoss(alpha=1.0, temperature=1.0)
    loss_t1 = loss_fn_t1(student_logits, teacher_logits, true_labels)
    
    loss_fn_t4 = DistillationLoss(alpha=1.0, temperature=4.0)
    loss_t4 = loss_fn_t4(student_logits, teacher_logits, true_labels)
    
    # Since we multiply the final KL by T^2, the final scaled loss_t4 will actually be 
    # roughly equal or slightly larger/smaller depending on the logits, but the effect
    # of scaling by T^2 should be evident. 
    # Let's verify that T^2 scaling is correctly applied by extracting the raw mean KL.
    raw_kl_t1 = loss_t1 / (1.0 ** 2)
    raw_kl_t4 = loss_t4 / (4.0 ** 2)
    
    # Raw KL should be smaller with higher temperature because distributions become more uniform
    assert raw_kl_t4 < raw_kl_t1

def test_distillation_loss_gradients():
    """Test that loss backward works and populates gradients."""
    student_logits = torch.randn(4, 28, requires_grad=True)
    teacher_logits = torch.randn(4, 28)
    true_labels = torch.randint(0, 2, (4, 28)).float()
    
    loss_fn = DistillationLoss(alpha=0.5, temperature=4.0)
    loss = loss_fn(student_logits, teacher_logits, true_labels)
    
    loss.backward()
    
    assert student_logits.grad is not None
    assert torch.any(student_logits.grad != 0)

def test_distillation_loss_manual_kl():
    """Test the multi-label per-class KL against a hand-computed reference."""
    T = 4.0
    student_logits = torch.tensor([[1.0, -1.0]]) # 1 element, 2 classes
    teacher_logits = torch.tensor([[2.0, -2.0]])
    true_labels = torch.tensor([[1.0, 0.0]])
    
    loss_fn = DistillationLoss(alpha=1.0, temperature=T)
    loss = loss_fn(student_logits, teacher_logits, true_labels)
    
    # Hand compute for class 0
    t_logit_c0 = 2.0 / T
    s_logit_c0 = 1.0 / T
    p_t_c0 = 1.0 / (1.0 + math.exp(-t_logit_c0))
    p_s_c0 = 1.0 / (1.0 + math.exp(-s_logit_c0))
    
    kl_pos_c0 = p_t_c0 * (math.log(p_t_c0) - math.log(p_s_c0))
    kl_neg_c0 = (1 - p_t_c0) * (math.log(1 - p_t_c0) - math.log(1 - p_s_c0))
    kl_c0 = kl_pos_c0 + kl_neg_c0
    
    # Hand compute for class 1
    t_logit_c1 = -2.0 / T
    s_logit_c1 = -1.0 / T
    p_t_c1 = 1.0 / (1.0 + math.exp(-t_logit_c1))
    p_s_c1 = 1.0 / (1.0 + math.exp(-s_logit_c1))
    
    kl_pos_c1 = p_t_c1 * (math.log(p_t_c1) - math.log(p_s_c1))
    kl_neg_c1 = (1 - p_t_c1) * (math.log(1 - p_t_c1) - math.log(1 - p_s_c1))
    kl_c1 = kl_pos_c1 + kl_neg_c1
    
    avg_kl = (kl_c0 + kl_c1) / 2.0
    expected_loss = avg_kl * (T ** 2)
    
    assert math.isclose(loss.item(), expected_loss, rel_tol=1e-5)

def test_student_model_forward():
    """Verify the model instantiates and forward-passes on a dummy batch."""
    model = StudentTransformer(
        vocab_size=30522,
        max_position_embeddings=512,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_labels=28
    )
    
    # Dummy batch shape (2, 64)
    input_ids = torch.randint(0, 30522, (2, 64))
    attention_mask = torch.ones((2, 64), dtype=torch.long)
    attention_mask[1, 50:] = 0 # add some padding
    
    logits = model(input_ids, attention_mask=attention_mask)
    
    assert logits.shape == (2, 28)
    
def test_student_model_parameter_count():
    """Verify parameter count is in the expected range (roughly 11M–20M)."""
    model = StudentTransformer(
        vocab_size=30522,
        max_position_embeddings=512,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        num_labels=28
    )
    
    num_params = model.count_parameters()
    
    # 11M to 20M parameters
    assert 11_000_000 <= num_params <= 20_000_000, f"Param count {num_params} outside expected range"

def test_student_model_random_initialization():
    """Verify weights are randomly initialized (not zero, not identical across layers)."""
    model = StudentTransformer(
        num_hidden_layers=4,
        hidden_size=256
    )
    
    # Check that weights are not all zeros
    for name, param in model.named_parameters():
        if 'weight' in name and 'LayerNorm' not in name:
            assert torch.sum(torch.abs(param.data)) > 0.0, f"Parameter {name} appears to be all zeros."
            
    # Check that layer 0 and layer 1 have different weights (not identical)
    # The TransformerEncoder in PyTorch clones the provided layer, but since we called _init_weights 
    # manually using model.apply(), they should be independently initialized.
    # Actually wait, PyTorch's TransformerEncoder layer cloning uses copy.deepcopy, 
    # but since we initialized AFTER constructing the model, the weights should be independent.
    
    layer_0_weight = model.encoder.layers[0].linear1.weight.data
    layer_1_weight = model.encoder.layers[1].linear1.weight.data
    
    assert not torch.allclose(layer_0_weight, layer_1_weight), "Layers 0 and 1 have identical weights."
