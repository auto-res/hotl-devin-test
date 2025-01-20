import pytest
import torch
import torch.nn as nn
from src.loranorm import LoRANorm


@pytest.fixture
def sample_inputs():
    """Fixture providing sample input tensors."""
    return {
        'small': torch.randn(2, 10),
        'medium': torch.randn(8, 20),
        'batch3d': torch.randn(4, 3, 16)
    }


@pytest.fixture
def loranorm_configs():
    """Fixture providing different LoRANorm configurations."""
    return [
        {'fan_in': 10, 'fan_out': 5, 'rank': 4},
        {'fan_in': 20, 'fan_out': 10, 'rank': 8, 'lora_dropout_p': 0.1},
        {'fan_in': 16, 'fan_out': 8, 'rank': 2, 'lora_alpha': 2.0}
    ]


def test_loranorm_shapes(sample_inputs, loranorm_configs):
    """Test output shapes for different input sizes and configurations."""
    # Test first config with all input sizes
    config = loranorm_configs[0]
    loranorm = LoRANorm(fan_in=10, fan_out=5)
    
    x = sample_inputs['small']
    output = loranorm(x)
    assert output.shape == (2, 5), "Wrong output shape for small input"
    
    # Test different configurations
    for config in loranorm_configs:
        loranorm = LoRANorm(**config)
        x = torch.randn(4, config['fan_in'])
        output = loranorm(x)
        assert output.shape == (4, config['fan_out']), \
            f"Wrong output shape for config {config}"


def test_weight_normalization():
    """Test that weight normalization is working correctly."""
    loranorm = LoRANorm(fan_in=10, fan_out=5, rank=4)
    
    # Check that normalized weights have unit norm
    w_norm = loranorm.compute_weight()
    norms = torch.norm(w_norm, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)


def test_enable_disable_lora():
    """Test enabling and disabling LoRA functionality."""
    loranorm = LoRANorm(fan_in=10, fan_out=5)
    x = torch.randn(2, 10)
    
    # Get output with LoRA enabled
    output_enabled = loranorm(x)
    
    # Disable LoRA and check output
    loranorm.disable_lora()
    output_disabled = loranorm(x)
    
    # Outputs should be different when LoRA is enabled vs disabled
    assert not torch.allclose(output_enabled, output_disabled, rtol=1e-4, atol=1e-4), \
        "LoRA disable had no effect"
    
    # Re-enable and check if we get similar output to original
    loranorm.enable_lora()
    output_reenabled = loranorm(x)
    torch.testing.assert_close(output_enabled, output_reenabled, rtol=1e-4, atol=1e-4)


def test_dropout():
    """Test that dropout is applied correctly."""
    torch.manual_seed(42)  # for reproducibility
    loranorm = LoRANorm(fan_in=10, fan_out=5, lora_dropout_p=0.5)
    x = torch.randn(2, 10)
    
    # Run multiple forward passes
    outputs = [loranorm(x) for _ in range(10)]
    
    # Outputs should be different due to dropout
    for i in range(len(outputs)-1):
        assert not torch.allclose(outputs[i], outputs[i+1], rtol=1e-4, atol=1e-4), \
            "Dropout seems to have no effect"


def test_layer_conversion():
    """Test conversion from different layer types."""
    # Test linear layer conversion
    linear = nn.Linear(20, 10)
    loranorm = LoRANorm.from_linear(linear, rank=4)
    assert loranorm.lora_A.shape == (4, 20)
    assert loranorm.lora_B.shape == (10, 4)
    
    # Test conv2d layer conversion
    conv = nn.Conv2d(3, 6, kernel_size=3)
    loranorm = LoRANorm.from_conv2d(conv, rank=4)
    fan_out, fan_in = conv.weight.view(conv.weight.shape[0], -1).shape
    assert loranorm.lora_A.shape == (4, fan_in)
    assert loranorm.lora_B.shape == (fan_out, 4)
    
    # Test embedding layer conversion
    emb = nn.Embedding(100, 32)
    loranorm = LoRANorm.from_embedding(emb, rank=4)
    assert loranorm.lora_A.shape == (32, 4)  # Transposed due to fan_in_fan_out=True
    assert loranorm.lora_B.shape == (4, 100)


def test_initialization():
    """Test that parameters are initialized correctly."""
    loranorm = LoRANorm(fan_in=10, fan_out=5, rank=4)
    
    # Check that lora_A and lora_B are initialized
    assert not torch.allclose(loranorm.lora_A, torch.zeros_like(loranorm.lora_A))
    assert not torch.allclose(loranorm.lora_B, torch.zeros_like(loranorm.lora_B))
    
    # Check scaling factor
    assert loranorm.scaling == loranorm.lora_alpha / loranorm.rank


def test_forward_deterministic():
    """Test that forward pass is deterministic when dropout is disabled."""
    torch.manual_seed(42)
    loranorm = LoRANorm(fan_in=10, fan_out=5, lora_dropout_p=0.0)
    x = torch.randn(2, 10)
    
    # Multiple forward passes should give identical results
    output1 = loranorm(x)
    output2 = loranorm(x)
    torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-4)
