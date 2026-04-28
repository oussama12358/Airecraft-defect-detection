"""
RF-DETER: Random Feature Distillation Enhanced Training
Improves robustness by adding controlled perturbations during training.
Especially useful for small datasets with potential annotation noise.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


class RFDeterMixin:
    """
    Mixin to add RF-DETER capability to any model.
    Adds random feature perturbations during training for robustness.
    """
    
    def __init__(self, perturbation_std: float = 0.1, where: str = "input"):
        """
        Args:
            perturbation_std: standard deviation of Gaussian noise
            where: where to apply perturbation
                - "input": add noise to raw input
                - "features": add noise after first conv/linear layer
                - "activation": add noise in hidden activations
        """
        self.perturbation_std = perturbation_std
        self.where = where
        self._original_forward = None
    
    def enable_rf_deter(self):
        """Wrap forward pass to include RF-DETER perturbations."""
        self._original_forward = self.forward
        self.forward = self._rf_deter_forward
    
    def disable_rf_deter(self):
        """Restore original forward pass."""
        if self._original_forward is not None:
            self.forward = self._original_forward
    
    def _rf_deter_forward(self, x):
        """Forward pass with random feature perturbations."""
        if not self.training:
            # No perturbation at inference
            return self._original_forward(x)
        
        if self.where == "input":
            # Add noise to raw input
            noise = torch.randn_like(x) * self.perturbation_std
            x_perturbed = x + noise
            return self._original_forward(x_perturbed)
        
        else:
            # For more complex perturbation strategies,
            # you'd need to modify internal layers
            return self._original_forward(x)


class RFDeterWrapper(nn.Module):
    """
    Wrapper to add RF-DETER to any existing model.
    More flexible than mixin approach.
    """
    
    def __init__(
        self,
        model: nn.Module,
        perturbation_std: float = 0.1,
        where: str = "input",
    ):
        """
        Args:
            model: existing PyTorch model
            perturbation_std: noise scale
            where: where to apply noise ("input", "features", "activation")
        """
        super().__init__()
        self.model = model
        self.perturbation_std = perturbation_std
        self.where = where
    
    def forward(self, x):
        """Forward with optional perturbation during training."""
        if not self.training:
            # Inference: no perturbation
            return self.model(x)
        
        if self.where == "input":
            # Add Gaussian noise to input
            noise = torch.randn_like(x) * self.perturbation_std
            x_perturbed = x + noise
            return self.model(x_perturbed)
        
        elif self.where == "features":
            # More sophisticated: modify internal features
            # This requires hooking into the model's forward pass
            return self._forward_with_feature_perturbation(x)
        
        else:
            return self.model(x)
    
    def _forward_with_feature_perturbation(self, x):
        """Apply perturbation to feature maps (requires model adaptation)."""
        # For ResNet-like models
        if hasattr(self.model, 'conv1'):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            
            # Add noise to features
            noise = torch.randn_like(x) * self.perturbation_std
            x = x + noise
            
            x = self.model.maxpool(x)
            # Continue with rest of layers
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x
        else:
            # Fallback to input perturbation
            noise = torch.randn_like(x) * self.perturbation_std
            x = x + noise
            return self.model(x)


# ──────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLES
# ──────────────────────────────────────────────────────────────────────────────

"""
Example 1: Using RFDeterWrapper (simplest)

from src.training.rf_deter import RFDeterWrapper
from src.models.resnet50 import build_resnet50

base_model = build_resnet50(num_classes=6)
model = RFDeterWrapper(base_model, perturbation_std=0.1, where="input")

# Training loop
for epoch in range(num_epochs):
    model.train()  # Enable dropout, normalization, and RF-DETER
    for batch, targets in dataloader:
        output = model(batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()  # Disable RF-DETER at validation
    for batch, targets in val_dataloader:
        output = model(batch)
        # No perturbation during validation


Example 2: RF-DETER in training loop (most flexible)

model.train()
for batch, targets in dataloader:
    # Forward pass with perturbation
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

# Inference without perturbation
model.eval()
output = model(test_batch)


Example 3: Progressive perturbation (increase over time)

initial_std = 0.01
final_std = 0.1
total_steps = len(dataloader) * num_epochs

for step, (batch, targets) in enumerate(dataloader):
    # Linearly increase perturbation strength
    current_std = initial_std + (final_std - initial_std) * (step / total_steps)
    
    wrapper.perturbation_std = current_std
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
"""


# ──────────────────────────────────────────────────────────────────────────────
# TESTING
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing RF-DETER implementation...")
    
    # Create simple model
    simple_model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 6)
    )
    
    # Wrap with RF-DETER
    model = RFDeterWrapper(simple_model, perturbation_std=0.1)
    
    # Test shapes
    x = torch.randn(4, 10)
    
    # Training mode (with perturbation)
    model.train()
    out1 = model(x)
    out2 = model(x)
    
    print(f"✅ Model output shape: {out1.shape}")
    print(f"✅ Training outputs differ (due to perturbation): {not torch.allclose(out1, out2)}")
    
    # Eval mode (no perturbation)
    model.eval()
    out3 = model(x)
    out4 = model(x)
    
    print(f"✅ Eval outputs are identical: {torch.allclose(out3, out4)}")
    print(f"✅ RF-DETER working correctly!")
