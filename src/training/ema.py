"""
Exponential Moving Average (EMA) for model weight stabilization.
Reduces weight fluctuations during training and improves generalization.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable


class EMAScheduler:
    """
    Exponential Moving Average weight update.
    
    Instead of using raw model weights, uses exponential moving average of weights
    computed during training. This reduces noise and improves generalization.
    
    Formula: ema_weights = decay * ema_weights + (1 - decay) * current_weights
    
    Typical decay: 0.999 (very smooth) to 0.99 (slightly less smooth)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cpu"):
        """
        Args:
            model: PyTorch model to apply EMA to
            decay: decay rate (higher = smoother, 0.999 is default for most papers)
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}  # Store EMA weights
        self.backup = {}  # Backup original weights
        
        # Initialize shadow (EMA) weights
        self._register_shadow_weights()
    
    def _register_shadow_weights(self):
        """Create shadow (EMA) copies of all model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    def update(self):
        """Update EMA weights after each training step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # EMA update: ema = decay * ema + (1 - decay) * current
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data +
                    (1 - self.decay) * param.data.to(self.device)
                )
    
    def apply_shadow(self):
        """Replace model weights with EMA weights."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].data.clone()
    
    def restore(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
    
    def state_dict(self) -> Dict:
        """Get EMA state for checkpointing."""
        return {
            "shadow": self.shadow,
            "decay": self.decay,
        }
    
    def load_state_dict(self, state: Dict):
        """Load EMA state from checkpoint."""
        self.shadow = state["shadow"]
        self.decay = state["decay"]
    
    def __enter__(self):
        """Context manager: apply shadow weights."""
        self.apply_shadow()
        return self
    
    def __exit__(self, *args):
        """Context manager: restore original weights."""
        self.restore()


# ──────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE FOR TRAINING
# ──────────────────────────────────────────────────────────────────────────────

"""
Example integration with training loop:

from src.training.ema import EMAScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YourModel().to(device)
optimizer = torch.optim.Adam(model.parameters())
ema = EMAScheduler(model, decay=0.999, device=device)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Training step
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        # Update EMA weights
        ema.update()
    
    # Validation with EMA weights
    with ema:  # Temporarily use EMA weights
        val_loss = evaluate(model, val_loader)
    
    # Save checkpoint with both model weights and EMA state
    checkpoint = {
        'model_state': model.state_dict(),
        'ema_state': ema.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

# At inference, use EMA weights for best performance
ema.apply_shadow()
predictions = model(test_image)
ema.restore()
"""


if __name__ == "__main__":
    # Simple test
    print("Testing EMA implementation...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    ema = EMAScheduler(model, decay=0.999)
    
    # Simulate a training step
    for _ in range(5):
        model[0].weight.data += torch.randn_like(model[0].weight) * 0.01
        ema.update()
    
    print(f"✅ EMA initialized with decay=0.999")
    print(f"✅ Shadow parameters registered: {len(ema.shadow)}")
    print(f"✅ EMA can be used with context manager: 'with ema:'")
