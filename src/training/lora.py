import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA-enabled linear layer.

    This wraps an existing nn.Linear and adds a low-rank update term.
    Only the LoRA adapters are trainable when `freeze_base_weights=True`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: int = 32,
        dropout: float = 0.0,
        bias: bool = True,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.merge_weights = merge_weights

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = self.dropout(x) @ self.lora_A.t()
            lora_out = lora_out @ self.lora_B.t()
            return result + lora_out * self.scaling
        return result

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False


def _replace_linear_with_lora(module: nn.Module, r: int, alpha: int, dropout: float):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            lora_linear = LoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=r,
                alpha=alpha,
                dropout=dropout,
                bias=child.bias is not None,
            )
            lora_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_linear.bias.data.copy_(child.bias.data)
            lora_linear.freeze_base_weights()
            setattr(module, name, lora_linear)
        else:
            _replace_linear_with_lora(child, r, alpha, dropout)


def apply_lora(
    model: nn.Module,
    r: int = 4,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply LoRA to linear layers in the model.

    Args:
        model: PyTorch model to modify.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: dropout on the LoRA path.
        target_modules: list of module names to target.
            If None, all nn.Linear layers are replaced.
    """
    if target_modules is None:
        _replace_linear_with_lora(model, r, alpha, dropout)
        return model

    for name, module in model.named_children():
        if name in target_modules:
            _replace_linear_with_lora(module, r, alpha, dropout)
        else:
            apply_lora(module, r, alpha, dropout, target_modules)
    return model
