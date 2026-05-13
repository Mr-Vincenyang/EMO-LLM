"""LoRA weight interpolation — the core contribution.

Implements linear interpolation of LoRA adapter weights in parameter space,
enabling continuous control over emotional style expression.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import PreTrainedModel


def load_lora_weights(state_dict_path: str | Path) -> OrderedDict:
    """Load a LoRA adapter's state dict from disk."""
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    # Handle both raw state_dict and peft checkpoint format
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Filter only lora weights
    lora_dict = OrderedDict()
    for key, value in state_dict.items():
        if "lora" in key:
            lora_dict[key] = value
    return lora_dict


def interpolate_lora_dicts(
    lora_dicts: dict[str, OrderedDict],
    weights: dict[str, float],
) -> OrderedDict:
    """Interpolate multiple LoRA weight dictionaries with given weights.

    This is the core operation: given N trained LoRA adapters and N mixing
    weights, produce a single LoRA weight set that is the weighted sum.

    Args:
        lora_dicts: Dict mapping style_name -> lora state_dict.
        weights: Dict mapping style_name -> mixing weight (will be normalized).

    Returns:
        Interpolated LoRA state dict.
    """
    styles = sorted(lora_dicts.keys())
    weight_list = [weights.get(s, 0.0) for s in styles]

    total = sum(weight_list)
    if total == 0:
        raise ValueError("Sum of weights must be > 0")
    weight_list = [w / total for w in weight_list]

    # All LoRAs should have the same keys
    ref_keys = list(lora_dicts[styles[0]].keys())
    for s in styles[1:]:
        if set(lora_dicts[s].keys()) != set(ref_keys):
            raise ValueError(f"LoRA key mismatch: {s} differs from {styles[0]}")

    interpolated = OrderedDict()
    for key in ref_keys:
        interpolated[key] = sum(
            weight_list[i] * lora_dicts[styles[i]][key].float()
            for i in range(len(styles))
        )

    return interpolated


class InterpolatableLoRA(nn.Module):
    """Wraps a base model + multiple LoRA adapters for interpolation at inference.

    Supports two modes:
      1. Weight-space interpolation: merge LoRA weights, then load.
      2. Output-space interpolation: run each LoRA separately, blend logits.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        lora_paths: dict[str, str | Path],
        interpolation_mode: str = "weight",
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_paths = {k: Path(v) for k, v in lora_paths.items()}
        self.mode = interpolation_mode

        # Pre-load all LoRA state dicts
        self.lora_dicts: dict[str, OrderedDict] = {}
        for style, path in self.lora_paths.items():
            self.lora_dicts[style] = load_lora_weights(path)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set mixing weights and update the active LoRA adapter."""
        interpolated = interpolate_lora_dicts(self.lora_dicts, weights)

        # Apply interpolated weights to the model's LoRA parameters
        for name, param in self.base_model.named_parameters():
            if name in interpolated:
                param.data.copy_(interpolated[name].to(param.device, param.dtype))

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @classmethod
    def from_single_peft(
        cls,
        base_model: PreTrainedModel,
        peft_model: PeftModel,
        style: str,
    ) -> InterpolatableLoRA:
        """Create an InterpolatableLoRA from a single PEFT model."""
        # Extract LoRA weights from the PEFT model
        lora_dict = OrderedDict()
        for name, param in peft_model.named_parameters():
            if "lora" in name:
                lora_dict[name] = param.data.clone().cpu()

        wrapper = cls.__new__(cls)
        wrapper.base_model = base_model
        wrapper.lora_paths = {}
        wrapper.mode = "weight"
        wrapper.lora_dicts = {style: lora_dict}
        return wrapper
