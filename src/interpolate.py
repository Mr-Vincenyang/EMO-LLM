"""LoRA weight interpolation — the core contribution.

Implements linear interpolation of LoRA adapter weights in parameter space,
enabling continuous control over emotional style expression.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel


def load_lora_weights(adapter_dir: str | Path) -> OrderedDict:
    """Load a LoRA adapter's state dict from a PEFT adapter directory."""
    adapter_dir = Path(adapter_dir)

    # Try safetensors first, then bin
    safetensor_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if safetensor_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensor_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or .bin found in {adapter_dir}")

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

    The core operation: given N trained LoRA adapters and N mixing
    weights, produce a single LoRA weight set as the weighted sum.

    ΔW_interp = Σ α_i · ΔW_i
    """
    styles = sorted(lora_dicts.keys())
    weight_list = [weights.get(s, 0.0) for s in styles]

    total = sum(weight_list)
    if total == 0:
        raise ValueError("Sum of weights must be > 0")
    weight_list = [w / total for w in weight_list]

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

    Weight-space interpolation: loads each style LoRA's parameters, blends them
    linearly, and patches the blended weights into a PEFT-wrapped model.
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
            print(f"  Loaded {style}: {len(self.lora_dicts[style])} LoRA params")

        # Load the first adapter as a PEFT model so we have LoRA layers to patch
        first_style = next(iter(self.lora_paths))
        first_path = str(self.lora_paths[first_style])
        self.peft_model = PeftModel.from_pretrained(base_model, first_path)
        # Merge into base to keep things clean, then re-wrap? No — we need the
        # lora layers present so we can patch their weights. Just keep as-is.

    def set_weights(self, weights: dict[str, float]) -> None:
        """Blend LoRA weights and patch into the active PEFT model."""
        interpolated = interpolate_lora_dicts(self.lora_dicts, weights)

        for name, param in self.peft_model.named_parameters():
            if name in interpolated:
                param.data.copy_(interpolated[name].to(param.device, param.dtype))

    def forward(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.peft_model.generate(*args, **kwargs)

    @property
    def device(self):
        return next(self.peft_model.parameters()).device
