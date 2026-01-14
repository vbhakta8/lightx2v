#!/usr/bin/env python3
"""
Convert HuggingFace T5/UMT5 checkpoint to WAN/LightX2V format.

HuggingFace format:
  encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
  encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight
  encoder.block.{i}.layer.0.layer_norm.weight
  encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight  (gate)
  encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight  (fc1)
  encoder.block.{i}.layer.1.DenseReluDense.wo.weight    (fc2)
  encoder.block.{i}.layer.1.layer_norm.weight
  shared.weight
  encoder.final_layer_norm.weight

WAN/LightX2V format:
  blocks.{i}.attn.{q,k,v,o}.weight
  blocks.{i}.pos_embedding.embedding.weight
  blocks.{i}.norm1.weight
  blocks.{i}.ffn.gate.0.weight
  blocks.{i}.ffn.fc1.weight
  blocks.{i}.ffn.fc2.weight
  blocks.{i}.norm2.weight
  token_embedding.weight
  norm.weight

Usage:
  python hf_t5_to_wan.py input.safetensors output.safetensors
  python hf_t5_to_wan.py input.safetensors output.pth --output-format pth
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def build_key_mapping():
    """Build mapping from HuggingFace keys to WAN keys."""
    mapping = {}

    # Per-block mappings (24 blocks for UMT5-XXL)
    for i in range(24):
        hf_prefix = f"encoder.block.{i}"
        wan_prefix = f"blocks.{i}"

        # Attention
        mapping[f"{hf_prefix}.layer.0.SelfAttention.q.weight"] = f"{wan_prefix}.attn.q.weight"
        mapping[f"{hf_prefix}.layer.0.SelfAttention.k.weight"] = f"{wan_prefix}.attn.k.weight"
        mapping[f"{hf_prefix}.layer.0.SelfAttention.v.weight"] = f"{wan_prefix}.attn.v.weight"
        mapping[f"{hf_prefix}.layer.0.SelfAttention.o.weight"] = f"{wan_prefix}.attn.o.weight"

        # Relative position embedding
        mapping[f"{hf_prefix}.layer.0.SelfAttention.relative_attention_bias.weight"] = f"{wan_prefix}.pos_embedding.embedding.weight"

        # Layer norms
        mapping[f"{hf_prefix}.layer.0.layer_norm.weight"] = f"{wan_prefix}.norm1.weight"
        mapping[f"{hf_prefix}.layer.1.layer_norm.weight"] = f"{wan_prefix}.norm2.weight"

        # FFN (DenseReluDense / Gated-GELU)
        mapping[f"{hf_prefix}.layer.1.DenseReluDense.wi_0.weight"] = f"{wan_prefix}.ffn.gate.0.weight"
        mapping[f"{hf_prefix}.layer.1.DenseReluDense.wi_1.weight"] = f"{wan_prefix}.ffn.fc1.weight"
        mapping[f"{hf_prefix}.layer.1.DenseReluDense.wo.weight"] = f"{wan_prefix}.ffn.fc2.weight"

    # Global mappings
    mapping["shared.weight"] = "token_embedding.weight"
    mapping["encoder.embed_tokens.weight"] = "token_embedding.weight"  # Alternative key
    mapping["encoder.final_layer_norm.weight"] = "norm.weight"

    return mapping


def convert_hf_to_wan(input_path: str, output_path: str, output_format: str = "safetensors"):
    """Convert HuggingFace T5 checkpoint to WAN format."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading checkpoint: {input_path}")

    # Load input checkpoint
    if input_path.suffix == ".safetensors":
        weights = {}
        with safe_open(str(input_path), framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    elif input_path.suffix in [".pth", ".pt", ".bin"]:
        weights = torch.load(str(input_path), map_location="cpu")
        if "state_dict" in weights:
            weights = weights["state_dict"]
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    print(f"Loaded {len(weights)} tensors")

    # Build mapping
    key_mapping = build_key_mapping()

    # Convert keys
    converted = {}
    unmapped = []

    for hf_key, tensor in weights.items():
        if hf_key in key_mapping:
            wan_key = key_mapping[hf_key]
            converted[wan_key] = tensor
            print(f"  {hf_key} -> {wan_key}")
        else:
            unmapped.append(hf_key)

    if unmapped:
        print(f"\nWarning: {len(unmapped)} unmapped keys:")
        for key in unmapped[:10]:
            print(f"  - {key}")
        if len(unmapped) > 10:
            print(f"  ... and {len(unmapped) - 10} more")

    print(f"\nConverted {len(converted)} tensors")

    # Verify expected keys
    expected_keys = set(key_mapping.values())
    missing = expected_keys - set(converted.keys())
    if missing:
        print(f"\nWarning: {len(missing)} missing keys in output:")
        for key in sorted(missing)[:10]:
            print(f"  - {key}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "safetensors":
        save_file(converted, str(output_path))
    elif output_format == "pth":
        torch.save(converted, str(output_path))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"\nSaved to: {output_path}")

    # Print summary
    total_params = sum(t.numel() for t in converted.values())
    total_size = sum(t.numel() * t.element_size() for t in converted.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size / 1e9:.2f} GB")

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace T5 to WAN format")
    parser.add_argument("input", help="Input checkpoint path (safetensors or pth)")
    parser.add_argument("output", help="Output checkpoint path")
    parser.add_argument("--output-format", choices=["safetensors", "pth"],
                        default="safetensors", help="Output format (default: safetensors)")

    args = parser.parse_args()

    convert_hf_to_wan(args.input, args.output, args.output_format)


if __name__ == "__main__":
    main()
