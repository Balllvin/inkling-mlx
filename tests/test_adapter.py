"""Tests for PEFT LoRA adapter loading and merging into Inkling 2 Mini MLX weights."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from inkling_mlx.model import apply_peft_lora_adapter


class PeftAdapterTests(unittest.TestCase):
    def _write_adapter(
        self,
        root: Path,
        tensors: dict[str, mx.array],
        *,
        r: int = 2,
        alpha: float = 4.0,
        subfolder: bool = False,
    ) -> Path:
        target = (root / "adapter") if subfolder else root
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": alpha}))
        mx.save_safetensors(str(target / "adapter_model.safetensors"), tensors)
        return root

    def test_merges_attention_and_message_gate_and_skips_vision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = {
                "model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 6), dtype=mx.bfloat16),
                "model.layers.0.self_attn.r_proj.weight": mx.zeros((4, 6), dtype=mx.bfloat16),
                "model.layers.0.self_attn.gate_proj.weight": mx.zeros((4, 6), dtype=mx.bfloat16),
                "model.layers.0.self_attn.sink_proj.weight": mx.zeros((2, 6), dtype=mx.bfloat16),
                "model.layers.10.k_message_gate.forget_gate_proj.weight": mx.ones((6, 2), dtype=mx.float32),
            }
            a_q = mx.ones((2, 6), dtype=mx.bfloat16)
            b_q = mx.full((4, 2), 0.5, dtype=mx.bfloat16)
            a_fg = mx.full((2, 2), 0.25, dtype=mx.bfloat16)
            b_fg = mx.ones((6, 2), dtype=mx.bfloat16)
            lora = {
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.linear.lora_A.weight": a_q,
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.linear.lora_B.weight": b_q,
                "base_model.model.model.language_model.layers.0.self_attn.relative_position_bias.linear.lora_A.weight": a_q,
                "base_model.model.model.language_model.layers.0.self_attn.relative_position_bias.linear.lora_B.weight": b_q,
                "base_model.model.model.language_model.layers.0.self_attn.o_proj.gate_proj.lora_A.weight": a_q,
                "base_model.model.model.language_model.layers.0.self_attn.o_proj.gate_proj.lora_B.weight": b_q,
                "base_model.model.model.language_model.layers.0.self_attn.sink_proj.lora_A.weight": a_q,
                "base_model.model.model.language_model.layers.0.self_attn.sink_proj.lora_B.weight": mx.ones((2, 2), dtype=mx.bfloat16),
                "base_model.model.model.language_model.layers.10.k_message_gate.forget_gate_proj.lora_A.weight": a_fg,
                "base_model.model.model.language_model.layers.10.k_message_gate.forget_gate_proj.lora_B.weight": b_fg,
                "base_model.model.model.vision_tower.layers.0.mlp.fc1.lora_A.weight": mx.ones((2, 8), dtype=mx.bfloat16),
                "base_model.model.model.vision_tower.layers.0.mlp.fc1.lora_B.weight": mx.ones((8, 2), dtype=mx.bfloat16),
            }
            self._write_adapter(root, lora, r=2, alpha=4.0, subfolder=True)
            stats = apply_peft_lora_adapter(clean, root)
            self.assertEqual(stats["merged_modules"], 5)
            self.assertEqual(stats["skipped_vision_tensors"], 2)
            # (B @ A) * (4/2) = 2 * (0.5 * 1.0 * 2) = 2.0
            self.assertTrue(mx.allclose(clean["model.layers.0.self_attn.q_proj.weight"].astype(mx.float32), mx.full((4, 6), 2.0)).item())
            self.assertEqual(clean["model.layers.10.k_message_gate.forget_gate_proj.weight"].dtype, mx.float32)
            # 1.0 + 2 * (1.0 * 0.25 * 2) = 2.0
            self.assertTrue(mx.allclose(clean["model.layers.10.k_message_gate.forget_gate_proj.weight"], mx.full((6, 2), 2.0)).item())

    def test_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = {"model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 6), dtype=mx.bfloat16)}
            lora = {
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.linear.lora_A.weight": mx.zeros((2, 5), dtype=mx.bfloat16),
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.linear.lora_B.weight": mx.zeros((4, 2), dtype=mx.bfloat16),
            }
            self._write_adapter(root, lora)
            with self.assertRaises(ValueError):
                apply_peft_lora_adapter(clean, root)


if __name__ == "__main__":
    unittest.main()
