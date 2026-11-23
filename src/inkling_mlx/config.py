"""Model configuration dataclasses for Inkling 2 Loop (3 passes, hybrid local/global attention)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional


def _filter_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


@dataclass(frozen=True)
class LoopConfig:
    core_start: int = 6
    core_end: int = 18
    num_passes: int = 3
    gate_rank: int = 64
    is_varloop: bool = False


@dataclass
class TextConfig:
    model_type: str = "inkling2_loop_text"
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    vocab_size: int = 201024
    unpadded_vocab_size: Optional[int] = 200058
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    model_max_length: int = 1048576
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 24
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    local_layer_ids: Optional[list[int]] = None
    layer_types: Optional[list[str]] = None
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: Optional[int] = 8192
    log_scaling_alpha: float = 0.1
    sconv_kernel_size: int = 4
    intermediate_size: int = 1152
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    n_shared_experts: int = 0
    route_scale: float = 8.0
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    attn_res_normalize_values: bool = True
    use_value_norm_no_gamma: bool = True
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float = 30.0
    loop_config: Optional[LoopConfig | dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextConfig":
        return cls(**_filter_fields(cls, data))

    def __post_init__(self) -> None:
        if isinstance(self.loop_config, dict):
            self.loop_config = LoopConfig(**_filter_fields(LoopConfig, self.loop_config))
        elif self.loop_config is None:
            self.loop_config = LoopConfig()
        loop = self.loop_config
        if loop.is_varloop:
            raise ValueError("Variable-depth Inkling 2 loops are not supported")
        block = self.n_layers_per_attn_res_block
        if (
            loop.num_passes < 1
            or not (0 < loop.core_start < loop.core_end <= self.num_hidden_layers)
            or loop.core_start % block != 0
            or loop.core_end % block != 0
        ):
            raise ValueError(f"Unsupported Inkling 2 loop_config: {loop}")

    @property
    def loop_layer_index_start(self) -> int:
        return self.loop_config.core_start

    @property
    def loop_layer_index_end(self) -> int:
        return self.loop_config.core_end

    @property
    def loop_num_passes(self) -> int:
        return self.loop_config.num_passes

    @property
    def loop_gate_rank(self) -> int:
        return self.loop_config.gate_rank

    @property
    def num_loop_layers(self) -> int:
        return self.loop_config.core_end - self.loop_config.core_start

    @property
    def total_num_hidden_layers(self) -> int:
        return self.num_hidden_layers + (self.loop_num_passes - 1) * self.num_loop_layers

    @property
    def layer_indices(self) -> tuple[int, ...]:
        indices = tuple(range(self.num_hidden_layers))
        loop = self.loop_config
        return (
            indices[: loop.core_start]
            + indices[loop.core_start : loop.core_end] * loop.num_passes
            + indices[loop.core_end :]
        )

    def layer_is_sliding(self, index: int) -> bool:
        if self.layer_types is not None:
            return self.layer_types[index] == "hybrid_sliding"
        if self.local_layer_ids is not None:
            return index in self.local_layer_ids
        return (index + 1) % 6 != 0


@dataclass
class VisionConfig:
    model_type: str = "inkling2_vision"
    native_patch_size: int = 20
    temporal_patch_size: int = 2
    adapter_merge_size: int = 2
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    hidden_size: int = 1024
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    num_channels: int = 3
    mlp_ratio: float = 4.0
    posemb_grid_size: int = 32
    layer_norm_eps: float = 1e-5
    text_hidden_size: int = 2304
    rope_theta: float = 10000.0
    out_hidden_size: int = 2304

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionConfig":
        return cls(**_filter_fields(cls, data))


@dataclass
class ModelConfig:
    model_type: str = "inkling2_loop"
    text_config: Optional[TextConfig | dict[str, Any]] = None
    vision_config: Optional[VisionConfig | dict[str, Any]] = None
    mtp_config: Optional[dict[str, Any]] = None
    image_token_id: int = 200054
    eos_token_id: int = 200006
    quantization: Optional[dict[str, Any]] = None
    quantization_config: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**_filter_fields(cls, data))

    def __post_init__(self) -> None:
        if not isinstance(self.text_config, TextConfig):
            self.text_config = TextConfig.from_dict(self.text_config or {})
        if not isinstance(self.vision_config, VisionConfig):
            self.vision_config = VisionConfig.from_dict(self.vision_config or {})
        self.vision_config.text_hidden_size = self.text_config.hidden_size
        self.vision_config.out_hidden_size = self.text_config.hidden_size
