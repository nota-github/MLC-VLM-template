import dataclasses
import logging
from typing import Any, Dict, Optional, Tuple

import tvm
from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor, op
from tvm.relax.frontend.nn.modules import Conv2D
from tvm.relax.frontend.nn.op import (
    broadcast_to,
    concat,
    matmul,
    permute_dims,
    reshape,
    softmax,
    wrap_nested,
)
from tvm.relax.op import arange, strided_slice

from mlc_llm import op as op_ext
from mlc_llm.nn import FlashInferPagedKVCache, PagedKVCache, RopeMode, TIRPagedKVCache
from ...support.config import ConfigBase

@dataclasses.dataclass
class ImpVisionConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """
    Config for the vision encoder
    """

    hidden_size: int = 1152
    image_size: int = 196
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 26
    patch_size: int = 14
    dtype: str = "float16"
    num_channels: int = 3
    layer_norm_eps: float = 1e-06
    attention_dropout: float = 0.0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

class SigLipVisionEmbeddings(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: ImpVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
            dtype=config.dtype,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(
            num=self.num_positions, dim=self.embed_dim, dtype=config.dtype
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = reshape(patch_embeds, shape=(batch_size, self.embed_dim, -1))
        patch_embeds = permute_dims(
            patch_embeds, axes=(0, 2, 1)
        )  # shape = [batch,grid*grid,embed_dim]
        embeddings = patch_embeds
        posi_ids = reshape(
            wrap_nested(arange(0, self.num_positions, dtype="int32"), name="arange"), shape=(1, -1)
        )
        batch_position_embedding = broadcast_to(
            self.position_embedding(posi_ids),
            shape=(batch_size, self.num_positions, self.embed_dim),
        )
        embeddings = embeddings + batch_position_embedding
        return embeddings

class ImpQuickGELU(Module):
    def forward(self, input_tensor: Tensor) -> Tensor:
        return op.gelu(input_tensor, approximate="tanh")
    
class SigLipMLP(Module):
    def __init__(self, config: ImpVisionConfig):
        super().__init__()
        self.activation_fn = ImpQuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, dtype=config.dtype)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, dtype=config.dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SigLipAttention(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: ImpVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=config.dtype)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        reshape_tensor = reshape(tensor, shape=(bsz, seq_len, self.num_heads, self.head_dim))
        permute_tensor = permute_dims(reshape_tensor, axes=(0, 2, 1, 3))
        return permute_tensor

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        bsz, tgt_len, embed_dim = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states) * self.scale, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        proj_shape = (
            bsz * self.num_heads,
            -1,
            self.head_dim,
        )  # shape of (batch*num_heads, seq_len,head_dim)

        query_states = reshape(query_states, shape=proj_shape)
        key_states = reshape(key_states, shape=proj_shape)
        value_states = reshape(value_states, shape=proj_shape)

        trans_key_states = permute_dims(key_states, axes=(0, 2, 1))

        attn_weights = matmul(query_states, trans_key_states)
        attn_weights = softmax(attn_weights, axis=-1)
        attn_output = matmul(attn_weights, value_states)
        attn_output = reshape(attn_output, shape=(bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = permute_dims(attn_output, axes=(0, 2, 1, 3))
        attn_output = reshape(attn_output, shape=(bsz, tgt_len, embed_dim))
        attn_output = self.out_proj(attn_output)

        return attn_output
    

class SigLipEncoderLayer(Module):
    def __init__(self, config: ImpVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(
            normalized_shape=self.embed_dim, eps=config.layer_norm_eps, dtype=config.dtype
        )
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            normalized_shape=self.embed_dim, eps=config.layer_norm_eps, dtype=config.dtype
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs
    

class SigLipEncoder(Module):
    def __init__(self, config: ImpVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        hidden_states = inputs_embeds
        encoder_states: Tuple[Any, ...] = ()
        for _, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]
        encoder_states = encoder_states + (hidden_states,)
        return encoder_states


class SigLipVisionTransformer(Module):
    def __init__(self, config: ImpVisionConfig):
        super().__init__()
        # embed_dim = config.hidden_size
        self.embeddings = SigLipVisionEmbeddings(config)
        # self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, dtype=config.dtype)
        self.encoder = SigLipEncoder(config)
        # self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, dtype=config.dtype)

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.embeddings(pixel_values)
        print(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        return encoder_outputs
    
class SigLipVisionModel(Module):
    def __init__(self, config: ImpVisionConfig):
        super().__init__()
        self.vision_model = SigLipVisionTransformer(config)

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.vision_model(pixel_values)[-1]
    

