"""
This file specifies how MLC's Llava parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .phiva_model import PhivaConfig, PhivaForCasualLM
from .phiva_quantization import awq_quant


def huggingface(model_config: PhivaConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.
    Parameters
    ----------
    model_config : PhivaConfig
        The configuration of the Llava model.
    quantization : Quantization
        The quantization configuration.
    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = PhivaForCasualLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)  # type: ignore[misc]
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    def _add(mlc_name, hf_name):
        mapping.add_mapping(
            mlc_name,
            [hf_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    def _concat_add(mlc_name, hf_names):
        mapping.add_mapping(
            mlc_name,
            hf_names,
            functools.partial(
                lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    _add("language_model.lm_head.weight", "language_model.lm_head.weight")
    _add("language_model.transformer.norm.weight", "language_model.model.norm.weight")
    _add("language_model.transformer.embd.weight", "language_model.model.embed_tokens.weight")

    prefix = "language_model.transformer.h"
    hf_prefix = "language_model.model.layers"
    for i in range(model_config.text_config.num_hidden_layers):
        _add(f"{prefix}.{i}.ln.weight", f"{hf_prefix}.{i}.input_layernorm.weight")
        _add(f"{prefix}.{i}.mlp.down_proj.weight", f"{hf_prefix}.{i}.mlp.down_proj.weight")
        _add(f"{prefix}.{i}.mlp.gate_up_proj.weight", f"{hf_prefix}.{i}.mlp.gate_up_proj.weight")
        _add(
            f"{prefix}.{i}.post_attention_layernorm.weight",
            f"{hf_prefix}.{i}.post_attention_layernorm.weight",
        )
        _add(f"{prefix}.{i}.mixer.out_proj.weight", f"{hf_prefix}.{i}.self_attn.o_proj.weight")
        _add(f"{prefix}.{i}.mixer.qkv_proj.weight", f"{hf_prefix}.{i}.self_attn.qkv_proj.weight")

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
    return mapping


def awq(model_config: PhivaConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of AWQ parameters.
    Parameters
    ----------
    model_config : PhivaConfig
        The configuration of the Llava model.
    quantization : Quantization
        The quantization configuration.
    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to AWQ.
    """
    model, _ = awq_quant(model_config, quantization)
    _, _named_params = model.export_tvm(spec=model.get_default_spec())
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.text_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"language_model.model.layers.{i}.self_attn"
        for quantize_suffix in ["qweight", "qzeros", "scales"]:
            mlc_name = f"{attn}.qkv_proj.{quantize_suffix}"
            assert mlc_name in named_parameters
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{attn}.q_proj.{quantize_suffix}",
                    f"{attn}.k_proj.{quantize_suffix}",
                    f"{attn}.v_proj.{quantize_suffix}",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

        # Concat gate and up in MLP
        mlp = f"language_model.model.layers.{i}.mlp"
        for quantize_suffix in ["qweight", "qzeros", "scales"]:
            mlc_name = f"{mlp}.gate_up_proj.{quantize_suffix}"
            assert mlc_name in named_parameters
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{mlp}.gate_proj.{quantize_suffix}",
                    f"{mlp}.up_proj.{quantize_suffix}",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

        # inv_freq is not used in the model
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
            )
    return mapping