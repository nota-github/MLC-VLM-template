"""
This file specifies how MLC's Phi parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""
import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .imp_model import Phi1Config, PhiConfig, ImpForCasualLM


def huggingface(model_config: PhiConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : PhiConfig
        The configuration of the Phi model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = ImpForCasualLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(  # pylint: disable=W0632:unbalanced-tuple-unpacking
        spec=model.get_default_spec()
    )
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

    _add("language_model.transformer.embd.weight", "transformer.embd.wte.weight")
    _add("multi_modal_projector.linear_1.bias", "transformer.mm_projector.0.bias") 
    _add("multi_modal_projector.linear_1.weight", "transformer.mm_projector.0.weight") 
    _add("multi_modal_projector.linear_2.bias", "transformer.mm_projector.2.bias") 
    _add("multi_modal_projector.linear_2.weight", "transformer.mm_projector.2.weight") 
    for mlc_name, _ in named_parameters.items():
        if mlc_name.split('.', 1)[0] == "language_model":
            if mlc_name not in mapping.param_map:
                hf_name = mlc_name.split('.', 1)[-1]
                _add(mlc_name, hf_name)
        elif mlc_name.split('.', 1)[0] == 'vision_tower':
            if mlc_name not in mapping.param_map:
                _add(mlc_name, "transformer.vision_tower."+mlc_name)
    return mapping


