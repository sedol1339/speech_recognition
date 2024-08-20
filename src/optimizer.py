import torch
from transformers.trainer_pt_utils import get_parameter_names


def group_param_names_for_weight_decay(module: torch.nn.Module, weight_decay: float):
    """
    Return dict which can be passed to optimizer constructor, so that weight decay
    will be applied to all weights except LayerNorm and bias.
    """
    wd_params = get_parameter_names(module, forbidden_layer_types=[torch.nn.LayerNorm])
    wd_params = [p for p in wd_params if 'bias' not in p]

    return [
        {
            'params': [p for name, p in module.named_parameters() if name in wd_params],
            'weight_decay': weight_decay,  # gets multiplied by lr
        },
        {
            'params': [
                p for name, p in module.named_parameters() if name not in wd_params
            ],
            'weight_decay': 0.0,
        },
    ]
