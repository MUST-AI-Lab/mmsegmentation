from typing import Dict
import torch
from numbers import Number
from collections import Sequence


def dict_fuse(obj_list, reference_obj):
    if isinstance(reference_obj, torch.Tensor):
        return torch.stack(obj_list)
    return obj_list


def dict_select(dict1: Dict[str, list], key: str, value: str):
    flag = [v == value for v in dict1[key]]
    return {
        k: dict_fuse([vv for vv, ff in zip(v, flag) if ff], v) for k, v in dict1.items()
    }


def dict_split(dict1, key):
    group_names = list(set(dict1[key]))
    dict_groups = {k: dict_select(dict1, key, k) for k in group_names}

    return dict_groups


def weighted_loss(loss, weight=1.0):
    if isinstance(weight, Number) and isinstance(loss, torch.Tensor):
        loss = loss*weight
    else:
        raise NotImplementedError()
    return loss

"""
def sequence_mul(obj, multiplier):
    if isinstance(obj, Sequence):
        return [o * multiplier for o in obj]
    else:
        return obj * multiplier 
"""

def patch_config_semi(cfg):
    # wrap for semi
    if cfg.get("semi_wrapper", None) is not None:
        cfg.semi_wrapper.model = cfg.model
        cfg.model = cfg.semi_wrapper
        cfg.pop("semi_wrapper")
        #This will disable the optimizer hook, since we use customized update logic.
        cfg.optimizer_config = None
    return cfg
