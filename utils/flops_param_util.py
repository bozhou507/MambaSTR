import inspect

import torch
import torch.nn as nn
from typing import Optional, Literal


from thop.profile import *
def _get_flops(
    model: nn.Module,
    inputs,
    custom_ops=None,
    verbose=True,
    report_missing=False,
):
    """modified from thop.profile.profile"""
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        # overwrite `verbose` option when enable report_missing
        verbose = True

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and report_missing:
                prRed(
                    "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params."
                    % m_type
                )

        if fn is not None:
            params = inspect.signature(fn).parameters
            handler_collection[m] = (
                m.register_forward_hook(fn, with_kwargs=('kwargs' in params)),
            )
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> int:
        total_ops = module.total_ops.item()
        for m in module.children():
            m_ops = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
        return total_ops

    total_ops = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler,) in handler_collection.items():
        op_handler.remove()
        m._buffers.pop("total_ops")

    return total_ops


def format_number(number: float, unit: Optional[Literal['K', 'M', 'G']]=None) -> float:
    if unit == 'K':
        number /= 1_000
    if unit == 'M':
        number /= 1_000_000
    elif unit == 'G':
        number /= 1_000_000_000
    return number


def get_flops(net, inputs, verbose=False, custom_ops=None, unit: Optional[Literal['K', 'M', 'G']]=None) -> float:
    """get FLOPs

    Returns:
        FLOPs in M or G
    """
    from .count_ops import count_mha
    default_custom_ops = {
        nn.MultiheadAttention: count_mha,
    }
    if custom_ops is not None:
        default_custom_ops.update(custom_ops)
    custom_ops = default_custom_ops
    
    def update_custom_ops(m: nn.Module):
        if hasattr(m, "count_extra_flops"):
            custom_ops.update({m.__class__: m.__class__.count_extra_flops})
    
    net.apply(update_custom_ops)
    from .copyutil import copy_internal_nodes
    inputs = copy_internal_nodes(inputs)
    flops = _get_flops(net, inputs=inputs, verbose=verbose, report_missing=verbose, custom_ops=custom_ops)
    return format_number(flops, unit)


def get_parameter_number(net, unit: Optional[Literal['K', 'M', 'G']]=None) -> float:
    """get Params

    Returns:
        Params in M or G
    """
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # return 'Trainable: {} M'.format(trainable_num/1000000)
    return format_number(trainable_num, unit)


def get_flops_params(net, inputs, verbose=False, custom_ops=None):
    """get FLOPs and Params
    
    Returns:
        FLOPs in G, Params in M
    """
    flops = get_flops(net, inputs=inputs, verbose=verbose, custom_ops=custom_ops)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # return 'Param: {} M, FLOPS: {} G'.format(params/1000000,flops/1000000000)
    return flops / 1_000_000_000, params / 1_000_000
