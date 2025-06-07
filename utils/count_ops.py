import torch.nn as nn


def count_mha_(m: nn.MultiheadAttention, query, key, value, output, **kwargs):
    if value is None:
        value = query
    D_q = m.kdim
    D_v = m.vdim
    L_q = query.size(-2)
    L_v = value.size(-2)
    m.total_ops += 2 * L_q * D_q ** 2 + L_v * D_v ** 2 + L_q * D_v ** 2 + L_q * L_v * (D_q + D_v)


def count_mha(m: nn.MultiheadAttention, args, kwargs, output):
    return count_mha_(m, *args, output=output, **kwargs)
