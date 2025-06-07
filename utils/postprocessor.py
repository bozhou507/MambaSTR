import torch
import numpy as np
from typing import List
from .dictionary import Dictionary


def indices2word(indices, dictionary: Dictionary) -> List[str]:
    """Convert the output probabilities of a single image to index and
        score.

        Args:
            indices (torch.Tensor): Character logits with shape :math:`(B, T)`.
            dictionary (Dictionary): A Dictionary.

        Returns:
            np.array(List[str])
    """
    results = []
    for b in indices:
        result = dictionary.idx2str(list(b), ignore_indexes=dictionary.special_indexes)
        results.append(result)
    return np.array(results)


def attn_postprocessor(logits, dictionary: Dictionary) -> List[str]:
    """Convert the output probabilities of a single image to index and
        score.

        Args:
            logits (torch.Tensor): Character logits with shape :math:`(B, T, C)`.
            dictionary (Dictionary): A Dictionary.

        Returns:
            np.array(List[str])
    """
    assert len(logits.shape) == 3  # (B, T, C)
    indices = torch.argmax(logits, dim=-1)
    indices = indices.detach().cpu()
    return indices2word(indices, dictionary)


def ctc_postprocessor(logits, dictionary: Dictionary) -> List[str]:
    """Convert the output probabilities of a single image to index and
        score.

        Args:
            logits (torch.Tensor): Character logits with shape :math:`(B, T, C)`.
            dictionary (Dictionary): A Dictionary.

        Returns:
            np.array(List[str])
    """
    padding_idx = dictionary.padding_idx
    assert len(logits.shape) == 3  # (B, T, C)
    indices = torch.argmax(logits, dim=-1)
    indices = indices.detach().cpu()
    results = []
    for b in indices:
        index = []
        prev_idx = padding_idx
        for tmp_value in b:
            if tmp_value not in (prev_idx, *dictionary.special_indexes):
                index.append(tmp_value)
            prev_idx = tmp_value
        result = dictionary.idx2str(list(index))
        results.append(result)
    return np.array(results)
