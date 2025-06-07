import torch
import warnings
from typing import Iterable, Callable


def moore_vote(candidates: Iterable, key: Callable=None):
    """Boyer-Moore Majority Vote Algorithm

    Find the candidate who has more than half of the votes.
    
    Args:
        candidates (Iterable): the candidates
        key (Callable): count votes according to key(candidate)
    """
    if key is None:
        key = lambda x: x
    
    candidate_keys = list(map(key, candidates))

    majority_pos = None
    count = 0

    for i in range(len(candidates)):
        if count == 0:
            majority_pos = i
            count = 1
        elif candidate_keys[i] == candidate_keys[majority_pos]:
            count += 1
        else:
            count -= 1
    
    count = 0
    for candidate_key in candidate_keys:
        if candidate_key == candidate_keys[majority_pos]:
            count += 1
    if count * 2 <= len(candidates):
        warnings.warn('No candidate has more than half of the votes, random candidate will be returned')

    return candidates[majority_pos]


def stack_tensor_list(item_list):
    try:
        stacked_items = torch.stack(item_list)
    except RuntimeError:
        # The runtime error is most likely caused by the inconsistency of tensor.shape in item_list
        # Relevant information is recorded here for debugging and to avoid interruption of training as much as possible
        message = 'RuntimeError:\n'
        majorty_item = moore_vote(item_list, key=lambda x: x.shape)
        respect_shape = majorty_item.shape
        message += f'respect_shape = {respect_shape}\n'
        new_item_list = []
        for item in item_list:
            if item.shape == respect_shape:
                new_item_list.append(item)
            else:
                message += f'found shape of {item.shape}' 
                new_item_list.append(majorty_item)
        warnings.warn(message)
        with open('debug.log', 'a') as f:
            message = __file__ + '\n' + message
            f.write(message)
        stacked_items = torch.stack(new_item_list)
    return stacked_items


def default_collate_fn(batch):
    """Orgnize a list of items

    if batch is a list of tuples, perform:
        [(a1, b1, ...), (a2, b2, ...), ..., (an, bn, ...)] => [(a1, a2, ..., an), (b1, b2, ..., bn), ...]
    
    Args:
        batch: a list of items, item can be tuple.
    """
    # assert isinstance(batch, list) and len(batch) > 0
    if isinstance(batch[0], tuple):
        item_lists = zip(*batch)
        collated_data = []
        for item_list in item_lists:
            collated_data.append(default_collate_fn(item_list))
        return collated_data
    else:
        if isinstance(batch[0], torch.Tensor):
            stacked_items = stack_tensor_list(batch)
            return stacked_items
        else:
            return batch
