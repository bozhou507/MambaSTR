import os
from torch.utils.data import DataLoader, DistributedSampler
from .dataset import BaseDataset


rank = os.environ.get("RANK")
word_size = os.environ.get("WORLD_SIZE")
is_distributed = (rank is not None and word_size is not None)
if is_distributed:
    rank = int(rank)
    word_size = int(word_size)


def get_train_dataloader(train_dataset: BaseDataset, train_batch_size: int, num_workers: int, sampler=None):
    if sampler is not None:
        train_dataloader_sample_params = dict(sampler=sampler)
    elif is_distributed:
        train_dataloader_sample_params = dict(sampler=DistributedSampler(dataset=train_dataset, rank=rank, num_replicas=word_size))
    else:
        train_dataloader_sample_params = dict(shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        persistent_workers=num_workers > 0,
        **train_dataloader_sample_params
    )
    return train_dataloader


def get_val_dataloader(val_dataset: BaseDataset, val_batch_size: int, num_workers: int):
    if is_distributed:
        val_dataloader_sample_params = dict(sampler=DistributedSampler(dataset=val_dataset, rank=rank, num_replicas=word_size, shuffle=False))
    else:
        val_dataloader_sample_params = dict(shuffle=False)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        persistent_workers=num_workers > 0,
        **val_dataloader_sample_params
    )
    return val_dataloader


def get_test_dataloader(test_dataset: BaseDataset, test_batch_size: int, num_workers: int):
    return get_val_dataloader(test_dataset, test_batch_size, num_workers)
