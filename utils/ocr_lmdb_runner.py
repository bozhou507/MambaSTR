import torch.nn as nn
from mmengine.runner import Runner
from typing import Union, List
import os

from .runner import get_runner as _get_runner
from .multimodel_runner import get_multimodel_runer
from .dataset import ConcatTrainDataset, ConcatValDataset, ConcatTestDataset


def get_ocr_lmdb_runner(
        model: Union[nn.Module, List[nn.Module]],
        train_lmdb_paths: List[str]=None,
        val_lmdb_paths: List[str]=None,
        test_lmdb_paths: List[str]=None,
        train_batch_size: int=0,
        val_batch_size: int=0,
        test_batch_size: int=0,
        num_train_workers: int=0,
        num_val_workers: int=0,
        num_test_workers: int=0,
        max_epochs: int=20,
        resume: bool=False,
        load_from: str=None,
        lr: float=1e-3,
        min_lr_ratio: int=0.5,  # min_lr = min_lr_ratio * lr
        work_dir: str='./work_dir',
        val_interval: int=1000,
        save_best: Union[str, List[str]]='avg/word_acc',
        top_k: Union[int, List[int]]=3,
        save_failure_cases=False) -> Runner:

    if isinstance(model, list) and len(model) == 1:
        model = model[0]

    if isinstance(model, list):
        models = model
        first_model = model[0]
        cfg_dicts = [dict(
            train_lmdb_paths=train_lmdb_paths,
            val_lmdb_paths=val_lmdb_paths,
            test_lmdb_paths=test_lmdb_paths,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_train_workers = num_train_workers,
            num_val_workers = num_val_workers,
            num_test_workers = num_test_workers,
            max_epochs=max_epochs,
            resume=resume,
            load_from=load_from,
            lr=lr,
            min_lr_ratio=min_lr_ratio,
            work_dir=work_dir,
            save_failure_cases=save_failure_cases,
            model=model) for model in models]
    else:
        first_model = model
        models = None
        cfg_dict = dict(
            train_lmdb_paths=train_lmdb_paths,
            val_lmdb_paths=val_lmdb_paths,
            test_lmdb_paths=test_lmdb_paths,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_train_workers = num_train_workers,
            num_val_workers = num_val_workers,
            num_test_workers = num_test_workers,
            max_epochs=max_epochs,
            resume=resume,
            load_from=load_from,
            lr=lr,
            min_lr_ratio=min_lr_ratio,
            work_dir=work_dir,
            save_failure_cases=save_failure_cases,
            model=model)

    if train_lmdb_paths is not None:
        assert train_batch_size > 0
        _datasets = first_model.get_ocr_lmdb_datasets(train_lmdb_paths, training=True)
        train_dataset = ConcatTrainDataset(datasets=_datasets)
        train_sampler = None
    else:
        train_dataset = None
        train_sampler = None

    if val_lmdb_paths is not None:
        assert val_batch_size > 0
        val_dataset_names = [lmdb_path.rsplit('/', 1)[-1] for lmdb_path in val_lmdb_paths]
        save_best = ['wavg/word_acc', 'avg/word_acc']  # + [f"{name}/word_acc" for name in val_dataset_names]
        val_dataset = ConcatValDataset(
            datasets=first_model.get_ocr_lmdb_datasets(val_lmdb_paths, training=False),
            dataset_names=val_dataset_names)
    else:
        val_dataset = None

    if test_lmdb_paths is not None:
        assert test_batch_size > 0
        test_dataset_names = [lmdb_path.rsplit('/', 1)[-1] for lmdb_path in test_lmdb_paths]
        test_dataset = ConcatTestDataset(
            datasets=first_model.get_ocr_lmdb_datasets(test_lmdb_paths, training=False),
            dataset_names=test_dataset_names)
    else:
        test_dataset = None

    assert models is None
    if models is None:
        return _get_runner(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_train_workers = num_train_workers,
            num_val_workers = num_val_workers,
            num_test_workers = num_test_workers,
            max_epochs=max_epochs,
            resume=resume,
            load_from=load_from,
            lr=lr,
            min_lr_ratio=min_lr_ratio,
            work_dir=work_dir,
            val_interval=val_interval,
            save_best=save_best,
            top_k=top_k,
            save_failure_cases=save_failure_cases,
            train_sampler=train_sampler,
            cfg_dict=cfg_dict)
    else:
        return get_multimodel_runer(
            models=models,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_train_workers = num_train_workers,
            num_val_workers = num_val_workers,
            num_test_workers = num_test_workers,
            max_epochs=max_epochs,
            resume=resume,
            load_from=load_from,
            lr=lr,
            min_lr_ratio=min_lr_ratio,
            work_dir=work_dir,
            save_failure_cases=save_failure_cases,
            train_sampler=train_sampler,
            cfg_dicts=cfg_dicts)
