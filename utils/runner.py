from torch.optim import Adam
from typing import List, Union
import os

import torch.nn as nn
from mmengine.runner import Runner
from mmengine.optim import AmpOptimWrapper
from .optim.schedulers import CosineAnnealingLinearWarmupRestarts

from .dataset import ConcatTrainDataset, ConcatValDataset, ConcatTestDataset
from .dataloader import get_train_dataloader, get_test_dataloader, get_val_dataloader
from .hooks import CheckpointHook
from .metrics import WordAccuracy
from .train_loop import TrainLoop
from .mmwrapper import convert_to_mmmodel, convert_lrscheduler_to_paramscheduler


def get_runner(
        model: nn.Module,
        train_dataset: ConcatTrainDataset=None,
        val_dataset: ConcatValDataset=None,
        test_dataset: ConcatTestDataset=None,
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
        save_failure_cases=False,
        train_sampler=None,
        cfg_dict: dict=dict()) -> Runner:

    word_size = os.environ.get("WORLD_SIZE")
    launcher=('pytorch' if word_size else 'none')
    model = convert_to_mmmodel(model)
    min_lr = min_lr_ratio * lr

    default_hooks = dict()

    if train_dataset is not None:
        assert train_batch_size > 0
        train_dataloader = get_train_dataloader(train_dataset, train_batch_size, num_train_workers, sampler=train_sampler)
        max_iters = max_epochs * len(train_dataloader)
        optimizer = Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-09)
        param_scheduler = CosineAnnealingLinearWarmupRestarts(optimizer, max_iters, max_lr=lr, min_lr=min_lr, warmup_steps=min(10000, max_iters // 10))
        cfg_dict.update(dict(optimizer=optimizer, scheduler=param_scheduler))
        param_scheduler = convert_lrscheduler_to_paramscheduler(param_scheduler)
        with_amp = False
        if with_amp:
            optim_wrapper = dict(type=AmpOptimWrapper, dtype='float16', optimizer=optimizer, clip_grad=dict(max_norm=5))
        else:
            optim_wrapper = dict(optimizer=optimizer, clip_grad=dict(max_norm=5))
        train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_by_epoch=False, val_interval=val_interval)
        default_hooks.update(dict(
            checkpoint=CheckpointHook(by_epoch=False, interval=len(train_dataloader), max_keep_ckpts=3, save_best=save_best, rule='greater', top_k=top_k)
        ))
    else:
        train_dataloader = None
        optimizer = None
        param_scheduler = None
        optim_wrapper = None
        train_cfg = None

    if val_dataset is not None:
        assert val_batch_size > 0
        val_dataloader = get_val_dataloader(val_dataset, val_batch_size, num_val_workers)
        val_cfg = dict()
        val_evaluator = dict(type=WordAccuracy, save_failure_cases=save_failure_cases)
    else:
        val_dataloader = None
        val_cfg = None
        val_evaluator = None

    if test_dataset is not None:
        assert test_batch_size > 0
        test_dataloader = get_test_dataloader(test_dataset, test_batch_size, num_test_workers)
        test_cfg = dict()
        test_evaluator = dict(type=WordAccuracy, save_failure_cases=save_failure_cases)
    else:
        test_dataloader = None
        test_cfg = None
        test_evaluator = None

    Runner.dump_config = lambda x: None
    runner = Runner(
        model=model,
        work_dir=work_dir + f"/{model.__class__.__name__}",
        default_hooks=default_hooks,
        # train cfg
        train_dataloader=train_dataloader,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        train_cfg=train_cfg,
        # val cfg
        val_dataloader=val_dataloader,
        val_cfg=val_cfg,
        val_evaluator=val_evaluator,
        # test cfg
        test_dataloader=test_dataloader,
        test_evaluator=test_evaluator,
        test_cfg=test_cfg,
        # others
        resume=resume,
        load_from=load_from,
        launcher=launcher
    )

    if resume and load_from is not None:
        runner.load_or_resume()
        if train_dataloader is not None:
            if runner.train_loop._epoch != runner.train_loop._iter // len(train_dataloader):
                runner.logger('Fixing current epoch')
                runner.train_loop._epoch = runner.train_loop._iter // len(train_dataloader)
            msg = f"fixed epoch: {runner.train_loop._epoch}, iter: {runner.train_loop._iter}, len(train_dataloader): {len(train_dataloader)}"
            runner.logger.info(msg)

    import mmengine.dist as dist
    if dist.is_main_process():
        import shutil
        from mmengine.config import Config
        mmcfg = Config(cfg_dict, format_python_code=False)
        cfg_save_path_pyext = runner.log_dir + '/' + runner.timestamp + '.py'
        cfg_save_path = runner.log_dir + '/' + runner.timestamp + '.cfg'
        mmcfg.dump(cfg_save_path_pyext)
        shutil.move(cfg_save_path_pyext, cfg_save_path)

    return runner
