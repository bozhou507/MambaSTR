import time
import copy
import shutil
import torch.nn as nn
from typing import Union, List
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.model import is_model_wrapper
from mmengine.optim import AmpOptimWrapper
from .train_loop import MultiRunnerTrainLoop

class MultimodelRunner:
    """All models share same config"""
    def __init__(
        self,
        models: Union[nn.Module, List[nn.Module]],
        work_dir: str,
        cfgs: Union[dict, List[dict]]=None,
        optim_wrappers=None,
        param_schedulers=None,
        **kwargs):
        if isinstance(models, nn.Module):
            models = [models]
        if isinstance(cfgs, dict):
            cfgs = [copy.deepcopy(cfgs) for _ in models]
        assert len(models) == len(cfgs)
        self.runners = []
        last_time = None
        train_cfg = kwargs.pop('train_cfg')
        assert isinstance(train_cfg, dict)
        if 'type' in train_cfg:
            assert train_cfg.pop('type') is MultiRunnerTrainLoop
        self._train_loop = MultiRunnerTrainLoop.__new__(MultiRunnerTrainLoop)
        kwargs.update({'train_cfg': self._train_loop})
        for model, optim_wrapper, param_scheduler, cfg in zip(models, optim_wrappers, param_schedulers, cfgs):
            if last_time is not None and (time.time() - last_time < 1):
                time.sleep(1)  # avoid different runners having the same timestamp
            runner = Runner(
                model=model,
                optim_wrapper=optim_wrapper,
                param_scheduler=param_scheduler,
                work_dir= f'{work_dir}/{model.__class__.__name__}',
                **kwargs)
            last_time = time.time()
            self.runners.append(runner)
            cfg = Config(cfg, format_python_code=False)
            cfg_save_path_pyext = runner.log_dir + '/' + runner.timestamp + '.py'
            cfg_save_path = runner.log_dir + '/' + runner.timestamp + '.cfg'
            cfg.dump(cfg_save_path_pyext)
            shutil.move(cfg_save_path_pyext, cfg_save_path)
        self._train_loop.__init__(
            runners=self.runners,
            dataloader=self.runners[0]._train_dataloader,
            **train_cfg)

    @property
    def train_loop(self):
        """:obj:`MultiRunnerTrainLoop`: A loop to run training."""
        return self._train_loop

    def train(self):
        """Launch training."""
        for runner in self.runners:
            if is_model_wrapper(runner.model):
                ori_model = runner.model.module
            else:
                ori_model = runner.model
            assert hasattr(ori_model, 'train_step'), (
                'If you want to train your model, please make sure your model '
                'has implemented `train_step`.')

            if runner._val_loop is not None:
                assert hasattr(ori_model, 'val_step'), (
                    'If you want to validate your model, please make sure your '
                    'model has implemented `val_step`.')

            # `build_optimizer` should be called before `build_param_scheduler`
            #  because the latter depends on the former
            runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
            # Automatically scaling lr by linear scaling rule
            runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)

            if runner.param_schedulers is not None:
                runner.param_schedulers = runner.build_param_scheduler(  # type: ignore
                    runner.param_schedulers)  # type: ignore

            if runner._val_loop is not None:
                runner._val_loop = runner.build_val_loop(
                    runner._val_loop)  # type: ignore
            # TODO: add a contextmanager to avoid calling `before_run` many times
            runner.call_hook('before_run')

            # initialize the model weights
            runner._init_model_weights()

            # make sure checkpoint-related hooks are triggered after `before_run`
            # runner.load_or_resume()

            # Initiate inner count of `optim_wrapper`.
            runner.optim_wrapper.initialize_count_status(
                runner.model,
                self._train_loop.iter,  # type: ignore
                self._train_loop.max_iters)  # type: ignore

        self.train_loop.run()  # type: ignore
        for runner in self.runners:
            runner.call_hook('after_run')


from torch.optim import Adam
from typing import List
import os

import torch.nn as nn
from mmengine.runner import Runner
from .optim.schedulers import CosineAnnealingLinearWarmupRestarts

from .dataset import ConcatTrainDataset, ConcatValDataset, ConcatTestDataset
from .dataloader import get_train_dataloader, get_test_dataloader, get_val_dataloader
from .hooks import CheckpointHook
from .metrics import WordAccuracy
from .mmwrapper import convert_to_mmmodel, convert_lrscheduler_to_paramscheduler


def get_multimodel_runer(
    models: List[nn.Module],
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
    save_failure_cases=False,
    train_sampler=None,
    cfg_dicts: List[dict]=None):

    if cfg_dicts is not None:
        assert len(cfg_dicts) == len(models)
    else:
        cfg_dicts = [dict() for _ in models]

    word_size = os.environ.get("WORLD_SIZE")
    launcher=('pytorch' if word_size else 'none')
    models = [convert_to_mmmodel(model) for model in models]
    min_lr = min_lr_ratio * lr

    default_hooks = dict()

    if train_dataset is not None:
        assert train_batch_size > 0
        train_dataloader = get_train_dataloader(train_dataset, train_batch_size, num_train_workers, sampler=train_sampler)
        max_iters = max_epochs * len(train_dataloader)
        optimizers = [Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-09) for model in models]
        param_schedulers = [CosineAnnealingLinearWarmupRestarts(optimizer, max_iters, max_lr=lr, min_lr=min_lr_ratio, warmup_steps=min(10000, max_iters // 10))
            for optimizer in optimizers]
        for cfg_dict, optimizer, param_scheduler in zip(cfg_dicts, optimizers, param_schedulers):
            cfg_dict.update(dict(optimizer=optimizer, scheduler=param_scheduler))
        param_schedulers = list(map(convert_lrscheduler_to_paramscheduler, param_schedulers))
        with_amp = False
        if with_amp:
            optim_wrappers = [dict(type=AmpOptimWrapper, dtype='float16', optimizer=optimizer, clip_grad=dict(max_norm=5)) for optimizer in optimizers]
        else:
            optim_wrappers = [dict(optimizer=optimizer, clip_grad=dict(max_norm=5)) for optimizer in optimizers]
        train_cfg = dict(max_epochs=max_epochs, val_by_epoch=False, val_interval=1000)
        default_hooks.update(dict(
            checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=len(train_dataloader), max_keep_ckpts=3, save_best='avg/word_acc', rule='greater', top_k=3)
        ))
    else:
        train_dataloader = None
        optimizers = None
        param_schedulers = None
        optim_wrappers = None
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
    runner = MultimodelRunner(
        models=models,
        work_dir=work_dir,
        default_hooks=default_hooks,
        # train cfg
        train_dataloader=train_dataloader,
        optim_wrappers=optim_wrappers,
        param_schedulers=param_schedulers,
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
        launcher=launcher,
        cfgs=cfg_dicts
    )
    return runner
