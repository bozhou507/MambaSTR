from mmengine.hooks.checkpoint_hook import (
    CheckpointHook as _CheckpointHook,
    FileClient, get_file_backend, deque,
    is_main_process)
from mmengine.runner import Runner
from . import osutil
from .bisectutil import bisect_right
from typing import Literal, Union, List
import torch


class CheckpointHook(_CheckpointHook):

    def __init__(
        self,
        by_epoch: bool=False,
        interval: int=-1,
        max_keep_ckpts: int=-1,
        save_best: Union[str, List[str], None]=None,
        rule: Literal['greater', 'less']=None,
        top_k: Union[int, List[int]]=1,
    ):
        super().__init__(
            interval=interval,
            by_epoch=by_epoch,
            max_keep_ckpts=max_keep_ckpts,
            save_best=save_best,
            rule=rule)
        self.dynamic_path = None
        self.top_k = top_k
        self.save_abnormal_inf_loss_ckpt = True
        self.save_abnormal_nan_loss_ckpt = True
        if self.save_best:
            assert by_epoch == False
            if isinstance(self.save_best, str):
                assert isinstance(self.top_k, int)
                assert self.top_k > 0
            else:
                assert len(self.save_best) >= 1
                if len(self.save_best) > 1:
                    if isinstance(self.top_k, int):
                        self.top_k = [self.top_k] * len(self.save_best)
                    assert len(self.save_best) == len(self.top_k)
                    for top_k in self.top_k:
                        assert top_k > 0
                elif len(self.save_best) == 1:
                    assert len(self.top_k) == 1
                    self.save_best = self.save_best[0]
                    self.top_k = self.top_k[0]
                    assert self.top_k > 0
                else:
                    self.save_best = None
                    import warnings
                    warnings.warn('CheckpointHook: save_best is [], no best checkpoint will be saved.')

    def remove_ckpt(self, ckpt_path: str) -> bool:
        is_removed = False
        if ckpt_path and is_main_process():
            if self.file_backend.isfile(ckpt_path):
                self.file_backend.remove(ckpt_path)
                is_removed = True
            elif self.file_backend.isdir(ckpt_path):
                # checkpoints saved by deepspeed are directories
                self.file_backend.rmtree(ckpt_path)
                is_removed = True
        return is_removed

    def before_train(self, runner: Runner) -> None:
        if self.out_dir is None:
            self.out_dir = runner.log_dir

        if self.file_client_args is None:
            self.file_backend = get_file_backend(self.out_dir, backend_args=self.backend_args)
        else:
            self.file_backend = FileClient.infer_client(self.file_client_args, self.out_dir)

        runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                self.best_ckpts = []  # [(acc, path), (acc, path), ...]
            else:
                self.best_ckpts_dict = dict()
                for key in self.key_indicators:
                    self.best_ckpts_dict[key] = []

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = []
            self.keep_ckpt_ids: deque = deque(keep_ckpt_ids, self.max_keep_ckpts)

    def after_train_iter(self, runner: Runner, *args, **kwargs):
        # Temporarily change by_epoch
        by_epoch_tmp = self.by_epoch
        self.by_epoch = False
        # Temporarily change the checkpoint file name template
        filename_tmpl_tmp = self.filename_tmpl
        self.filename_tmpl = 'dynamic_iter_{}.pth'
        dynamic_save_ok = False
        loss = kwargs['outputs']['loss']
        # save_abnormal_inf_loss_ckpt
        if self.save_abnormal_inf_loss_ckpt and torch.isinf(loss):
            abnormal_type = 'inf'
            self.filename_tmpl = 'abnormal_' + abnormal_type + '_loss_iter_{}.pth'
            runner.logger.info(f'Saving abnormal loss checkpoint at {runner.iter + 1} iterations')
            step = runner.iter + 1
            meta = dict(epoch=runner.epoch, iter=step)
            self._save_checkpoint_with_step(runner, step, meta=meta)
            dynamic_save_ok = True
            self.save_abnormal_inf_loss_ckpt = False
        # save_abnormal_nan_loss_ckpt
        if self.save_abnormal_nan_loss_ckpt and torch.isnan(loss):
            abnormal_type = 'nan'
            self.filename_tmpl = 'abnormal_' + abnormal_type + '_loss_iter_{}.pth'
            runner.logger.info(f'Saving abnormal loss checkpoint at {runner.iter + 1} iterations')
            step = runner.iter + 1
            meta = dict(epoch=runner.epoch, iter=step)
            self._save_checkpoint_with_step(runner, step, meta=meta)
            dynamic_save_ok = True
            self.save_abnormal_nan_loss_ckpt = False
        try:
            if self.dynamic_path is None:
                self.dynamic_path = osutil.concat_paths([runner.work_dir, 'dynamic_save_ckpt'])
            # self.dynamic_path stores the iteration interval for temporarily saving checkpoints
            if osutil.is_path_exist(self.dynamic_path) and not dynamic_save_ok:
                with open(self.dynamic_path, 'r') as f:
                    iter_interval_str = f.read()
                if iter_interval_str:
                    try:
                        iter_interval = int(iter_interval_str)
                    except:
                        runner.logger.info(f'Expecting integer, got {iter_interval_str}')
                    else:
                        # If iter_interval is read successfully, it is ready to save
                        if self.every_n_train_iters(runner, iter_interval, self.save_begin):
                            runner.logger.info(f'Dynamically saving checkpoint at {runner.iter + 1} iterations')
                            step = runner.iter + 1
                            meta = dict(epoch=runner.epoch, iter=step)
                            self._save_checkpoint_with_step(runner, step, meta=meta)
                            dynamic_save_ok = True
        except:
            # If an exception is caused by improper operation, it should not affect the normal progress of training.
            runner.logger.warning('dynamic_save_ckpt failed')
        # Resume by_epoch
        self.by_epoch = by_epoch_tmp
        # Resume checkpoint file name template
        self.filename_tmpl = filename_tmpl_tmp
        if not dynamic_save_ok:
            return super().after_train_iter(runner, *args, **kwargs)

    def _save_best_checkpoint(self, runner, metrics):
        if not self.save_best:
            return

        ckpt_filename = self.filename_tmpl.format(runner.iter)
        cur_type, cur_time = 'iter', runner.iter
        
        meta = dict(epoch=runner.epoch, iter=runner.iter)

        # save best logic
        # get score from messagehub
        def helper(key_indicator: str, rule: str, best_ckpts: list, top_k: int):
            assert rule in ['greater', 'less']
            key_score = metrics[key_indicator]
            ckpt_name = f'{key_indicator}_{key_score:0.4f}_{ckpt_filename}'

            new_item = (key_score, ckpt_name)
            key_func = lambda x: x[0]
            index = bisect_right(best_ckpts, key_func(new_item), key=key_func, reversed_order=(rule == 'greater'))
            if index == top_k:
                return

            if len(best_ckpts) == top_k:
                path = self.file_backend.join_path(self.out_dir, best_ckpts.pop()[1])
                if self.remove_ckpt(path):
                    runner.logger.info(f'The previous checkpoint {path} is removed')

            best_ckpts.insert(index, new_item)
            runner.save_checkpoint(
                self.out_dir,
                filename=ckpt_name,
                file_client_args=self.file_client_args,
                save_optimizer=False,
                save_param_scheduler=False,
                meta=meta,
                by_epoch=False,
                backend_args=self.backend_args)
            runner.logger.info(
                f'The checkpoint with top {index + 1} {key_indicator} {key_score:0.4f} '
                f'at {cur_time} {cur_type} is saved to {ckpt_name}.')
        if len(self.key_indicators) == 1:
            assert len(self.rules) == 1
            helper(self.key_indicators[0], self.rules[0], self.best_ckpts, self.top_k)
        else:
            for key_indicator, rule, top_k in zip(self.key_indicators, self.rules, self.top_k):
                helper(key_indicator, rule, self.best_ckpts_dict[key_indicator], top_k)

        # if self.last_ckpt is not None:
        #     self._save_checkpoint_with_step(runner, cur_time, meta)
