import logging
from typing import Dict, List, Tuple
from mmengine.runner import Runner, BaseLoop
from torch.utils.data import DataLoader
from typing import Sequence
from mmengine.logging import print_log

class TrainLoop(BaseLoop):

    def __init__(
            self,
            runner: Runner,
            dataloader: DataLoader,
            max_epochs: int=None,
            max_iters: int=None,
            val_by_epoch: bool=False,
            val_begin: int=1,
            val_interval: int=1,
            val_last: bool=True):
        super().__init__(runner, dataloader)
        self.iters_per_epoch = len(self.dataloader)
        assert max_epochs is not None or max_iters is not None, 'both `max_epochs` and `max_iters` are None'
        if max_epochs is not None:
            assert isinstance(max_epochs, int), f'`max_epochs` should be a integer number, but get {max_epochs}.'
            self._max_epochs = max_epochs
            self._max_iters = self._max_epochs * self.iters_per_epoch
            if max_iters is not None:
                assert self._max_iters == max_iters, '`max_epochs` should be consistent with `max_iters`'
        else:
            assert isinstance(max_iters, int), f'`max_iters` should be a integer number, but get {max_iters}'
            self._max_iters = max_iters
            assert (max_iters % len(self.dataloader) == 0), f'`max_iters` should be divisible by len(self.dataloader)={len(self.dataloader)}, but get {max_iters}'
            self._max_epochs = max_iters // self.iters_per_epoch
        self._epoch = 0
        self._iter = 0
        if val_by_epoch:
            if val_interval > self.max_epochs:
                print_log(f'val_by_epoch is True but `val_interval` > `max_epochs`, no validation will be performed', logger='current', level=logging.WARNING)
        elif val_interval > self.max_iters:
                print_log(f'val_by_epoch is False but `val_interval` > `max_iters`, no validation will be performed', logger='current', level=logging.WARNING)
        self.val_by_epoch = val_by_epoch
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.val_last = val_last
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self):
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

        if (self.runner.val_loop is not None
                and self.val_last
                and ((self.val_by_epoch and self.max_epochs % self.val_interval != 0)
                     or (not self.val_by_epoch and self.max_iters % self.val_interval != 0))):
            self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for data_batch in self.dataloader:
            self.run_iter(data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

        if (self.runner.val_loop is not None
                and self.val_by_epoch
                and self._epoch >= self.val_begin
                and self._epoch % self.val_interval == 0):
            self.runner.val_loop.run()

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        batch_idx = self._iter % self.iters_per_epoch
        self.runner.call_hook('before_train_iter', batch_idx=batch_idx, data_batch=data_batch)
        self.runner.model.train()
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

        if (self.runner.val_loop is not None
                and not self.val_by_epoch
                and self._iter >= self.val_begin
                and self._iter % self.val_interval == 0):
            self.runner.val_loop.run()


class MultiRunnerTrainLoop(BaseLoop):

    def __init__(
            self,
            runners: List[Runner],
            dataloader: DataLoader,
            max_epochs: int=None,
            max_iters: int=None,
            val_by_epoch: bool=False,
            val_begin: int=1,
            val_interval: int=1,
            val_last: bool=True):
        super().__init__(runners[0], dataloader)
        self._runner = None
        self._runners = runners
        self.iters_per_epoch = len(self.dataloader)
        assert max_epochs is not None or max_iters is not None, 'both `max_epochs` and `max_iters` are None'
        if max_epochs is not None:
            assert isinstance(max_epochs, int), f'`max_epochs` should be a integer number, but get {max_epochs}.'
            self._max_epochs = max_epochs
            self._max_iters = self._max_epochs * self.iters_per_epoch
            if max_iters is not None:
                assert self._max_iters == max_iters, '`max_epochs` should be consistent with `max_iters`'
        else:
            assert isinstance(max_iters, int), f'`max_iters` should be a integer number, but get {max_iters}'
            self._max_iters = max_iters
            assert (max_iters % len(self.dataloader) == 0), f'`max_iters` should be divisible by len(self.dataloader)={len(self.dataloader)}, but get {max_iters}'
            self._max_epochs = max_iters // self.iters_per_epoch
        self._epoch = 0
        self._iter = 0
        if val_by_epoch:
            if val_interval > self.max_epochs:
                print_log(f'val_by_epoch is True but `val_interval` > `max_epochs`, no validation will be performed', logger='current', level=logging.WARNING)
        elif val_interval > self.max_iters:
                print_log(f'val_by_epoch is False but `val_interval` > `max_iters`, no validation will be performed', logger='current', level=logging.WARNING)
        self.val_by_epoch = val_by_epoch
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.val_last = val_last
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self):
        """Launch training."""
        for runner in self._runners:
            runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

        if (self.val_last
            and ((self.val_by_epoch and self.max_epochs % self.val_interval != 0)
            or (not self.val_by_epoch and self.max_iters % self.val_interval != 0))):
            for runner in self._runners:
                if runner.val_loop is not None:
                    runner.val_loop.run()

        for runner in self._runners:
            runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        for runner in self._runners:
            runner.call_hook('before_train_epoch')
            runner.model.train()
        for data_batch in self.dataloader:
            self.run_iter(data_batch)

        for runner in self._runners:
            runner.call_hook('after_train_epoch')
        self._epoch += 1

        if (self.val_by_epoch
            and self._epoch >= self.val_begin
            and self._epoch % self.val_interval == 0):
            for runner in self._runners:
                if runner.val_loop is not None:
                    runner.val_loop.run()

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        batch_idx = self._iter % self.iters_per_epoch
        for runner in self._runners:
            runner.call_hook('before_train_iter', batch_idx=batch_idx, data_batch=data_batch)
            runner.model.train()
            # Enable gradient accumulation mode and avoid unnecessary gradient
            # synchronization during gradient accumulation process.
            # outputs should be a dict of loss.
            outputs = runner.model.train_step(data_batch, optim_wrapper=runner.optim_wrapper)

            runner.call_hook(
                'after_train_iter',
                batch_idx=batch_idx,
                data_batch=data_batch,
                outputs=outputs)
        self._iter += 1

        if (not self.val_by_epoch
            and self._iter >= self.val_begin
            and self._iter % self.val_interval == 0):
            for runner in self._runners:
                if runner.val_loop is not None:
                    runner.val_loop.run()
