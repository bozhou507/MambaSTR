from mmengine.model import BaseModel
from mmengine.optim import _ParamScheduler
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

_Model = type('_Model', (nn.Module,), {})


__all__ = ['convert_to_mmmodel', 'convert_lrscheduler_to_paramscheduler']


def convert_to_mmmodel(model: _Model):
    cls = type(model)

    class MMModel(cls, BaseModel):
        def forward(self, *args, **kwargs):
            kwargs.pop('mode')
            return cls.forward(self, *args, **kwargs)

    MMModel.__name__ = cls.__name__
    mmmodel = MMModel.__new__(MMModel)
    BaseModel.__init__(mmmodel)
    data_preprocessor = mmmodel.data_preprocessor
    mmmodel.__dict__.update(model.__dict__)
    if not hasattr(mmmodel, 'data_preprocessor'):
        mmmodel.data_preprocessor = data_preprocessor
    return mmmodel


def convert_lrscheduler_to_paramscheduler(scheduler: _LRScheduler):
    cls = type(scheduler)
    ParamScheduler_cls = type('ParamScheduler_cls', (cls, _ParamScheduler), {
        'by_epoch': False,
        'get_last_value': scheduler.get_last_lr,
        '_get_value': scheduler.get_lr})
    paramscheduler = ParamScheduler_cls.__new__(ParamScheduler_cls)
    paramscheduler.__dict__.update(scheduler.__dict__)
    return paramscheduler
