import torch
import torch.nn as nn
from typing import Literal

from .dictionary import Dictionary
from .losses import CrossEntropyLoss, CTCLoss
from .postprocessor import ctc_postprocessor, attn_postprocessor, indices2word


class ForwardTextRecognitionLogits(nn.Module):

    def __init__(
            self, 
            loss_type: Literal['ce', 'attn', 'ctc'],
            dictionary: Dictionary,
            reduction: Literal['none', 'mean']='mean'):
        super().__init__()
        assert loss_type in ['ce', 'attn', 'ctc'], f'Unexpected loss_type: {loss_type}'
        self.loss_type = loss_type
        self.dictionary = dictionary
        self.softmax = nn.Softmax(dim=-1)
        self.cross_entropy_loss = CrossEntropyLoss(self.dictionary.padding_idx, reduction=reduction)
        self.ctc_loss = CTCLoss(self.dictionary.padding_idx, zero_infinity=True)

    def forward_ctc(self, logits: torch.Tensor, tgt: torch.Tensor=None, batch_weights: torch.Tensor=None):
        if tgt is None:
            assert self.training is False
            return ctc_postprocessor(self.softmax(logits), self.dictionary)
        if self.training:
            ctc_loss = self.ctc_loss(logits, tgt)
            if batch_weights is not None:
                raise NotImplemented
                ctc_loss = ctc_loss.reshape(batch_weights.size(0), -1).mean(-1)
                ctc_loss = (ctc_loss * batch_weights).sum()
            return dict(ctc_loss=ctc_loss)
        else:
            tgt = tgt.detach().cpu()
            return (ctc_postprocessor(self.softmax(logits), self.dictionary),
                    indices2word(tgt, self.dictionary))

    def forward_ce(self, logits: torch.Tensor, tgt: torch.Tensor=None, batch_weights: torch.Tensor=None):
        if tgt is None:
            assert self.training is False
            return attn_postprocessor(logits, self.dictionary)
        if self.training:
            ce_loss=self.cross_entropy_loss(logits, tgt)
            if batch_weights is not None:
                raise NotImplemented
                ce_loss = ce_loss.reshape(batch_weights.size(0), -1).mean(-1)
                ce_loss = (ce_loss * batch_weights).sum()
            return dict(ce_loss=ce_loss)
        else:
            words = attn_postprocessor(logits, self.dictionary)
            gt = indices2word(tgt.detach().cpu(), self.dictionary)
            return words, gt

    def forward_attention(self, logits: torch.Tensor, tgt: torch.Tensor=None, batch_weights: torch.Tensor=None):
        if tgt is None:
            assert self.training is False
            return attn_postprocessor(logits[:, :-1].contiguous(), self.dictionary)
        if self.training:
            logits = logits[:, :-1].contiguous()
            tgt = tgt[:, 1:].contiguous()
            attn_loss = self.cross_entropy_loss(logits, tgt)
            if batch_weights is not None:
                effective_lens = (tgt != self.dictionary.padding_idx).sum(-1)
                attn_loss = attn_loss.reshape(batch_weights.size(0), -1).sum(-1) / effective_lens
                attn_loss = (attn_loss * batch_weights).sum()
            return dict(attn_loss=attn_loss)
        else:
            words = attn_postprocessor(logits[:, :-1].contiguous(), self.dictionary)
            gt = indices2word(tgt[:, 1:].contiguous().detach().cpu(), self.dictionary)
            return words, gt

    def forward(self, logits: torch.Tensor, tgt: torch.Tensor=None, batch_weights: torch.Tensor=None, loss_type=None):
        if loss_type is None:
            loss_type = self.loss_type
        if loss_type == 'ctc':
            return ForwardTextRecognitionLogits.forward_ctc(self, logits, tgt, batch_weights)
        if loss_type == 'ce':
            return ForwardTextRecognitionLogits.forward_ce(self, logits, tgt, batch_weights)
        if loss_type == 'attn':
            return ForwardTextRecognitionLogits.forward_attention(self, logits, tgt, batch_weights)
        raise ValueError(f'Unexpected loss_type: {loss_type}')