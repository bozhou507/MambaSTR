import torch
import torch.nn as nn


class CrossEntropyLoss:

    def __init__(self, padding_idx: int, label_smoothing: float=.1, reduction='mean'):
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=padding_idx,
            reduction=reduction,
            label_smoothing=label_smoothing)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross Entropy Loss

        Args:
            logits: A raw logit tensor of shape :math:`(B, T, C)`, 
        where B is batch size, T is sequence length and C is channel num.
            labels: A tensor of shape :math:`(B, T)`.
        
        Returns:
            loss: A float Tensor
        """
        n_class = logits.size(-1)
        logits = logits.view(-1, n_class)
        labels = labels.view(-1)
        return self.cross_entropy(logits, labels)


class CTCLoss:

    def __init__(self, padding_idx, zero_infinity=False):
        self.padding_idx = padding_idx
        self.ctcloss = nn.CTCLoss(
            blank=padding_idx,
            reduction='mean',
            zero_infinity=zero_infinity)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Connectionist Temporal Classification Loss

        Args:
            logits: A raw logit tensor of shape :math:`(B, T, C)`, where B is batch size,
        T is input sequence length and C is channel num.
            labels: A tensor of shape :math:`(B, S)`, where S is the max target length.
            dictionary: A Dictionary instance.
        
        Returns:
            loss: A float Tensor
        """
        B, T, _ = logits.shape
        log_probs = logits.log_softmax(dim=-1)
        log_probs = log_probs.permute(1, 0, 2).contiguous()
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
        padding_positions = (labels == self.padding_idx)
        _, label_lengths = padding_positions.max(dim=1)
        return self.ctcloss(log_probs, labels, input_lengths, label_lengths)
