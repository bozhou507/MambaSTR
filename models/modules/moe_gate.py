import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List


class SwitchGate(nn.Module):
    """Copy paste from <https://github.com/kyegomez/MoE-Mamba/blob/main/moe_mamba/model.py#L7>
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor. Shape [B, L, D]

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        batch_size, seq_len, _ = x.shape
        x = x.flatten(0, -2)
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            raise NotImplemented
            # 下面的代码中 load 与 importance 的 shape 不同，会报错。
            # load = gate_scores.sum(0)  # Sum over all examples
            # importance = gate_scores.sum(1)  # Sum over all experts

            # # Aux loss is mean suqared difference between load and importance
            # loss = ((load - importance) ** 2).mean()

            # return gate_scores.reshape(batch_size, seq_len, -1), loss

        return gate_scores.reshape(batch_size, seq_len, -1), None


class MoEGate(nn.Module):
    """
    MoEGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor. Shape [B, L, D]

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        return F.softmax(self.w_gate(x), dim=-1)


class MixExperts(nn.Module):

    def __init__(
        self,
        emb_dim: int,  # 输入维度
        num_experts: int,  # 专家数量
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.w_gate = nn.Sequential(
            nn.Linear(emb_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, outs: List[torch.Tensor]):
        """
        
        Args:
            x: shape of [B, L, D]
            outs: out[i].shape is [B, L, D]
        
        Returns:
            shape of [B, L, D]
        """
        scores = self.w_gate(x)
        outs = torch.stack(outs)
        out = torch.einsum("n b l d, b l n -> b l d", outs, scores)
        return out

    def count_extra_flops(m, args, output):
        """count extra flops which has not covered by forward() of children modules"""
        x = args[0]
        N = m.num_experts
        _, L, D = x.shape
        m.total_ops = m.total_ops # for code completion
        m.total_ops += N * L * D * 2


if __name__ == '__main__':
    model = SwitchGate(192, 8).cuda(1)
    x = torch.rand([3, 10, 192]).cuda(1)
    gate_scores, loss = model(x)
    print(gate_scores.shape)  # (3, 10, 8)
    # print(gate_scores)

    
    model2 = MoEGate(192, 8).cuda(1)
    gate_scores = model2(x)
    print(gate_scores.shape)  # (3, 10, 8)
    print(gate_scores)
