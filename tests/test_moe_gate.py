import header
import torch
from models.modules.moe_gate import MixExperts


def test_mix_experts():
    B, L, emb_dim = 2, 5, 10
    num_experts = 3
    moe_gate = MixExperts(emb_dim=emb_dim, num_experts=num_experts).cuda()
    x = torch.rand([B, L, emb_dim]).cuda()
    outs = [torch.rand([B, L, emb_dim]).cuda() for _ in range(num_experts)]
    out = moe_gate(x, outs)
    assert tuple(out.shape) == (B, L, emb_dim)