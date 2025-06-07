import header
import torch


def test_bimamba_simple():
    from models.mamba.bimamba_simple import BiMamba
    model = BiMamba(d_model=192).cuda()
    N, L, E = 10, 20, 192
    x = torch.rand([N, L, E]).cuda()
    y = model(x)
    assert y.shape == (N, L, E)


def test_bimamba_simple_mask():
    from models.mamba.bimamba_simple import BiMamba
    model = BiMamba(d_model=192).cuda()
    N, L, E = 10, 20, 192
    x = torch.rand([N, L, E]).cuda()
    for mask in [
        torch.zeros(L, dtype=torch.float).cuda(),
        torch.ones(L, dtype=torch.float).cuda(),
        torch.tensor((([1] * 6 + [0]) * 3)[:-1], dtype=torch.float).cuda()
    ]:
        assert len(mask) == L
        y = model(x, mask=mask)
        torch.sum(y).backward()
        assert y.shape == (N, L, E)


def test_bimamba_encoder_tiny():
    from models.mambastr.bimamba_encoder import bimamba_encoder_tiny
    model = bimamba_encoder_tiny().cuda()
    B, L, E = 3, 10, model.embed_dim
    x = torch.rand([B, L, E]).cuda()
    assert model(x).shape == (B, L, E)


def test_bimamba_encoder_moe_tiny():
    from models.mambastr.bimamba_encoder import bimamba_encoder_tiny
    model = bimamba_encoder_tiny(with_moe=True).cuda()
    B, L, E = 3, 10, model.embed_dim
    x = torch.rand([B, L, E]).cuda()
    assert model(x).shape == (B, L, E)
