import header
import torch
import numpy as np


def test_localvim_encoder_tiny():
    from models.mambastr.mambastr_encoder import localvim_encoder_tiny
    img_size = np.array([16, 16])
    patch_size = np.array([2, 2])
    grid_size = img_size // patch_size
    embed_dim = 64
    model = localvim_encoder_tiny(
        img_size=tuple(img_size),
        patch_size=tuple(patch_size),
        embed_dim=embed_dim,
        depth=4,
        if_abs_pos_embed=False,
        fixed_token_size=True).cuda()
    # B, C, H, W = 36 * 80, 3, *img_size
    B, C, H, W = 1, 3, *img_size
    x = torch.rand([B, C, H, W]).cuda()
    assert model(x).shape == (B, grid_size[0] * grid_size[1], embed_dim)


def test_multimamba_encoder():
    from models.mambastr.mambastr_encoder import multimamba_encoder
    img_size = np.array([15, 15])
    patch_size = np.array([3, 3])
    grid_size = img_size // patch_size
    model = multimamba_encoder(
        img_size=tuple(img_size),
        patch_size=tuple(patch_size),
        embed_dim=64,
        depth=1,
        if_abs_pos_embed=False,
        fixed_token_size=True
    ).cuda()
    # B, C, H, W = 36 * 200, 3, *img_size
    B, C, H, W = 1, 3, *img_size
    x = torch.rand([B, C, H, W]).cuda()
    assert model(x).shape == (B, grid_size[0] * grid_size[1], model.embed_dim)


def test_multimamba_encoder_moe():
    from models.mambastr.mambastr_encoder import multimamba_encoder
    img_size = np.array([15, 15])
    patch_size = np.array([3, 3])
    grid_size = img_size // patch_size
    model = multimamba_encoder(
        img_size=tuple(img_size),
        patch_size=tuple(patch_size),
        embed_dim=64,
        depth=1,
        if_abs_pos_embed=False,
        fixed_token_size=True,
        with_moe=True
    ).cuda()
    # B, C, H, W = 36 * 200, 3, *img_size
    B, C, H, W = 1, 3, *img_size
    x = torch.rand([B, C, H, W]).cuda()
    assert model(x).shape == (B, grid_size[0] * grid_size[1], model.embed_dim)
