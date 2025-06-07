import header
import torch

from models.mamba.swin_bimamba import get_permute_ids, SwinBiMamba, SwinBiMambaExperts


def test_permute_ids():
    h, w = (3, 4)
    seq_len = h * w

    # print(torch.arange(seq_len).reshape(w, h).transpose(0, 1))
    # tensor([[ 0,  3,  6,  9],
    #         [ 1,  4,  7, 10],
    #         [ 2,  5,  8, 11]])

    raw_input = torch.arange(seq_len).reshape(w, h).transpose(0, 1).reshape(-1)
    raw_seq = raw_input.tolist()

    # mask = tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    mask = torch.ones(seq_len, dtype=torch.float)
    permute_ids, rpermute_ids = get_permute_ids(mask, img_emb_size=(h, w), window_scan_direction='rf')
    assert raw_input.index_select(0, permute_ids).tolist() == [ 0,  3,  6,  9,  1,  4,  7, 10,  2,  5,  8, 11]
    # assert raw_input.index_select(0, rpermute_ids).tolist.tolist() == [ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11]
    assert torch.index_select(raw_input, 0, permute_ids).index_select(0, rpermute_ids).tolist() == raw_seq
    assert torch.index_select(raw_input, 0, permute_ids.index_select(0, rpermute_ids)).tolist() == raw_seq

    # mask = tensor([1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.])
    mask[seq_len // 2] = 0
    permute_ids, rpermute_ids = get_permute_ids(mask, img_emb_size=(h, w), window_scan_direction='rf')
    assert raw_input.index_select(0, permute_ids).tolist() == [ 0,  3,  1,  4,  2,  5,  6,  9,  7, 10,  8, 11]
    # assert raw_input.index_select(0, rpermute_ids).tolist.tolist() == [ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11]
    assert torch.index_select(raw_input, 0, permute_ids).index_select(0, rpermute_ids).tolist() == raw_seq
    assert torch.index_select(raw_input, 0, permute_ids.index_select(0, rpermute_ids)).tolist() == raw_seq

    # mask = tensor([0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.])
    mask[0] = 0
    permute_ids, rpermute_ids = get_permute_ids(mask, img_emb_size=(h, w), window_scan_direction='rf')
    assert raw_input.index_select(0, permute_ids).tolist() == [ 0,  3,  1,  4,  2,  5,  6,  9,  7, 10,  8, 11]
    # assert raw_input.index_select(0, rpermute_ids).tolist.tolist() == [ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11]
    assert torch.index_select(raw_input, 0, permute_ids).index_select(0, rpermute_ids).tolist() == raw_seq
    assert torch.index_select(raw_input, 0, permute_ids.index_select(0, rpermute_ids)).tolist() == raw_seq

    # mask = tensor([1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1.])
    mask = torch.tensor([1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1.], dtype=torch.float)
    permute_ids, rpermute_ids = get_permute_ids(mask, img_emb_size=(h, w), window_scan_direction='rf')
    assert raw_input.index_select(0, permute_ids).tolist() == [ 0,  1,  2,  3,  6,  4,  7,  5,  8,  9, 10, 11]
    # assert raw_input.index_select(0, rpermute_ids).tolist.tolist() == [ 0,  1,  2,  3,  5,  7,  4,  6,  8,  9, 10, 11]
    assert torch.index_select(raw_input, 0, permute_ids).index_select(0, rpermute_ids).tolist() == raw_seq
    assert torch.index_select(raw_input, 0, permute_ids.index_select(0, rpermute_ids)).tolist() == raw_seq

    # mask = tensor([1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1.])
    mask = torch.tensor([1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1.], dtype=torch.float)
    permute_ids, rpermute_ids = get_permute_ids(mask, img_emb_size=(h, w), window_scan_direction='rf')
    assert raw_input.index_select(0, permute_ids).tolist() == [ 0,  1,  3,  6,  4,  7,  2,  5,  9, 10,  8, 11]
    # assert raw_input.index_select(0, rpermute_ids).tolist.tolist() == [ 0,  1,  6,  2,  4,  7,  3,  5, 10,  8,  9, 11]
    assert torch.index_select(raw_input, 0, permute_ids).index_select(0, rpermute_ids).tolist() == raw_seq
    assert torch.index_select(raw_input, 0, permute_ids.index_select(0, rpermute_ids)).tolist() == raw_seq


def test_swin_bimamba():
    E = 192
    N, L = 10, 20
    model = SwinBiMamba(
        window_size=6,
        seq_len=L,
        mamba_cls_kwargs=dict(d_model=E)).cuda()
    x = torch.rand([N, L, E]).cuda()
    y = model(x)
    torch.sum(y).backward()
    assert y.shape == (N, L, E)


def test_swin_bimamba_winsize0():
    E = 192
    N, L = 10, 20
    model = SwinBiMamba(
        window_size=0,
        seq_len=L,
        mamba_cls_kwargs=dict(d_model=E)).cuda()
    x = torch.rand([N, L, E]).cuda()
    y = model(x)
    torch.sum(y).backward()
    assert y.shape == (N, L, E)


def test_swin_bimamba_experts():
    B, L, D = 1, 128, 10
    model = SwinBiMambaExperts(
        emb_dim=D,
        img_emb_size=(8, 16),
        window_scan_direction='cf',
        window_sizes=[8 * 4, 8 * 8],
        window_offsets=[8 * 2, 8 * 4],
        drop_path_rates=(0, 0),
        with_moe=True
    ).cuda()
    x = torch.rand(B, L, D).cuda()
    y = model(x)
    torch.sum(y).backward()
    assert y.shape == (B, L, D)