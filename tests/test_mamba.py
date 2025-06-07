import header
import torch

from models.mamba.mamba_simple import Mamba

def test_mamba():
    model = Mamba(d_model=192).cuda()
    E = 192
    # The input batch size and sequence length can be inconsistent
    N, L = 10, 20
    x = torch.rand([N, L, E]).cuda()
    for mask in [
        None,
        torch.tensor((([1] * 6 + [0]) * 3)[:-1], dtype=torch.float).cuda()
    ]:
        if mask is not None:
            assert len(mask) == L
            y = model(x, mask=mask)
        else:
            y = model(x)
        torch.sum(y).backward()
        assert y.shape == (N, L, E)
        # The input batch size and sequence length can be inconsistent
    N, L = 5, 30
    x = torch.rand([N, L, E]).cuda()
    y = model(x)
    assert y.shape == (N, L, E)


def test_mamba_step():
    model = Mamba(d_model=192, layer_idx=0).cuda()
    E = 192
    N, L = 10, 20
    x = torch.rand([N, L, E]).cuda()
    # mask = None
    mask = torch.tensor((([1] * 6 + [0]) * 3)[:-1], dtype=torch.float).cuda()
    # mask = torch.tensor([1] * 19 + [0], dtype=torch.float).cuda()
    # mask = torch.tensor([0] * 19 + [0], dtype=torch.float).cuda()
    if mask is not None:
        y = model(x, mask=mask)
    else:
        y = model(x)
    assert y.shape == (N, L, E)
    *states, = model.allocate_inference_cache(N)
    steped_y = []
    for i in range(L):
        if mask is not None and mask[i] == 0:
            states[1].zero_()
        in_embedding = x[:, i:i+1, :]
        out_embedding, *_ = model.step(in_embedding, *states)
        steped_y.append(out_embedding)
    steped_y = torch.cat(steped_y, 1)
    assert steped_y.shape == y.shape
    assert ((y - steped_y).abs() < 1e-6).all()  # The results of the two decoding methods are consistent

    from mamba_ssm.utils.generation import InferenceParams
    inference_params = InferenceParams(max_seqlen=L, max_batch_size=N)
    steped_y2 = []
    for i in range(L):
        in_embedding = x[:, i:i+1, :]
        # *states, = model._get_states_from_cache(inference_params, N)
        # out_embedding, *_ = model.step(in_embedding, *states)
        inference_params.seqlen_offset = 1  # Greater than 0 is ok. Normally, it should be set to i
        if mask is not None:
            out_embedding = model(in_embedding, inference_params=inference_params, mask=mask[i:i+1])
        else:
            out_embedding = model(in_embedding, inference_params=inference_params)
        steped_y2.append(out_embedding)
    steped_y2 = torch.cat(steped_y2, 1)
    assert steped_y.shape == steped_y2.shape
    # assert ((steped_y - steped_y2).abs() < 1e-6).all(), f'{(steped_y - steped_y2).abs().max()}'  # The results of the two decoding methods are consistent
    assert (steped_y == steped_y2).all(), f'{(steped_y - steped_y2).abs().max()}'  # The results of the two decoding methods are consistent

    params = [x for x in model.parameters() if x.requires_grad]
    def clean_param():
        for p in params:
            p.grad = None
    def sum_grad():
        s = 0
        for p in params:
            if p.grad is not None:
                s += torch.mean(p.grad)
        return s

    # assert sum_grad() == 0
    # torch.sum(y).backward()
    # sumgrad = sum_grad()
    # assert sumgrad != 0

    # clean_param()
    # torch.sum(steped_y).backward()
    # sumgrad1 = sum_grad()
    # assert ((sumgrad1 - sumgrad).abs() < 1e-6).all(), f'{(sumgrad1 - sumgrad).abs().max()}'
    # # assert (sumgrad1 == sumgrad).all(), f'{(sumgrad1 - sumgrad).abs().max()}'

    # clean_param()
    # torch.sum(steped_y2).backward()
    # sumgrad2 = sum_grad()
    # assert (sumgrad1 == sumgrad2).all(), f'{(sumgrad1 - sumgrad2).abs().max()}'
