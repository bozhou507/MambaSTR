import torch


def get_sinusoid_encoding_table_1d(num_pos, emb_dim) -> torch.Tensor:
    """Sinusoid position encoding table 1D.
    
    Return shape: (num_pos, emb_dim)
    """
    sinusoid_table_1d = torch.empty(num_pos, emb_dim, dtype=torch.float)
    pos = torch.arange(0, num_pos, dtype=torch.float).unsqueeze(dim=1)
    _2i = torch.arange(0, emb_dim, step=2, dtype=torch.float)
    sinusoid_table_1d[:, 0::2] = torch.sin(pos / (10000 ** (_2i / emb_dim)))
    sinusoid_table_1d[:, 1::2] = torch.cos(pos / (10000 ** (_2i / emb_dim)))
    return sinusoid_table_1d


def get_sinusoid_encoding_table_2d(h, w, emb_dim):
    """Sinusoid position encoding table 2D.
    
    Return shape: (h * w, emb_dim)
    """
    assert emb_dim % 2 == 0, 'emb_dim must be divided by 2'
    h_sinusoid_table_1d = get_sinusoid_encoding_table_1d(h, emb_dim // 2)
    w_sinusoid_table_1d = get_sinusoid_encoding_table_1d(w, emb_dim // 2)
    h_sinusoid_table_2d = h_sinusoid_table_1d.unsqueeze(1).repeat([1, w, 1]).reshape([h * w, -1])
    w_sinusoid_table_2d = w_sinusoid_table_1d.repeat([h, 1])
    sinusoid_table_2d = torch.cat([h_sinusoid_table_2d, w_sinusoid_table_2d], dim=1)
    return sinusoid_table_2d
