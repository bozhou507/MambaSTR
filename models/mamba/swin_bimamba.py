import torch
import torch.nn as nn
from .bimamba_simple import BiMamba, Block, post_add_norm
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn
from timm.layers import DropPath
from typing import Tuple, Literal, List, Optional
import torch.nn.functional as F

from models.modules.moe_gate import MixExperts


class ChainInitParams:

    def __init__(self) -> None:
        self.pre_permute = None


def get_permute_ids(
    mask: torch.Tensor,
    img_emb_size: Tuple[int, int],
    window_scan_direction: Literal['rf', 'cf']
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assume that img_emb is numbered as follows:

    0 3 | 6  9
    1 4 | 7 10
    2 5 | 8 11

    The result of scanning the window row first is:

    0 3 1 4 2 5 | 6 9 7 10 8 11
    """
    assert window_scan_direction == 'rf'
    h, w = img_emb_size
    ids = torch.arange(h * w).reshape(h, w).transpose(0, 1).reshape(-1)
    end_ids = torch.where(mask == 0)[0].tolist() + [h * w]
    if end_ids[0] == 0:
        end_ids = end_ids[1:]
    start_ids = [0] + end_ids[:-1]
    def get_rf_res(s: int, e: int):
        if s % h == 0 and e % h == 0:
            return ids[s:e].reshape(-1, h).transpose(0, 1).reshape(-1)
        padl_len = s % h
        padr_len = -e % h
        res = F.pad(ids[s:e], [padl_len, padr_len], value=-1).reshape(-1, h).transpose(0, 1).reshape(-1)
        return res[res >= 0]
    permute_ids = torch.cat([get_rf_res(s, e) for s, e in zip(start_ids, end_ids)])

    # rpermute_ids = torch.argsort(permute_ids)
    rpermute_ids = torch.empty(h * w, dtype=torch.int64)
    rpermute_ids[permute_ids] = torch.arange(h * w)

    return permute_ids, rpermute_ids


class SwinBiMamba(nn.Module):
    """BiMamba performs global interaction by default, while SwinBiMamba supports
    local interaction for windows of window_size. To avoid the problem of window 
    boundary faults, each SwinBiMamba will perform two window-based interactions, 
    with the window size unchanged. However, the boundary will be offset by half 
    of the window size by default, and the window offset can be customized through 
    window_offset.

    Args:
        window_size: size of the interaction window (number of tokens)
        seq_len: length of the input sequence
        window_offset: number of tokens offset of the window relative to the first 
            BiMamba when passing the second BiMamba layer, default is half of the 
            interaction window size
        img_emb_size: image size corresponding to the sequence, this information 
            is needed to correctly divide the window when the input is column-first
        window_scan_direction: 'rf' means the input sequence is expanded row-first, 
            'cf' means the input sequence is expanded column-first
        drop_path_rates: sampling rate of drop_path, two BiMamba blocks correspond 
            to two drop_path_rate
        chain_init_params: the order of tokens needs to be frequently shuffled and 
            restored during sequence transmission, this parameter is used to combine 
            the previous restoration arrangement with the current shuffle arrangement 
            to save one reordering cost
    """

    def __init__(
        self,
        window_size: int,
        seq_len: int,
        window_offset: Optional[int]=None,
        mamba_cls=BiMamba,
        img_emb_size: Tuple[int,int]=None,
        window_scan_direction: Literal['rf', 'cf']='cf',
        mamba_cls_kwargs: dict=None,
        drop_path_rates: List[float]=[0, 0],
        norm_cls: bool=RMSNorm,
        chain_init_params: ChainInitParams=ChainInitParams(),  # To chain permute_ids during initialization
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if window_size > 0:
            assert window_size >= 2 and window_size <= seq_len
            assert window_scan_direction in ['rf', 'cf']  # ["row first", "column first"]
            mask1 = torch.ones(seq_len, dtype=torch.float)
            mask1[0::window_size] = 0
            mask2 = torch.ones(seq_len, dtype=torch.float)
            if window_offset is None:
                window_offset = window_size // 2
            assert 0 < window_offset < window_size
            mask2[window_offset::window_size] = 0
            self.register_buffer("mask1", mask1)
            self.register_buffer("mask2", mask2)
        else:
            self.mask1 = None
            self.mask2 = None

        _mamba_cls_kwargs = dict()
        if mamba_cls_kwargs is not None:
            _mamba_cls_kwargs.update(mamba_cls_kwargs)

        dim = _mamba_cls_kwargs.pop('d_model')
        self.residual_in_fp32 = True
        self.fused_add_norm = True
        self.mamba_layers = nn.ModuleList([Block(
            dim,
            mixer_cls=mamba_cls,
            norm_cls=norm_cls,
            fused_add_norm=self.fused_add_norm,
            residual_in_fp32=self.residual_in_fp32,
            drop_path=(0 if i == 0 else drop_path_rates[0]),
            bimamba_kwargs=_mamba_cls_kwargs) for i in range(2)])
        drop_path_rate = drop_path_rates[-1]
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm = norm_cls(dim)

        self.hidden_states_need_permute = (window_size > 0 and window_scan_direction != 'cf')
        pre_permute = chain_init_params.pre_permute
        chain_init_params.pre_permute = None
        if self.hidden_states_need_permute:
            permute_ids1, rpermute_ids1 = get_permute_ids(mask1, img_emb_size, window_scan_direction=window_scan_direction)
            permute_ids2, rpermute_ids2 = get_permute_ids(mask2, img_emb_size, window_scan_direction=window_scan_direction)
            if pre_permute is not None:
                permute_ids1 = torch.index_select(pre_permute, 0, permute_ids1)
                pre_permute = None
            permute_ids2 = torch.index_select(rpermute_ids1, 0, permute_ids2)
            self.register_buffer("permute_ids1", permute_ids1)
            self.register_buffer("permute_ids2", permute_ids2)
            chain_init_params.pre_permute = rpermute_ids2
        elif pre_permute is not None:
            raise NotImplementedError

    def forward(self, x):
        hidden_states, residual = x, None
        for layer_idx, mamba_layer in enumerate(self.mamba_layers, start=1):
            if self.hidden_states_need_permute:
                permute_ids = self.get_buffer(f'permute_ids{layer_idx}')
                hidden_states = torch.index_select(hidden_states, 1, permute_ids)
                if residual is not None:
                    residual = torch.index_select(residual, 1, permute_ids)
            mask = getattr(self, f'mask{layer_idx}')
            hidden_states, residual = mamba_layer(hidden_states, residual, mask=mask)

        return post_add_norm(
            hidden_states,
            residual,
            norm_module=self.norm,
            drop_path_fn=self.drop_path,
            fused_add_norm=self.fused_add_norm)

    def count_extra_flops(m, args, output):
        """count extra flops which has not covered by forward() of children modules"""
        m.total_ops = m.total_ops # for code completion
        x, = args
        L, D = x.shape[1:]
        m.total_ops = m.total_ops # for code completion
        if m.hidden_states_need_permute:
            m.total_ops += L * D * 2


class ChainPostprocessLayer(nn.Module):

    def __init__(self, chain_init_params: ChainInitParams=ChainInitParams(), **kwargs):
        super().__init__(**kwargs)
        pre_permute = chain_init_params.pre_permute
        chain_init_params.pre_permute = None
        self.hidden_states_need_permute = False
        if pre_permute is not None:
            self.hidden_states_need_permute = True
            self.register_buffer("permute_ids", pre_permute)
    
    def forward(self, x: torch.Tensor):
        """x: [B, L, D]"""
        if self.hidden_states_need_permute:
            return x.index_select(1, self.permute_ids)
        return x


class SwinBiMambaExperts(nn.Module):
    """Different experts are responsible for windows of different sizes"""

    def __init__(
        self,
        emb_dim: int,  # Dimensions of the input sequence
        img_emb_size: Tuple[int, int],  # The shape of the image corresponding to the input sequence
        window_scan_direction: Literal['rf', 'cf'],  # Scan direction of input sequence
        window_sizes: List[int],  # The window size of each expert
        window_offsets: List[int],  # Window bias for each expert
        drop_path_rates: Tuple[float, float],  # The drop_path_rate of the two BiMamba layers corresponding to each expert, all experts share a set of drop_path_rates
        norm_cls=RMSNorm,
        with_moe=False,  # whether BiMamba use MOE fusion or not
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_experts = len(window_sizes)
        assert self.num_experts > 0
        assert len(window_offsets) == self.num_experts
        assert len(drop_path_rates) == 2
        self.in_proj = nn.Linear(emb_dim, emb_dim * self.num_experts)
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            chain_init_params = ChainInitParams()
            self.experts.append(nn.Sequential(
                SwinBiMamba(
                    seq_len=img_emb_size[0] * img_emb_size[1],
                    img_emb_size=img_emb_size,
                    window_size=window_sizes[i],
                    window_offset=window_offsets[i],
                    window_scan_direction=window_scan_direction,
                    chain_init_params=chain_init_params,
                    drop_path_rates=drop_path_rates,
                    norm_cls=norm_cls,
                    mamba_cls=BiMamba,
                    mamba_cls_kwargs=dict(d_model=emb_dim, with_moe=with_moe)),
                ChainPostprocessLayer(chain_init_params=chain_init_params)
            ))
        self.moe_gate = MixExperts(emb_dim=emb_dim, num_experts=self.num_experts)

    def forward(self, x: torch.Tensor):
        """
        
        Args:
            x: shape of [B, L, D]
        """
        xs = self.in_proj(x).chunk(self.num_experts, dim=-1)
        outs = []
        for i in range(self.num_experts):
            outs.append(self.experts[i](xs[i]))
        return self.moe_gate(x, outs)


class SwinBiMambaExpertsBlock(nn.Module):

    def __init__(
        self,
        emb_dim: int,  # Dimensions of the input sequence
        img_emb_size: Tuple[int, int],  # The shape of the image corresponding to the input sequence
        window_scan_direction: Literal['rf', 'cf'],  # Scan direction of input sequence
        window_sizes: List[int],  # The window size of each expert
        window_offsets: List[int],  # Window bias for each expert
        drop_path_rates: Tuple[float, float],  # The drop_path_rate of the two BiMamba layers corresponding to each expert, all experts share a set of drop_path_rates
        with_moe=False,  # whether BiMamba use MOE fusion or not
        fused_add_norm=True,
        residual_in_fp32=True,
        norm_cls=RMSNorm,
        *args, **kwargs
    ):
        super().__init__()
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = SwinBiMambaExperts(
            emb_dim=emb_dim,
            img_emb_size=img_emb_size,
            window_scan_direction=window_scan_direction,
            window_sizes=window_sizes,
            window_offsets=window_offsets,
            drop_path_rates=drop_path_rates,
            norm_cls=norm_cls,
            with_moe=with_moe,
            *args, **kwargs
        )
        self.drop_path = DropPath(drop_path_rates[-1])
        self.norm = norm_cls(emb_dim)

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]=None):
        """
        
        Args:
            hidden_states: shape of [B, L, D]
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                self.drop_path(hidden_states) if residual is not None else hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual