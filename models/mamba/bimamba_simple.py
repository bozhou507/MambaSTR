# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    import warnings
    warnings.warn('bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None')
    bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    import warnings
    warnings.warn('RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None')
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class BiMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v2",  # bimamba v2
        if_devide_out=True,  # devide by 2
        init_layer_scale=None,
        with_moe=False
    ):
        assert bimamba_type in ['v1', 'v2'], f'unknown bimamba_type {bimamba_type}'
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2":
            assert self.use_fast_path
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.with_moe = with_moe
        if self.with_moe:
            assert bimamba_type == 'v2'
            self.num_experts = 2
            self.w_gate = nn.Sequential(
                nn.Linear(self.d_model, self.num_experts),
                nn.Softmax(dim=-1)
            )

    def forward(self, hidden_states, mask=None, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        assert inference_params is None, "Doesn't support outputting the states"
        if self.with_moe:
            gate_scores = self.w_gate(hidden_states)
        else:
            gate_scores = None
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path:
            if self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                out = bimamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    mask=mask
                )    
            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    mask=mask
                )
                mask_b = torch.roll(mask, -1).flip([-1]) if mask is not None else None
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                    mask=mask_b
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                if self.with_moe:
                    outs = torch.stack([out, out_b.flip([-1])])
                    out = torch.einsum("n b d l, b l n -> b l d", outs, gate_scores)
                    out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
                elif not self.if_devide_out:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                else:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)
        
        else:
            raise NotImplemented('Only fast path is supported for now.')

        if self.init_layer_scale is not None:
                out = out * self.gamma    
        return out

    def count_extra_flops(m, args, output):
        """count extra flops which has not covered by forward() of children modules"""
        hidden_states, = args
        L = hidden_states.size(1)
        N = m.d_state
        # m.total_ops = m.total_ops for code completion
        # modified from https://github.com/hunto/LocalMamba/blob/main/classification/lib/models/local_vim.py#L420
        # 1 in_proj
        m.total_ops += m.in_proj.in_features * m.in_proj.out_features * L
        if m.in_proj.bias is not None:
            m.total_ops += m.in_proj.out_features * L
        # 2 MambaInnerFnNoOutProj
        mamba_inner_ops = 0
        # 2.1 causual conv1d
        mamba_inner_ops += (L + m.d_conv - 1) * m.d_inner * m.d_conv
        # 2.2 x_proj
        mamba_inner_ops += L * m.x_proj.in_features * m.x_proj.out_features
        # 2.3 dt_proj
        mamba_inner_ops += L * m.dt_proj.in_features * m.dt_proj.out_features
        # 2.4 selective scan
        """
        u: r(B D L)
        delta: r(B D L)
        A: r(D N)
        B: r(B N L)
        C: r(B N L)
        D: r(D)
        z: r(B D L)
        delta_bias: r(D), fp32
        """
        D = m.d_inner
        # A
        mamba_inner_ops += D * L * N
        # B
        mamba_inner_ops += D * L * N * 2
        # C
        mamba_inner_ops += (D * N + D * N) * L
        # D
        mamba_inner_ops += D * L
        # Z
        mamba_inner_ops += D * L
        m.total_ops += mamba_inner_ops * 2
        # 2.5 out_proj
        m.total_ops += L * m.out_proj.in_features * m.out_proj.out_features

        if m.with_moe:
            n = 2
            B = 1
            # outs: [n, B, D, L]
            # gate_scores [B, L, n]
            # torch.einsum("n b d l, b l n -> b l d", outs, gate_scores)
            m.total_ops += n * B * L * D + n * B * L * D


from timm.layers import DropPath
from typing import Union


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls=BiMamba, norm_cls: Union[str, nn.Module]=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,
        drop_path=0., bimamba_kwargs=dict(),
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        norm_cond_msg = 'Only LayerNorm and RMSNorm are supported for fused_add_norm'
        if isinstance(norm_cls, str):
            assert norm_cls in ['LayerNorm', 'RMSNorm'], norm_cond_msg + f', not {norm_cls}.'
            if norm_cls == 'RMSNorm':
                norm_cls = RMSNorm
            else:
                norm_cls = nn.LayerNorm
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim, **bimamba_kwargs)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            norm_cls_name = self.norm.__class__.__name__
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), norm_cond_msg + f', not {norm_cls_name}.'

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states) if residual is not None else hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **kwargs)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def post_add_norm(
    hidden_states: Tensor,
    residual: Optional[Tensor],
    norm_module: nn.Module,
    drop_path_fn: Optional[nn.Module]=nn.Identity(),
    fused_add_norm: bool=False,
) -> torch.Tensor:
    if not fused_add_norm:
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + drop_path_fn(hidden_states)
        hidden_states = norm_module(residual.to(dtype=norm_module.weight.dtype))
    else:
        fused_add_norm_fn = rms_norm_fn if isinstance(norm_module, RMSNorm) else layer_norm_fn
        hidden_states = fused_add_norm_fn(
            drop_path_fn(hidden_states) if residual is not None else hidden_states,
            norm_module.weight,
            norm_module.bias,
            residual=residual,
            prenorm=False,  # Set prenorm=False here since we don't need the residual
            eps=norm_module.eps,
        )
    return hidden_states


from functools import partial


class BiMambaBlock(Block):

    def __init__(
        self,
        dim: int,
        eps=1e-5,
        fused_add_norm=True,
        residual_in_fp32=True,
        **kwargs,
    ):
        mixer_cls=partial(BiMamba, **kwargs)
        norm_cls = partial(nn.LayerNorm if RMSNorm is None else RMSNorm, eps=eps)
        super().__init__(
            dim,
            mixer_cls=mixer_cls,
            norm_cls=norm_cls,
            drop_path=0.,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32)
