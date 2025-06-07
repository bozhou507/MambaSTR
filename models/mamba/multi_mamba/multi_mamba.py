# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
import logging
from typing import List, Tuple


try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj
except ImportError:
    import warnings
    warnings.warn('mamba_inner_fn_no_out_proj = None')
    mamba_inner_fn_no_out_proj = None

from .local_scan import local_scan, local_scan_bchw, local_reverse


class MultiScan(nn.Module):

    # ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w7', 'w7_flip')
    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'ws', 'ws_flip', 'wl', 'wl_flip')

    def __init__(self, dim, choices=None, token_size=(14, 14)):
        super().__init__()
        self.token_size = token_size
        if choices is None:
            self.choices = MultiScan.ALL_CHOICES
            self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in self.choices])
            self.weights = nn.Parameter(1e-3 * torch.randn(len(self.choices), 1, 1, 1))
            self._iter = 0
            self.logger = logging.getLogger()
            self.search = True
        else:
            self.choices = choices
            self.search = False

    def forward(self, xs, gate_scores: torch.Tensor=None):
        """
        Input @xs: [[B, L, D], ...]
        Input @gate_scores: [B, L, len(choices)]
        """
        if self.search:
            weights = self.weights.softmax(0)
            xs = [norm(x) for norm, x in zip(self.norms, xs)]
            xs = torch.stack(xs) * weights
            x = xs.sum(0)
            if self._iter % 200 == 0:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    self.logger.info(str(weights.detach().view(-1).tolist()))
            self._iter += 1
        else:
            xs = torch.stack(xs)
            if gate_scores is not None:
                x = torch.einsum("n B L D, B L n -> B L D", xs, gate_scores)
            else:
                x = xs.sum(0)
        return x

    def multi_scan(self, x, token_sizes=None):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        for direction in self.choices:
            xs.append(self.scan(x, direction, hws=token_sizes))
        return xs

    def multi_reverse(self, xs, token_sizes=None):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction, hws=token_sizes))
        return new_xs

    def scan(self, x, direction='h', hws: List[Tuple[int,int]]=None):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        if self.token_size is None:
            assert hws is not None
            assert len(x.shape) == 3, f'got x.shape == {x.shape}'
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                batch_x = []
                for sample_x, (h, w) in zip(x, hws):
                    batch_x.append(rearrange(sample_x, '(h w) d -> d (w h)', h=h, w=w))
                return torch.stack(batch_x)
            elif direction == 'v_flip':
                batch_x = []
                for sample_x, (h, w) in zip(x, hws):
                    batch_x.append(rearrange(sample_x, '(h w) d -> d (w h)', h=h, w=w))
                return torch.stack(batch_x).flip([-1])
            elif direction.startswith('w'):
                K = direction[1:].split('_')[0]
                flip = direction.endswith('flip')
                batch_x = []
                for sample_x, (h, w) in zip(x, hws):
                    batch_x.append(local_scan(sample_x.unsqueeze(0), K, h, w, flip=False, allow_padding=False).squeeze(0))
                return torch.stack(batch_x).flip([-1]) if flip else torch.stack(batch_x)
            else:
                raise RuntimeError(f'Direction {direction} not found.')

        assert hws is None
        H, W = self.token_size
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = direction[1:].split('_')[0]
                flip = direction.endswith('flip')
                return local_scan(x, K, H, W, flip=flip)
                # return LocalScanTriton.apply(x.transpose(-2, -1), K, flip, H, W)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = direction[1:].split('_')[0]
                flip = direction.endswith('flip')
                return local_scan_bchw(x, K, H, W, flip=flip)
                # return LocalScanTriton.apply(x, K, flip, H, W).flatten(2)
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h', hws: List[Tuple[int,int]]=None):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        if self.token_size is None:
            assert hws is not None
            if direction == 'h':
                return x
            elif direction == 'h_flip':
                return x.flip([-1])
            elif direction == 'v':
                batch_x = []
                for sample_x, (h, w) in zip(x, hws):
                    batch_x.append(rearrange(sample_x, 'd (h w) -> d (w h)', h=h, w=w))
                return torch.stack(batch_x)
            elif direction == 'v_flip':
                x = x.flip([-1])
                batch_x = []
                for sample_x, (h, w) in zip(x, hws):
                    batch_x.append(rearrange(sample_x, 'd (h w) -> d (w h)', h=h, w=w))
                return torch.stack(batch_x)
            elif direction.startswith('w'):
                K = direction[1:].split('_')[0]
                flip = direction.endswith('flip')
                if flip:
                    x.flip([-1])
                batch_x = []
                for sample_x, (h, w) in zip(x, hws):
                    batch_x.append(local_reverse(sample_x.unsqueeze(0), K, h, w, flip=False).squeeze(0))
                return torch.stack(batch_x)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        
        assert hws is None
        H, W = self.token_size
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction.startswith('w'):
            K = direction[1:].split('_')[0]
            flip = direction.endswith('flip')
            return local_reverse(x, K, H, W, flip=flip)
            # return LocalReverseTriton.apply(x, K, flip, H, W)
        else:
            raise RuntimeError(f'Direction {direction} not found.')    
        
    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn #* s_attn  # [B, N, C]
        return ori_x * attn


class MultiMamba(nn.Module):
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
        bimamba_type="none",
        directions=None,
        token_size=(14, 14),
        use_middle_cls_token=False,
        with_moe=False
    ):
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
        self.token_size = token_size
        self.use_middle_cls_token = use_middle_cls_token

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()


        self.multi_scan = MultiScan(self.d_inner, choices=directions, token_size=token_size)
        '''new for search'''
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        for i in range(len(self.multi_scan.choices)):
            setattr(self, f'A_log_{i}', nn.Parameter(A_log))
            getattr(self, f'A_log_{i}')._no_weight_decay = True

            conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            setattr(self, f'conv1d_{i}', conv1d)

            x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            setattr(self, f'x_proj_{i}', x_proj)

            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
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
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

            setattr(self, f'dt_proj_{i}', dt_proj)

            D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            D._no_weight_decay = True
            setattr(self, f'D_{i}', D)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.with_moe = with_moe
        if self.with_moe:
            assert directions is not None
            self.num_experts = len(directions)
            self.w_gate = nn.Sequential(
                nn.Linear(self.d_model, self.num_experts),
                nn.Softmax(dim=-1)
            )
        else:
            self.attn = BiAttn(self.d_inner)

    def forward(self, hidden_states, token_sizes=None, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        if self.with_moe:
            gate_scores = self.w_gate(hidden_states)
        else:
            gate_scores = None
        xz = self.in_proj(hidden_states)

        if self.use_middle_cls_token:
            """
            Steps to use middle cls token
            # 1. split cls token out
            # 2. do 2d scan
            # 3. append cls token to the middle
            # 4. ssm
            # 5. split cls token out
            # 6. reverse tokens
            # 7. append cls token to the middle
            """
            cls_position = (xz.shape[1] - 1) // 2
            cls_token = xz[:, cls_position:cls_position+1]
            xz = torch.cat([xz[:, :cls_position], xz[:, cls_position+1:]], dim=1)

        xs = self.multi_scan.multi_scan(xz, token_sizes=token_sizes)  # [[BDL], [BDL], ...]
        if self.use_middle_cls_token:
            # step 3
            xs = [torch.cat([x[:, :, :cls_position], cls_token.transpose(-2, -1), x[:, :, cls_position:]], dim=2) for x in xs]

        outs = []
        for i, xz in enumerate(xs):
            # xz = rearrange(xz, "b l d -> b d l")
            A = -torch.exp(getattr(self, f'A_log_{i}').float())
            conv1d = getattr(self, f'conv1d_{i}')
            x_proj = getattr(self, f'x_proj_{i}')
            dt_proj = getattr(self, f'dt_proj_{i}')
            D = getattr(self, f'D_{i}')
            
            out = mamba_inner_fn_no_out_proj(
                xz,
                conv1d.weight,
                conv1d.bias,
                x_proj.weight,
                dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                D,
                delta_bias=dt_proj.bias.float(),
                delta_softplus=True,
            )
            outs.append(out)

        if self.use_middle_cls_token:
            # step 5
            new_outs = []
            cls_tokens = []
            for out in outs:
                cls_tokens.append(out[:, :, cls_position:cls_position+1])
                new_outs.append(torch.cat([out[:, :, :cls_position], out[:, :, cls_position+1:]], dim=2))
            outs = new_outs

        outs = self.multi_scan.multi_reverse(outs, token_sizes=token_sizes)

        if self.use_middle_cls_token:
            # step 7
            new_outs = []
            for out, cls_token in zip(outs, cls_tokens):
                new_outs.append(torch.cat([out[:, :, :cls_position], cls_token, out[:, :, cls_position:]], dim=2))
            outs = new_outs

        if self.with_moe:
            outs = [rearrange(out, 'b d l -> b l d') for out in outs]
        else:
            outs = [self.attn(rearrange(out, 'b d l -> b l d')) for out in outs]
        out = self.multi_scan(outs, gate_scores=gate_scores)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

        return out


from torch import Tensor
from timm.layers import DropPath
from typing import Optional

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    import warnings
    warnings.warn('RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None')
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
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
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, token_sizes=None, inference_params=None
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
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, token_sizes, inference_params=inference_params)
        return hidden_states, residual
