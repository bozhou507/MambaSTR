import torch
from torch import nn
from utils.dictionary import get_dictionary_for_attn
from utils.taskutil import SceneTextRecognitionTask
from utils.position_embedding import get_sinusoid_encoding_table_2d
from models.modules.patch_embedding import PatchEmbed
from models.preprocessor.stn import STN
from models.mamba.swin_bimamba import SwinBiMamba, SwinBiMambaExperts, ChainInitParams, ChainPostprocessLayer
from .bimamba_encoder import bimamba_encoder_tiny
from .mamba_attntion_decoder import MambaAttentionDecoder
from . import dataset
from typing import Literal, List, Optional, Tuple


class MambaSTR(SceneTextRecognitionTask):
    """Parameter design principle: By default, the latter parameters will not affect 
    the behavior of the former parameters. After the user passes the parameters, the 
    latter parameters are given priority."""

    def __init__(
        self,
        emb_dim,
        swinmamba_depth=3,
        window_size=8 * 4,
        window_offsets: Optional[List[int]]=None,
        bimamba_depth=12,
        decoder_mamba_depth_per_layer=12,
        with_moe=False,
        patch_size=(4, 8),
        anchors: list=None,
        with_stn: bool=False,
        with_swinbimamba_experts: bool=False,
        window_sizes_and_offsets_list: List[Tuple[List[int], List[int]]]=None,
        drop_prob: float=.1,
        autoregressive: bool=True,
        loss_type_: Literal['ce', 'attn', 'ctc']=None,
        with_raw_image: bool=False,
    ):
        self.patch_size = patch_size
        if anchors is None or len(anchors) == 0:
            anchors = [(8, 16)]  # default img_size (32, 128)
        self.fixshape_anchor = anchors[0]
        self.num_patches = self.fixshape_anchor[0] * self.fixshape_anchor[1]
        self.emb_dim = emb_dim
        self.with_moe = with_moe
        self.with_stn = with_stn
        loss_type = 'attn' if autoregressive else 'ce'
        if loss_type_ is not None:
            loss_type = loss_type_
        self.loss_type = loss_type
        super().__init__(
            img_size = (self.patch_size[0] * self.fixshape_anchor[0], self.patch_size[1] * self.fixshape_anchor[1]),
            loss_type=self.loss_type,
            reduction='mean')
        self.max_word_length = self.dictionary.max_word_length
        if self.with_stn:
            self.stn = STN(
                in_channels=3,
                resized_image_size=(32, 64),
                output_image_size=self.img_size,
                num_control_points=20,
                margins=[0.05, 0.05])
        self.num_encoder = 2
        # assert self.emb_dim % 2 == 0
        self.encoder_dim = self.emb_dim
        self.stacked_encoder_dim = self.num_encoder * self.encoder_dim
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, embed_dim=self.emb_dim)
        if self.with_stn:
            self.register_buffer("pos_embed", get_sinusoid_encoding_table_2d(*self.patch_embed.grid_size, self.emb_dim))
        if self.stacked_encoder_dim != self.emb_dim:
            self.fc_before_encoders = nn.Linear(self.emb_dim, self.stacked_encoder_dim)
        else:
            self.fc_before_encoders = None
        if self.encoder_dim != self.emb_dim:
            self.fc_after_encoders = nn.Linear(self.encoder_dim, self.emb_dim)
        else:
            self.fc_after_encoders = None
        dpr = [x.item() for x in torch.linspace(0, drop_prob, swinmamba_depth * 2 + bimamba_depth)]
        if window_offsets is None:
            window_offsets = [window_size // 2 for _ in range(swinmamba_depth)]
        else:
            assert len(window_offsets) == swinmamba_depth
        if with_swinbimamba_experts:
            assert len(window_sizes_and_offsets_list) == swinmamba_depth
            for i in range(swinmamba_depth):
                assert len(window_sizes_and_offsets_list[i]) == 2
                assert len(window_sizes_and_offsets_list[i][0]) == len(window_sizes_and_offsets_list[i][1])
        def get_encoder(dim: int, window_scan_direction: Literal['rf', 'cf']):
            if with_swinbimamba_experts:
                return nn.Sequential(*([
                    SwinBiMambaExperts(
                        emb_dim=self.emb_dim,
                        img_emb_size=self.patch_embed.grid_size,
                        window_scan_direction=window_scan_direction,
                        window_sizes=window_sizes_and_offsets_list[i][0],
                        window_offsets=window_sizes_and_offsets_list[i][1],
                        drop_path_rates=dpr[i * 2 : i * 2 + 2],
                        with_moe=self.with_moe
                    ) for i in range(swinmamba_depth)]
                    + [bimamba_encoder_tiny(embed_dim=dim, depth=bimamba_depth, with_moe=self.with_moe, drop_path_rate=dpr[swinmamba_depth * 2:])]))
            else:
                chain_init_params = ChainInitParams()
                return nn.Sequential(*([
                    SwinBiMamba(
                        img_emb_size=self.patch_embed.grid_size,
                        window_size=window_size,
                        window_offset=window_offsets[i],
                        window_scan_direction=window_scan_direction,
                        seq_len=self.num_patches,
                        chain_init_params=chain_init_params,
                        drop_path_rates=dpr[i * 2: i * 2 + 2],
                        mamba_cls_kwargs=dict(d_model=dim, with_moe=self.with_moe))
                    for i in range(swinmamba_depth)]
                    + [ChainPostprocessLayer(chain_init_params=chain_init_params)]
                    + [bimamba_encoder_tiny(embed_dim=dim, depth=bimamba_depth, with_moe=self.with_moe, drop_path_rate=dpr[swinmamba_depth * 2:])]))
        self.encoder1 = get_encoder(self.encoder_dim, window_scan_direction='cf')
        self.encoder2 = get_encoder(self.encoder_dim, window_scan_direction='rf')
        self.decoder = MambaAttentionDecoder(
            version=0,
            parallel_decode=not autoregressive,
            dictionary=self.dictionary,
            embed_dim=self.emb_dim,
            num_heads=self.emb_dim // 32,
            mamba_depth_per_layer=decoder_mamba_depth_per_layer)
        self.w_gate = nn.Sequential(
            nn.Linear(self.emb_dim, self.num_encoder),
            nn.Softmax(dim=-1)
        )
        self.with_raw_image = with_raw_image

    def get_image_folder_dataset(self, folder_path: str, training: bool=False):
        return dataset.get_image_folder_dataset(folder_path, (64, 256), with_raw_image=self.with_raw_image, training=training)

    def get_ocr_lmdb_dataset(self, lmdb_path: str, training: bool):
        return dataset.get_ocr_lmdb_dataset(lmdb_path, (64, 256), self.dictionary, with_raw_image=self.with_raw_image, training=training)

    def get_ocr_lmdb_datasets(self, lmdb_paths: str, training: bool):
        return [self.get_ocr_lmdb_dataset(lmdb_path, training) for lmdb_path in lmdb_paths]

    def _forward_embedding(self, x: torch.Tensor, tgt: torch.Tensor = None, *, return_logits=False, return_attn_map=False):
        B = x.size(0)
        x = x.reshape(B, *self.patch_embed.grid_size, self.emb_dim)
        if self.fc_before_encoders is not None:
            x = self.fc_before_encoders(x)
        x1, x2 = torch.chunk(x, self.num_encoder, dim=-1)
        x1 = x1.transpose(1, 2).reshape(B, -1, self.encoder_dim)
        x2 = x2.reshape(B, -1, self.encoder_dim)
        out_enc1 = self.encoder1(x1)
        out_enc2 = self.encoder2(x2)
        out_enc2 = out_enc2.reshape(B, *self.patch_embed.grid_size, self.encoder_dim).transpose(1, 2).reshape(B, -1, self.encoder_dim)
        out_enc = torch.stack([out_enc1, out_enc2])  # 2, B, L, D
        gate_scores = self.w_gate(x1)
        out_enc = torch.einsum("n b l d, b l n -> b l d", out_enc, gate_scores)
        if self.fc_after_encoders is not None:
            out_enc = self.fc_after_encoders(out_enc)

        logits, attn_map = self.decoder(out_enc, tgt, return_attn_map=return_attn_map)
        out = logits if return_logits else self.postprocessor(logits, tgt)
        if return_attn_map:
            return out, attn_map
        else:
            return out

    def _forward(self, x: torch.Tensor, tgt: torch.Tensor = None, *, return_logits=False, return_attn_map=False):
        """
        
        Args:
            x: [B, C, H, W]
            pos_emb: [B, num_patches, E]
            tgt: target
        """
        if self.with_stn:
            x = self.stn(x)
        x = self.patch_embed(x)  # (B, self.num_patches, embed_dim)
        x = x + self.get_buffer("pos_embed")
        return self._forward_embedding(x, tgt, return_logits=return_logits, return_attn_map=return_attn_map)

    def forward(self, *args, **kwargs):
        start_idx = 0
        for arg in args:
            if not isinstance(arg, list):
                break
            start_idx += 1
        args = list(args[start_idx:])
        return self._forward(*args, **kwargs)

    def count_extra_flops(m, args, output):
        """count extra flops which has not covered by forward() of children modules"""
        B, L, D = 1, m.num_patches, m.encoder_dim
        N = m.num_encoder
        m.total_ops = m.total_ops # for code completion

        # out_enc: [N, B, L, D]
        # gate_scores [B, L, N]
        # torch.einsum("n b l d, b l n -> b l d", out_enc, gate_scores)
        m.total_ops += N * B * L * D + N * B * L * D


def mambastr_tiny(save_failure_cases=False) -> MambaSTR:
    cls = type('mambastr_tiny', (MambaSTR, ), {})
    swinbimambaexperts_depth = 3
    window_sizes_and_offsets_list = [([8 * 4, 8 * 8], [8 * 2, 8 * 4])] * swinbimambaexperts_depth
    return cls(
        emb_dim=128,
        anchors=[(8, 16)],
        with_stn=True,
        with_swinbimamba_experts=True,
        window_sizes_and_offsets_list=window_sizes_and_offsets_list,
        swinmamba_depth=swinbimambaexperts_depth,
        bimamba_depth=12,
        decoder_mamba_depth_per_layer=1,
        with_moe=True,
        with_raw_image=save_failure_cases)


def mambastr_small(save_failure_cases=False) -> MambaSTR:
    cls = type('mambastr_small', (MambaSTR, ), {})
    swinbimambaexperts_depth = 3
    window_sizes_and_offsets_list = [([8 * 4, 8 * 8], [8 * 2, 8 * 4])] * swinbimambaexperts_depth
    return cls(
        emb_dim=160,
        anchors=[(8, 16)],
        with_stn=True,
        with_swinbimamba_experts=True,
        window_sizes_and_offsets_list=window_sizes_and_offsets_list,
        swinmamba_depth=swinbimambaexperts_depth,
        bimamba_depth=12,
        decoder_mamba_depth_per_layer=1,
        with_moe=True,
        with_raw_image=save_failure_cases)


def mambastr(save_failure_cases=False) -> MambaSTR:
    cls = type('mambastr', (MambaSTR, ), {})
    swinbimambaexperts_depth = 3
    window_sizes_and_offsets_list = [([8 * 4, 8 * 8], [8 * 2, 8 * 4])] * swinbimambaexperts_depth
    return cls(
        emb_dim=192,
        anchors=[(8, 16)],
        with_stn=True,
        with_swinbimamba_experts=True,
        window_sizes_and_offsets_list=window_sizes_and_offsets_list,
        swinmamba_depth=swinbimambaexperts_depth,
        decoder_mamba_depth_per_layer=1,
        with_moe=True,
        with_raw_image=save_failure_cases)