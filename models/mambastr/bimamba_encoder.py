from typing import Tuple, Union, List
from models.mamba.models_vim import VisionMamba


class BiMambaEncoder(VisionMamba):
    """A PyTorch implementation of Mamba
    """

    def __init__(
            self,
            img_size: Tuple[int, int]=(32, 128), 
            patch_size=16,
            stride=16,
            depth=24,
            embed_dim=192,
            channels=3,
            num_classes=1000,
            ssm_cfg=None,
            drop_rate=0,
            drop_path_rate=0.1,
            norm_epsilon: float=0.00001,
            rms_norm: bool=True,  # with rms_norm
            initializer_cfg=None,
            fused_add_norm=True,  # fused add norm
            residual_in_fp32=True,  # residual in fp32
            device=None,
            dtype=None,
            ft_seq_len=None,
            pt_hw_seq_len=14,
            if_bidirectional=False,
            final_pool_type='all',  # 输出全部 logits
            if_abs_pos_embed=False,
            if_rope=False,
            if_rope_residual=False,
            flip_img_sequences_ratio=-1,
            bimamba_type="v2",  # use bimamba v2
            if_cls_token=False,
            if_devide_out=True,  # devide out by 2
            init_layer_scale=None,
            use_double_cls_token=False,
            use_middle_cls_token=False,
            with_moe=False,
            **kwargs):
        super().__init__(
            img_size=img_size, patch_size=patch_size, stride=stride, depth=depth, embed_dim=embed_dim,
            channels=channels, num_classes=num_classes, ssm_cfg=ssm_cfg, drop_rate=drop_rate,
            drop_path_rate=drop_path_rate, norm_epsilon=norm_epsilon, rms_norm=rms_norm,
            initializer_cfg=initializer_cfg, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32,
            device=device, dtype=dtype, ft_seq_len=ft_seq_len, pt_hw_seq_len=pt_hw_seq_len,
            if_bidirectional=if_bidirectional, final_pool_type=final_pool_type, if_abs_pos_embed=if_abs_pos_embed,
            if_rope=if_rope, if_rope_residual=if_rope_residual, flip_img_sequences_ratio=flip_img_sequences_ratio,
            bimamba_type=bimamba_type, if_cls_token=if_cls_token, if_devide_out=if_devide_out,
            init_layer_scale=init_layer_scale, use_double_cls_token=use_double_cls_token, use_middle_cls_token=use_middle_cls_token,
            with_patch_embed=False, with_head=False, bimamba_kwargs=dict(with_moe=with_moe), **kwargs)

    def forward(self, x, return_features=True, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        return super().forward(x, return_features, inference_params, if_random_cls_token_position, if_random_token_rank)


def bimamba_encoder_tiny(
    embed_dim=192, depth=24, with_moe=False,
    drop_path_rate: Union[float, List[float]]=1.0,
):
    return BiMambaEncoder(
        embed_dim=embed_dim, depth=depth, drop_path_rate=drop_path_rate, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=False, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_devide_out=True, with_moe=with_moe)
