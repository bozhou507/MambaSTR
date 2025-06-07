import torch
from torch import nn
from models.mamba.mamba_simple import Block as MambaBlock, Mamba, RMSNorm, layer_norm_fn
from models.modules.attention import MultiheadAttention
from models.modules.ffn import PositionwiseFeedForward
from utils.dictionary import Dictionary
from utils.position_embedding import get_sinusoid_encoding_table_1d
from mamba_ssm.utils.generation import InferenceParams
from functools import partial


class MambaAttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        mamba_depth,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        drop_prob: float=.1,
        norm_epsilon: float=1e-5,
        fused_add_norm: bool=True,
        residual_in_fp32: bool=True,
        version: int=9999_9999,
    ):
        super().__init__()
        self.version = version
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        rms_norm_cls = partial(RMSNorm, eps=norm_epsilon)
        self.mamba_layers = nn.ModuleList([MambaBlock(
            dim=embed_dim,
            mixer_cls=Mamba,
            layer_idx=layer_idx,
            mlp_cls=nn.Identity,
            norm_cls=rms_norm_cls,
            fused_add_norm=self.fused_add_norm,
            residual_in_fp32=self.residual_in_fp32
        ) for layer_idx in range(mamba_depth)])
        self.rms_norm = rms_norm_cls(embed_dim)
        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=norm_epsilon)
        self.ffn = PositionwiseFeedForward(d_model=embed_dim, hidden=embed_dim * mlp_ratio, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=norm_epsilon)

    def forward(
        self,
        enc: torch.Tensor,
        query: torch.Tensor,
        inference_params=None
    ):
        """

        Args:
            enc: embedding from encoder. shape (B, L, E)
            tgt: groud truth label. shape (B, max_word_length, E)
        """
        # mamba_layers
        residual = None
        for mamba_layer in self.mamba_layers:
            query, residual = mamba_layer(query, residual, inference_params=inference_params)
        ## add & norm
        assert residual is not None
        if not self.fused_add_norm:
            residual = residual + query
            query = self.rms_norm(residual.to(dtype=self.rms_norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            query, residual = layer_norm_fn(
                query,
                self.rms_norm.weight,
                self.rms_norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.rms_norm.eps,
                is_rms_norm=isinstance(self.rms_norm, RMSNorm)
            )

        # attention
        if self.version < 2024_1216:
            residual = query
        query, attn_map = self.attention(query, enc, enc, return_attn_map=True)
        query = self.dropout1(query)

        # ffn
        if self.version < 2024_1216:
            query = self.norm1(residual + query)
            residual = query
        else:
            residual = residual + query
            query = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

        query = self.ffn(query)
        query = self.dropout2(query)
        residual = residual + query
        query = self.norm2(residual.to(dtype=self.norm2.weight.dtype))

        return query, attn_map


class MambaAttentionDecoder(nn.Module):
    """
    
    Args:
        parallel_decode: If set to True, decode in parallel, otherwise decode autoregressively.
    """

    def __init__(
        self,
        dictionary: Dictionary,
        embed_dim: int=192,
        num_heads: int=6,
        mamba_depth_per_layer: int=12,
        depth: int=1,
        parallel_decode: bool=False,
        drop_prob: float=.1,
        norm_epsilon: float=1e-5,
        fused_add_norm: bool=True,
        residual_in_fp32: bool=True,
        version: int=9999_9999,
    ):
        super().__init__()
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.dictionary = dictionary
        self.embed_dim = embed_dim
        self.parallel_decode = parallel_decode
        self.max_word_length = self.dictionary.max_word_length
        if not self.parallel_decode:
            self.tok_emb = nn.Embedding(self.dictionary.num_classes, self.embed_dim, self.dictionary.padding_idx)
        self.register_buffer('pos_emb', get_sinusoid_encoding_table_1d(self.max_word_length, self.embed_dim))
        self.dropout = nn.Dropout(p=0.1)
        self.layers = nn.ModuleList([
            MambaAttentionDecoderLayer(
                mamba_depth=mamba_depth_per_layer,
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                drop_prob=drop_prob,
                norm_epsilon=norm_epsilon,
                fused_add_norm=self.fused_add_norm,
                residual_in_fp32=self.residual_in_fp32,
                version=version) for _ in range(depth)
        ])
        self.linear = nn.Linear(self.embed_dim, self.dictionary.num_classes)
        if self.parallel_decode:
            self.query_embed = nn.Parameter(torch.empty(1, self.max_word_length, self.embed_dim))
            nn.init.trunc_normal_(self.query_embed, std=.02)

    def decode(self, enc: torch.Tensor, query: torch.Tensor=None, return_attn_map: bool=False, inference_params: InferenceParams=None):
        if not self.parallel_decode:
            query = self.tok_emb(query)
        else:
            B = enc.size(0)
            query = query.expand(B, -1, -1)
        if inference_params:
            query = self.dropout(query + self.pos_emb[inference_params.seqlen_offset, :])
        else:
            query = self.dropout(query + self.pos_emb)
        for layer in self.layers:
            query, attn_map = layer(enc, query, inference_params=inference_params)
        if return_attn_map:
            return self.linear(query), attn_map
        else:
            return self.linear(query), None

    def forward(
        self,
        enc: torch.Tensor,
        tgt: torch.Tensor=None,
        return_attn_map: bool=False,
    ):
        """

        Args:
            enc: embedding from encoder. shape (B, L, E)
            tgt: groud truth label. shape (B, max_word_length, E)
            return_attn_map: return attention map
        """
        if self.parallel_decode:
            return self.decode(enc, self.query_embed, return_attn_map=return_attn_map)
        if tgt is None:
            B = enc.shape[0]
            inference_params = InferenceParams(max_seqlen=self.max_word_length, max_batch_size=B)
            last_indice = torch.full((B, 1), fill_value=self.dictionary.start_idx, dtype=torch.int, device=enc.device)
            logit_list = []
            if return_attn_map:
                attn_map_list = []
            for seqlen_offset in range(self.dictionary.max_word_length):
                inference_params.seqlen_offset = seqlen_offset
                if return_attn_map:
                    last_logit, attn_map = self.decode(enc, last_indice, return_attn_map=True, inference_params=inference_params)
                    attn_map_list.append(attn_map)
                else:
                    last_logit, _ = self.decode(enc, last_indice, return_attn_map=False, inference_params=inference_params)
                logit_list.append(last_logit)
                last_indice = torch.argmax(last_logit.squeeze(1), dim=-1, keepdim=True)
            logits = torch.cat(logit_list, 1)
            if return_attn_map:
                attn_maps = torch.cat(attn_map_list, 1)
                return logits, attn_maps
            else:
                return logits, None
        else:
            return self.decode(enc, tgt, return_attn_map=return_attn_map)
