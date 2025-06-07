import header
import torch


def test_mamba_attention_decoder():
    from utils.dictionary import get_dictionary_for_attn
    from models.mambastr.mamba_attntion_decoder import MambaAttentionDecoder
    model = MambaAttentionDecoder(dictionary=get_dictionary_for_attn()).cuda()
    max_word_length = model.dictionary.max_word_length
    num_classes = model.dictionary.num_classes
    B, L, D = 3, 10, 192
    enc = torch.rand([B, L, D]).cuda()
    tgt = torch.randint(0, num_classes - 1, (B, max_word_length)).cuda()
    logits, _ = model(enc, tgt)
    assert logits.shape == (B, max_word_length, num_classes)
    torch.sum(logits).backward()
