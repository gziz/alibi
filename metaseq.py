import math
import torch


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor (t) with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


decoder_attention_heads = 8
use_alibi = True

def build_alibi_tensor(max_seq_len: int, n_attention_heads: int):
    """Returns tensor shaped (n_head, 1, max_seq_len)"""

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(n_attention_heads))
    # (S, 1 ,1) * (H, 1, S) -> (H, 1, S)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * \
        torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(n_attention_heads, -1, -1) 
    # (H, 1, S) -> (H, 1, S)
    alibi = alibi.view(n_attention_heads, 1, max_seq_len)
    return alibi


def buffered_future_mask(tensor, input_tokens=None):
    cur_seq_len, batch_size = tensor.size(0), tensor.size(1)
    max_seq_len = 512 # hard coded !!!
    need_to_make_new_mask = True

    if need_to_make_new_mask:
        _future_mask = torch.triu(
            fill_with_neg_inf(
                torch.zeros([max_seq_len, max_seq_len], device=tensor.device)
            ), 1,)

        if use_alibi:
            alibi = build_alibi_tensor(max_seq_len, decoder_attention_heads)
            # (H, 1, MaxS) -> (N * H, 1, MaxS)
            alibi = alibi.repeat(batch_size, 1, 1)
            # (1, MaxS, MaxS) + (H * N, 1, MaxS) -> (H * N, MaxS, MaxS)
            _future_mask = _future_mask.unsqueeze(0) + alibi

        _future_mask = _future_mask.to(tensor)
        if use_alibi:
            return _future_mask[: batch_size * decoder_attention_heads, :cur_seq_len, :cur_seq_len,]

        return _future_mask[:cur_seq_len, :cur_seq_len]
        