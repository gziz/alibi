# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Protocol, final
import math
import torch
from torch import Tensor

_neg_inf = float("-inf")


class AttentionMaskGenerator(Protocol):
    """Generates an attention mask."""

    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input for which to generate the mask. *Shape:* :math:`(N,S,M)`,
            or :math:`(S,M)` when unbatched, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`M` is the model size.

        :returns:
            An implementation-defined attention mask specific to the generator.
            *Shape:* :math:`(S,S)`, where :math:`S` is the sequence length.
        """


@final
class CausalAttentionMaskGenerator:
    """Generates a causal attention mask for self attention.

    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.
    """

    _cached_attn_mask: Optional[Tensor]

    def __init__(self) -> None:
        self._cached_attn_mask = None

    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input for which to generate the mask. *Shape:* :math:`(N,S,M)`,
            or :math:`(S,M)` when unbatched, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`M` is the model size.

        :returns:
            An attention mask whose upper triangular part above the main
            diagonal is filled with negative infinity (i.e. ``float("-inf")``),
            while its rest is filled with zero. *Shape:* :math:`(S,S)`, where
            :math:`S` is the sequence length.

        Usage:

        >>> import torch
        >>>
        >>> from fairseq2.nn.transformer import CausalAttentionMaskGenerator
        >>>
        >>> g = CausalAttentionMaskGenerator()
        >>> g(torch.empty(4, 10, 3))
        tensor([[0., -inf, -inf, -inf],
                [0.,   0., -inf, -inf],
                [0.,   0.,   0., -inf],
                [0.,   0.,   0.,   0.]])
        """
        mask = self._cached_attn_mask

        if x.dim() == 2:
            seq_len = x.size(0)
        else:
            seq_len = x.size(1)

        if mask is None or mask.device != x.device or mask.size(0) < seq_len:
            mask = x.new_full([seq_len, seq_len], _neg_inf)

            mask.triu_(diagonal=1)

            self._cached_attn_mask = mask

        return mask[:seq_len, :seq_len]


@final
class ALiBiAttentionMaskGenerator:
    """Generates a mask for self attention as described in
    :cite:t:`DBLP:journals/corr/abs-2108-12409`.
    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.
    """

    _cached_attn_mask: Optional[Tensor]

    def __init__(self, num_heads: int) -> None:
        self._cached_attn_mask = None
        self.num_heads = num_heads

    def get_slopes(self, num_heads: int) -> Tensor:
        """Compute the slopes"""

        def get_slopes_power_of_2(num_heads: int, step: int = 1) -> Tensor:
            start = 2 ** (-8 / num_heads)
            return torch.pow(start, torch.arange(1, 1 + num_heads, step))

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_pow_2 = 2 ** math.floor(math.log2(num_heads))
            base_slopes = get_slopes_power_of_2(closest_pow_2)
            num_slopes_left = num_heads - closest_pow_2
            extra_slopes = get_slopes_power_of_2(2 * closest_pow_2, step=2)

            return torch.cat([base_slopes, extra_slopes[:num_slopes_left]])

    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input for which to generate the mask. *Shape:* :math:`(N,S,M)`,
            or :math:`(S,M)` when unbatched, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`M` is the model size.
        :returns:
            An ALiBi mask. *Shape:* :math:`(H, S, S)`, where
            :math:`S` is the sequence length and :math:`H`is the number of heads.
        """

        mask = self._cached_attn_mask
        seq_len = x.size(-2)

        if mask is None or mask.device != x.device or mask.size(-2) < seq_len:
            slopes = self.get_slopes(self.num_heads)

            arange_tensor = torch.arange(seq_len)[None, None, :]
            arange_tensor = arange_tensor.expand((self.num_heads, -1, -1))

            alibi_biases = arange_tensor * slopes[:, None, None]
            mask = alibi_biases + CausalAttentionMaskGenerator()(x)

            self._cached_attn_mask = mask

        return mask[:, :seq_len, :seq_len]
