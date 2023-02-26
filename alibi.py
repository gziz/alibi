import math

import torch
from torch import Tensor

class ALiBiAttentionMaskGenerator:
    """Generates a mask for self attention as described in
    :cite:t:`DBLP:journals/corr/abs-2108-12409`.
    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.
    """

    def get_slopes(self, n_heads: int) -> Tensor:

        n = 2 ** math.floor(math.log2(n_heads))
        # m0: start and ratio of the geometric series
        m0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m0, torch.arange(1, 1 + n))

        # if the n_heads is not a power of 2
        if n < n_heads:
            m_hat_0 = 2.0 ** (-8.0 / 2*n)

            # avoid adding repeated slopes (m) with a step of 2
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            m = torch.cat([m, m_hat])

        return m
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x:
            The input for which to generate the mask. *Shape:* :math:`(N,S,M)`,
            or :math:`(S,M)` when unbatched, where :math:`N` is the batch size,
            :math:`S` is the sequence length, and :math:`M` is the model size.
        :returns:
            An ALiBi mask. *Shape:* :math:`(S,S,M)`, where
            :math:`S` is the sequence length and :math:`M`is the model size.
            
        """
        seq_len, n_heads = x.size(-2), x.size(-1)
        m = self.get_slopes(n_heads)

        distance_vec = torch.arange(seq_len)[None, :]
        distance = distance_vec.expand((seq_len, seq_len))

        return distance[:, :, None] * m[None, None, :]