from dataclasses import dataclass

import torch
from torch import nn

from pythia_kvpress.presses.base import BasePress


@dataclass
class ScorerPress(BasePress):
    """
    Base class for score-based KV cache compression methods.

    This follows the KVPress design:
      1. subclasses implement score(...)
      2. compress(...) keeps KV pairs with the highest scores

    Differences from KVPress:
      - supports both compression_ratio and explicit budget
      - sorts selected indices to preserve chronological cache order
      - uses keys.shape[-1] instead of module.head_dim for robustness

    Parameters
    ----------
    compression_ratio:
        Fraction of KV pairs to remove. Used only when budget is None.

    budget:
        Explicit number of KV pairs to keep. This is convenient for experiments
        such as K = 256 / 512 / 768 / 1024.

    keep_order:
        If True, selected token indices are sorted after top-k so the compressed
        cache remains in chronological order.
    """

    compression_ratio: float = 0.0
    budget: int | None = None
    keep_order: bool = True

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        if self.budget is not None:
            assert self.budget > 0, "budget must be positive"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Compute importance scores for each KV pair.

        Returns
        -------
        scores:
            Tensor with shape [batch_size, num_heads, seq_len].
            Higher scores indicate tokens that should be kept.
        """
        raise NotImplementedError

    def get_n_kept(self, k_len: int) -> int:
        if self.budget is not None:
            return min(self.budget, k_len)

        return max(1, int(k_len * (1 - self.compression_ratio)))

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        k_len = keys.shape[2]
        n_kept = self.get_n_kept(k_len)

        if n_kept >= k_len:
            return keys, values

        scores = self.score(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=attentions,
            kwargs=kwargs,
        )

        expected_shape = keys.shape[:3]
        if scores.shape != expected_shape:
            raise ValueError(
                f"scores shape must be {expected_shape}, got {scores.shape}"
            )

        # Keep KV pairs with highest scores.
        indices = scores.topk(n_kept, dim=-1).indices

        # Keep chronological token order in the compressed cache.
        if self.keep_order:
            indices = indices.sort(dim=-1).values

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])

        keys = keys.gather(dim=2, index=gather_indices).contiguous()
        values = values.gather(dim=2, index=gather_indices).contiguous()

        return keys, values