from dataclasses import dataclass

import torch
from torch import nn

from pythia_kvpress.presses.scorer import ScorerPress


@dataclass
class KNormPress(ScorerPress):
    """
    KNorm-style KV pruning.

    keep_low_norm=True:
        keep keys with smaller L2 norm.

    keep_low_norm=False:
        keep keys with larger L2 norm.
    """

    keep_low_norm: bool = True

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> torch.Tensor:
        norms = keys.float().norm(dim=-1)

        if self.keep_low_norm:
            return -norms

        return norms