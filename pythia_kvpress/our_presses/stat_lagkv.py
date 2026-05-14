from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pythia_kvpress.presses.scorer import ScorerPress


@dataclass
class StatLagKVPress(ScorerPress):
    """
    StatLagKV: statistical lag-relative KV cache compression.

    This is the teammate-modified LagKV method.

    Compared with original LagKV, the only conceptual change is the scoring
    function. For each lag block C_i, the next block C_{i+1} is used as the
    reference. Each token v in C_i is scored by:

        mean_R = mean(C_{i+1})
        var_R  = var(C_{i+1})

        z_score(v, R) = mean_j ((v_j - mean_R_j)^2 / var_R_j)

        cosine_novelty(v, R) = 1 - cos(v, mean_R)

        score = alpha * z_score + (1 - alpha) * cosine_novelty

    Higher score means the token is more different from the following reference
    block, so it is considered more worth keeping.

    Parameters
    ----------
    n_sink:
        Number of initial sink tokens to always keep.

    lag_size:
        Block size for lag-relative scoring.

    cross_scoring:
        If False, rank-normalize scores within each lag block, matching the
        LagKV-style local ranking behavior.

    alpha:
        Mixture coefficient.
        alpha=0.1 means mostly cosine novelty plus a small z-score term.

    eps:
        Numerical stability.
    """

    n_sink: int = 4
    lag_size: int = 128
    cross_scoring: bool = False
    alpha: float = 0.1
    eps: float = 1e-6

    def __post_init__(self):
        super().__post_init__()

        if self.n_sink < 0:
            raise ValueError("n_sink must be non-negative.")

        if self.lag_size <= 0:
            raise ValueError("lag_size must be positive.")

        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        if self.eps <= 0:
            raise ValueError("eps must be positive.")

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
        Compute per-token scores.

        keys / values:
            [batch, heads, seq_len, head_dim]

        returns:
            scores [batch, heads, seq_len]
        """
        bsz, num_heads, q_len, head_dim = keys.shape

        # Too short for two meaningful lag blocks.
        if q_len <= self.n_sink + 2:
            return torch.ones(
                (bsz, num_heads, q_len),
                dtype=keys.dtype,
                device=keys.device,
            )

        effective_lag = self.lag_size

        # Shrink lag window for short context so that we can still form
        # at least two blocks: one current block and one reference block.
        if q_len < self.n_sink + 2 * effective_lag:
            effective_lag = (q_len - self.n_sink) // 2

        if effective_lag < 2:
            return torch.ones(
                (bsz, num_heads, q_len),
                dtype=keys.dtype,
                device=keys.device,
            )

        # Use complete lag blocks after sink tokens.
        end_idx = self.n_sink + ((q_len - self.n_sink) // effective_lag) * effective_lag

        # The last complete block has no following reference block, so protect it.
        # Also protect any incomplete tail.
        tail_len = effective_lag + q_len - end_idx

        key_blocks = keys[:, :, self.n_sink:end_idx, :].contiguous().view(
            bsz,
            num_heads,
            -1,
            effective_lag,
            head_dim,
        )

        value_blocks = values[:, :, self.n_sink:end_idx, :].contiguous().view(
            bsz,
            num_heads,
            -1,
            effective_lag,
            head_dim,
        )

        key_score = self._get_states_score(key_blocks)
        value_score = self._get_states_score(value_blocks)

        score = (key_score + value_score) / 2.0

        if not self.cross_scoring:
            # Local rank-normalization inside each lag block.
            score = score.argsort(dim=-1).argsort(dim=-1)
            score = score.to(torch.float32) / float(effective_lag)
            score = score.to(keys.dtype)

        sink_score = torch.ones(
            (bsz, num_heads, self.n_sink),
            dtype=score.dtype,
            device=score.device,
        )

        tail_score = torch.ones(
            (bsz, num_heads, tail_len),
            dtype=score.dtype,
            device=score.device,
        )

        score = torch.cat(
            (
                sink_score,
                score.reshape(bsz, num_heads, -1),
                tail_score,
            ),
            dim=-1,
        )

        if score.shape != (bsz, num_heads, q_len):
            raise RuntimeError(
                f"StatLagKV score shape mismatch: got {tuple(score.shape)}, "
                f"expected {(bsz, num_heads, q_len)}"
            )

        return score

    def _get_states_score(self, target_v: torch.Tensor) -> torch.Tensor:
        """
        target_v:
            [batch, heads, num_blocks, lag_size, head_dim]

        returns:
            [batch, heads, num_blocks - 1, lag_size]
        """
        ref = target_v[:, :, 1:, :, :]
        v = target_v[:, :, :-1, :, :]

        mean_r = ref.mean(dim=-2, keepdim=True)
        var_r = ref.var(dim=-2, keepdim=True, unbiased=False).clamp_min(self.eps)

        # Statistical distance to the following block.
        z_score = ((v - mean_r) ** 2 / var_r).mean(dim=-1)

        # Direction novelty relative to the following block mean.
        v_norm = F.normalize(v.float(), dim=-1, eps=self.eps)
        r_norm = F.normalize(mean_r.float(), dim=-1, eps=self.eps)

        cosine_novelty = 1.0 - (v_norm * r_norm).sum(dim=-1)
        cosine_novelty = cosine_novelty.to(dtype=z_score.dtype)

        score = self.alpha * z_score + (1.0 - self.alpha) * cosine_novelty

        return score
