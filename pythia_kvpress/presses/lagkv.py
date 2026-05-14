from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pythia_kvpress.presses.scorer import ScorerPress


@dataclass
class LagKVPress(ScorerPress):
    """
    LagKV: Lag-relative information-based KV cache compression.

    This is the original LagKV-style scoring method adapted to this project.

    Core idea:
      1. Keep the first n_sink tokens as attention sinks.
      2. Divide the remaining sequence into lag_size blocks.
      3. Use the following block as a reference to score the previous block.
      4. Keep KV tokens with the highest scores through ScorerPress.

    Example:
        blocks after sink tokens:
            C0, C1, C2, C3, ...

        scoring:
            C1 scores C0
            C2 scores C1
            C3 scores C2

        The last full block and the non-full tail are protected because they
        do not have a following reference block.

    Parameters
    ----------
    compression_ratio:
        Inherited from ScorerPress. Fraction of KV tokens to remove when budget
        is not specified.

    budget:
        Inherited from ScorerPress. Explicit number of KV tokens to keep.

    keep_order:
        Inherited from ScorerPress. Sort selected token indices to preserve
        chronological cache order.

    n_sink:
        Number of initial sink tokens to always keep.

    lag_size:
        Partition size for lag-relative scoring.

    cross_scoring:
        If False, scores are rank-normalized inside each lag block, matching the
        original LagKV behavior.
    """

    n_sink: int = 4
    lag_size: int = 128
    cross_scoring: bool = False
    eps: float = 1e-6

    def __post_init__(self):
        super().__post_init__()

        if self.n_sink < 0:
            raise ValueError("n_sink must be non-negative.")

        if self.lag_size <= 0:
            raise ValueError("lag_size must be positive.")

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

        if q_len < self.n_sink + 2 * self.lag_size:
            # Too short to form two complete lag blocks.
            # Return a stable fallback score: protect sink tokens and prefer
            # more recent tokens in the sliding part.
            score = torch.ones(
                (bsz, num_heads, q_len),
                dtype=keys.dtype,
                device=keys.device,
            )

            if q_len > self.n_sink:
                sliding_len = q_len - self.n_sink
                score[:, :, self.n_sink:] = (
                    torch.arange(sliding_len, device=keys.device, dtype=torch.float32)
                    / max(sliding_len, 1)
                ).to(keys.dtype)

            return score

        # Use only complete lag blocks after sink tokens.
        end_idx = self.n_sink + ((q_len - self.n_sink) // self.lag_size) * self.lag_size

        # The final complete block has no next reference block, so it is kept
        # together with any incomplete tail.
        tail_len = self.lag_size + q_len - end_idx

        key_blocks = keys[:, :, self.n_sink:end_idx, :].contiguous().view(
            bsz,
            num_heads,
            -1,
            self.lag_size,
            head_dim,
        )

        value_blocks = values[:, :, self.n_sink:end_idx, :].contiguous().view(
            bsz,
            num_heads,
            -1,
            self.lag_size,
            head_dim,
        )

        key_score = self._get_states_score(key_blocks)
        value_score = self._get_states_score(value_blocks)

        # Combine key and value variation scores.
        score = (key_score + value_score) / 2.0

        if not self.cross_scoring:
            # Rank-normalize scores inside each lag block.
            # Higher rank means more important token.
            score = score.argsort(dim=-1).argsort(dim=-1)
            score = score.to(torch.float32) / float(self.lag_size)
            score = score.to(keys.dtype)

        # Protect sink tokens.
        sink_shape = (bsz, num_heads, self.n_sink)
        sink_score = torch.ones(
            sink_shape,
            dtype=score.dtype,
            device=score.device,
        )

        # Protect the last full lag block and the incomplete tail.
        tail_shape = (bsz, num_heads, tail_len)
        tail_score = torch.ones(
            tail_shape,
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
                f"LagKV score shape mismatch: got {tuple(score.shape)}, "
                f"expected {(bsz, num_heads, q_len)}"
            )

        return score

    def _get_states_score(self, target_v: torch.Tensor) -> torch.Tensor:
        """
        Evaluate lag-relative scores for each token.

        target_v:
            [batch, heads, num_blocks, lag_size, head_dim]

        returns:
            [batch, heads, num_blocks - 1, lag_size]
        """
        ref = target_v[:, :, 1:, :, :]
        v = target_v[:, :, :-1, :, :]

        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(
            -1,
            -1,
            -1,
            self.lag_size,
            -1,
        )

        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(
            -1,
            -1,
            -1,
            self.lag_size,
            -1,
        )

        denom = (max_r - min_r).clamp_min(self.eps)

        # Original LagKV min-max lag-relative score.
        score = ((v - min_r) / denom).std(dim=-1).softmax(dim=-1)

        return score
