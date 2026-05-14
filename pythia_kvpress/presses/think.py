from dataclasses import dataclass

import torch
from torch import nn

from pythia_kvpress.presses.base import BasePress
from pythia_kvpress.presses.snapkv import SnapKVPress


@dataclass
class ThinKPress(BasePress):
    """
    ThinK-inspired channel-wise key compression for Pythia/GPT-NeoX.

    This implementation follows the KVPress-style ThinK simplification:
      - it does NOT shorten the key tensor shape
      - it sets low-importance key channels to zero
      - therefore it does not reduce real KV memory in this dense-cache version

    It is useful as a channel-pruning / channel-ablation baseline.

    key_channel_compression_ratio:
        Fraction of key channels to zero out.

    window_size:
        Number of most recent tokens used to estimate query channel importance.
    """

    key_channel_compression_ratio: float = 0.5
    window_size: int = 32

    def __post_init__(self):
        if not (0.0 <= self.key_channel_compression_ratio < 1.0):
            raise ValueError("key_channel_compression_ratio must be in [0, 1).")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")

    def compute_window_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        """
        Recompute the last window query states and apply RoPE.

        returns:
            query_states [batch, heads, window, head_dim]
        """
        window = min(self.window_size, hidden_states.shape[1])

        query_states = SnapKVPress.compute_prerope_query_states(
            module=module,
            hidden_states=hidden_states[:, -window:, :],
        )

        query_states = SnapKVPress._apply_rope_to_query(
            query_states=query_states,
            position_embeddings=position_embeddings,
        )

        return query_states

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.key_channel_compression_ratio == 0:
            return keys, values

        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is None:
            raise ValueError("ThinKPress requires position_embeddings in attention kwargs.")

        bsz, num_heads, seq_len, head_dim = keys.shape

        n_pruned = int(head_dim * self.key_channel_compression_ratio)
        if n_pruned <= 0:
            return keys, values

        # Query channel importance from recent observation window.
        queries = self.compute_window_queries(
            module=module,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )

        # queries_norm: [B, H, D]
        queries_norm = torch.pow(queries.float(), 2).mean(dim=2)

        # keys_norm: [B, H, D]
        keys_norm = torch.pow(keys.float(), 2).mean(dim=2)

        # Higher score means more important channel.
        key_channel_scores = queries_norm * keys_norm

        # Prune dimensions with lowest scores.
        prune_indices = key_channel_scores.topk(
            n_pruned,
            dim=-1,
            largest=False,
        ).indices  # [B, H, n_pruned]

        # Expand to all sequence positions.
        prune_indices = prune_indices.unsqueeze(2).expand(
            -1,
            -1,
            seq_len,
            -1,
        )

        # Keep tensor shape unchanged, but zero out selected key channels.
        new_keys = keys.clone()
        new_keys.scatter_(dim=-1, index=prune_indices, value=0)

        return new_keys.contiguous(), values

    @property
    def compression_ratio(self):
        # Only keys are masked, values are unchanged.
        # KVPress reports this as half of key-channel ratio.
        return self.key_channel_compression_ratio / 2

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(
            f"compression_ratio cannot be set for {type(self).__name__}; "
            "use key_channel_compression_ratio instead."
        )
