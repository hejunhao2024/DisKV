import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pythia_kvpress.presses.scorer import ScorerPress

try:
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class SnapKVPress(ScorerPress):
    """
    SnapKV for Pythia/GPT-NeoX.

    This is a simplified but faithful prefill SnapKV implementation:
      - use the last window_size context tokens as observation window
      - compute their attention to previous KV tokens
      - use averaged attention scores to select important KV pairs
      - always keep the observation window by assigning it max score

    Important:
      During PPL evaluation this should be used as a prefill press only.
      The target tokens are not included in the observation window, so there is
      no target leakage.
    """

    window_size: int = 32
    kernel_size: int = 7

    def __post_init__(self):
        super().__post_init__()
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.budget is not None and self.budget <= self.window_size:
            raise ValueError(
                f"budget ({self.budget}) should be larger than window_size ({self.window_size}) "
                "so the observation window can be fully kept."
            )

    @staticmethod
    def _apply_rope_to_query(query_states, position_embeddings):
        """
        Apply GPT-NeoX RoPE to query states.

        query_states:
            [batch, heads, window, head_dim]

        position_embeddings:
            tuple(cos, sin), usually each [batch, seq_len, rotary_dim or head_dim]
        """
        cos, sin = position_embeddings

        # Keep only the last window positions.
        window = query_states.shape[2]
        cos = cos[:, -window:, :]
        sin = sin[:, -window:, :]

        # Prefer the official GPT-NeoX implementation when available.
        if apply_rotary_pos_emb is not None:
            try:
                query_states, _ = apply_rotary_pos_emb(
                    query_states,
                    query_states,
                    cos,
                    sin,
                )
                return query_states
            except Exception:
                # Fall back to a local partial-rotary implementation below.
                pass

        # Fallback: apply RoPE to the prefix whose dim matches cos/sin.
        rotary_dim = cos.shape[-1]
        q_rot = query_states[..., :rotary_dim]
        q_pass = query_states[..., rotary_dim:]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1)

    @staticmethod
    def compute_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Recompute pre-RoPE query states from GPT-NeoX fused QKV projection.

        hidden_states:
            [batch, window, hidden_size]

        returns:
            query_states [batch, heads, window, head_dim]
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_size

        qkv = module.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, num_heads, 3 * head_dim).transpose(1, 2)
        query_states, _, _ = qkv.chunk(3, dim=-1)
        return query_states

    def compute_window_attention(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        window_size: int,
        position_embeddings,
    ) -> torch.Tensor:
        """
        Compute attention weights from the observation-window queries to all keys,
        then return weights for tokens before the observation window.

        returns:
            attn_prefix [batch, heads, window_size, k_len - window_size]
        """
        bsz, num_heads, k_len, head_dim = keys.shape
        window = min(window_size, hidden_states.shape[1], k_len - 1)

        query_states = self.compute_prerope_query_states(
            module,
            hidden_states[:, -window:, :],
        )

        query_states = self._apply_rope_to_query(
            query_states,
            position_embeddings,
        )

        attn_weights = torch.matmul(
            query_states.float(),
            keys.float().transpose(2, 3),
        ) / math.sqrt(head_dim)

        # Causal mask for observation-window queries.
        # Query absolute positions are [k_len-window, ..., k_len-1].
        key_pos = torch.arange(k_len, device=keys.device)
        query_pos = torch.arange(k_len - window, k_len, device=keys.device)
        causal_mask = key_pos.view(1, 1, 1, k_len) > query_pos.view(1, 1, window, 1)
        attn_weights = attn_weights.masked_fill(causal_mask, torch.finfo(attn_weights.dtype).min)

        attn_weights = torch.softmax(attn_weights, dim=-1).to(keys.dtype)

        # SnapKV scores only the tokens before the observation window.
        attn_prefix = attn_weights[..., : k_len - window]
        return attn_prefix

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> torch.Tensor:
        bsz, num_heads, k_len, _ = keys.shape

        if k_len <= 1:
            return torch.ones(keys.shape[:3], dtype=keys.dtype, device=keys.device)

        window = min(self.window_size, hidden_states.shape[1], k_len - 1)
        prefix_len = k_len - window

        if prefix_len <= 0:
            return torch.ones(keys.shape[:3], dtype=keys.dtype, device=keys.device)

        # Pythia attention usually does not return attention weights, so recompute.
        if attentions is not None:
            # attentions should be [B, H, q_len, k_len]
            attn_prefix = attentions[..., -window:, :prefix_len]
        else:
            position_embeddings = kwargs.get("position_embeddings", None)
            if position_embeddings is None:
                raise ValueError("SnapKVPress requires position_embeddings in attention kwargs.")

            attn_prefix = self.compute_window_attention(
                module=module,
                hidden_states=hidden_states,
                keys=keys,
                window_size=window,
                position_embeddings=position_embeddings,
            )

        # Average over observation queries.
        scores_prefix = attn_prefix.mean(dim=-2)  # [B, H, prefix_len]

        # Smooth scores over nearby token positions.
        if self.kernel_size > 1 and scores_prefix.shape[-1] > 1:
            scores_prefix = F.avg_pool1d(
                scores_prefix,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
            scores_prefix = scores_prefix[..., :prefix_len]

        # Always keep the observation window.
        max_score = scores_prefix.max().detach()
        scores = F.pad(
            scores_prefix,
            pad=(0, window),
            value=float(max_score),
        )

        assert scores.shape == keys.shape[:3], (
            f"SnapKV scores shape {scores.shape} does not match keys shape {keys.shape[:3]}"
        )

        return scores