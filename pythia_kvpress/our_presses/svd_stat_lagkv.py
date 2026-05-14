from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from pythia_kvpress.presses.base import BasePress
from pythia_kvpress.our_presses.svd_value_kv import SVDValueKVPress
from pythia_kvpress.our_presses.stat_lagkv import StatLagKVPress


@dataclass
class SVDStatLagKVPress(BasePress):
    """
    SVD-ValueKV + StatLagKV.

    Order:
        1. SVD-ValueKV reconstructs / approximates Value cache in feature dimension.
        2. StatLagKV prunes KV tokens in sequence dimension.

    In the current implementation, SVD-ValueKV is still a dense-cache simulation:
        V -> low-rank coefficients -> V_hat

    Then StatLagKV performs real token pruning on:
        K, V_hat

    During decoding:
        - SVD-ValueKV still reconstructs the newly appended Value token.
        - StatLagKV is only applied during prefill, not during every decode step.
    """

    # ===== SVD-ValueKV parameters =====
    rank: int = 32
    basis_method: str = "attn"
    center: bool = False
    svd_device: str = "cuda"
    attn_obs_len: int = 128
    attn_weight_power: float = 2.0
    temperature: float = 4.0
    weight_cap_quantile: float = 0.0
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    layer_rank_map: Optional[dict[int, int]] = None

    # ===== StatLagKV parameters =====
    budget: int = 512
    n_sink: int = 4
    lag_size: int = 128
    cross_scoring: bool = False
    lag_alpha: float = 0.1

    def __post_init__(self):
        self.svd_press = SVDValueKVPress(
            mode="both",
            rank=self.rank,
            basis_method=self.basis_method,
            center=self.center,
            svd_device=self.svd_device,
            attn_obs_len=self.attn_obs_len,
            attn_weight_power=self.attn_weight_power,
            temperature=self.temperature,
            weight_cap_quantile=self.weight_cap_quantile,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            layer_rank_map=self.layer_rank_map,
        )

        self.stat_lagkv_press = StatLagKVPress(
            mode="prefill",
            budget=self.budget,
            n_sink=self.n_sink,
            lag_size=self.lag_size,
            cross_scoring=self.cross_scoring,
            alpha=self.lag_alpha,
        )

    def post_init_from_model(self, model: nn.Module):
        if hasattr(super(), "post_init_from_model"):
            super().post_init_from_model(model)

        if hasattr(self.svd_press, "post_init_from_model"):
            self.svd_press.post_init_from_model(model)

        if hasattr(self.stat_lagkv_press, "post_init_from_model"):
            self.stat_lagkv_press.post_init_from_model(model)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ):
        """
        keys / values:
            [batch, heads, seq_len, head_dim]

        returns:
            compressed keys / values
        """
        # query_len > 1 means prefill.
        # query_len == 1 means decode.
        if hidden_states is not None and hidden_states.ndim >= 2:
            query_len = hidden_states.shape[1]
        else:
            query_len = keys.shape[2]

        # Step 1: SVD-ValueKV.
        # Prefill: fit basis and reconstruct prefill V.
        # Decode: reconstruct newly appended V using fitted basis.
        keys_svd, values_svd = self.svd_press.compress(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=attentions,
            kwargs=kwargs,
        )

        # Step 2: StatLagKV token pruning only during prefill.
        if query_len > 1:
            keys_out, values_out = self.stat_lagkv_press.compress(
                module=module,
                hidden_states=hidden_states,
                keys=keys_svd,
                values=values_svd,
                attentions=attentions,
                kwargs=kwargs,
            )
            return keys_out, values_out

        return keys_svd, values_svd

    def theoretical_feature_kv_ratio(self):
        if hasattr(self.svd_press, "theoretical_feature_kv_ratio"):
            return self.svd_press.theoretical_feature_kv_ratio()
        return ""

    def get_fit_summary(self):
        if hasattr(self.svd_press, "get_fit_summary"):
            return self.svd_press.get_fit_summary()
        return {}
