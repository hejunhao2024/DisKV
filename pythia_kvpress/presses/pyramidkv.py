from dataclasses import dataclass

import torch
from torch import nn

from pythia_kvpress.presses.snapkv import SnapKVPress


@dataclass
class PyramidKVPress(SnapKVPress):
    """
    PyramidKV for Pythia/GPT-NeoX.

    This reuses SnapKV scoring, but allocates different KV budgets to
    different layers.

    Lower layers keep more KV tokens.
    Higher layers keep fewer KV tokens.

    In this project, `budget` means the average number of KV tokens to keep
    across layers. For example, budget=512 means the average layer budget is
    approximately 512, but lower layers may keep more and higher layers may
    keep less.

    Parameters
    ----------
    budget:
        Average target KV budget across layers.

    window_size:
        SnapKV observation window. This window is always kept.

    kernel_size:
        Pooling kernel for smoothing SnapKV attention scores.

    beta:
        Controls pyramid steepness.
        Larger beta => smaller top-layer budget and larger lower-layer budget.
    """

    beta: int = 20

    def __post_init__(self):
        super().__post_init__()
        if self.beta < 1:
            raise ValueError("beta must be >= 1.")

    def _get_average_budget(self, k_len: int) -> int:
        if self.budget is not None:
            return min(self.budget, k_len)

        # Fallback to compression_ratio if explicit budget is not provided.
        return max(1, int(k_len * (1.0 - self.compression_ratio)))

    def get_layer_budget(self, module: nn.Module, k_len: int) -> int:
        """
        Compute a layer-wise pyramid budget.

        We treat `budget` as the average total budget per layer.

        Let:
          avg_total = budget
          window = observation window, always kept
          avg_history = avg_total - window

        Then assign the history part with a linear pyramid:
          lower layer history budget = max_history
          upper layer history budget = min_history

        The average history budget is approximately avg_history.
        """
        num_layers = module.config.num_hidden_layers
        layer_idx = module.layer_idx

        avg_total = self._get_average_budget(k_len)
        window = min(self.window_size, k_len - 1)

        if avg_total >= k_len:
            return k_len

        # Need to keep the observation window plus at least one historical KV.
        avg_total = max(avg_total, window + 1)

        prefix_len = k_len - window
        avg_history = max(1, avg_total - window)

        # Pyramid shape, adapted from PyramidKV / KVCache-Factory idea.
        min_history = max(1.0, avg_history / self.beta)
        max_history = 2.0 * avg_history - min_history

        # Do not allocate more historical tokens than exist.
        if max_history > prefix_len:
            max_history = float(prefix_len)
            min_history = max(1.0, 2.0 * avg_history - max_history)

        # If the pyramid is invalid, fall back to a flat SnapKV-style budget.
        if max_history < min_history or min_history < 1:
            return avg_total

        if num_layers <= 1:
            history_budget = avg_history
        else:
            step = (max_history - min_history) / (num_layers - 1)
            history_budget = max_history - layer_idx * step

        n_kept = int(round(window + history_budget))
        n_kept = max(window + 1, n_kept)
        n_kept = min(k_len, n_kept)

        return n_kept

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
        n_kept = self.get_layer_budget(module, k_len)

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

        indices = scores.topk(n_kept, dim=-1).indices

        # Keep chronological token order.
        if self.keep_order:
            indices = indices.sort(dim=-1).values

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])

        keys = keys.gather(dim=2, index=gather_indices).contiguous()
        values = values.gather(dim=2, index=gather_indices).contiguous()

        return keys, values