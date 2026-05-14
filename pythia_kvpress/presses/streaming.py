from dataclasses import dataclass

import torch

from pythia_kvpress.presses.base import BasePress


@dataclass
class StreamingLLMPress(BasePress):
    """
    Sink + recent KV cache compression.

    mode="prefill":
        After context prefill, keep first n_sink tokens + most recent tokens.

    mode="decode":
        During token-by-token decoding, keep the cache within budget by
        retaining first n_sink tokens + most recent tokens.
    """

    budget: int = 512
    n_sink: int = 4

    def __post_init__(self):
        if self.budget <= 0:
            raise ValueError("budget must be positive.")
        if self.n_sink < 0:
            raise ValueError("n_sink must be non-negative.")
        if self.n_sink >= self.budget:
            raise ValueError("n_sink must be smaller than budget.")

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        seq_len = keys.shape[2]

        if seq_len <= self.budget:
            return keys, values

        recent = self.budget - self.n_sink

        if self.n_sink == 0:
            return (
                keys[:, :, -recent:, :].contiguous(),
                values[:, :, -recent:, :].contiguous(),
            )

        new_keys = torch.cat(
            [
                keys[:, :, : self.n_sink, :],
                keys[:, :, -recent:, :],
            ],
            dim=2,
        )

        new_values = torch.cat(
            [
                values[:, :, : self.n_sink, :],
                values[:, :, -recent:, :],
            ],
            dim=2,
        )

        return new_keys.contiguous(), new_values.contiguous()