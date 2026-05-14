from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

from pythia_kvpress.cache_utils import (
    get_cache_from_kwargs,
    get_layer_kv,
    set_layer_kv,
)
from pythia_kvpress.hooks import register_attention_hooks


PressMode = Literal["prefill", "decode", "both"]


@dataclass
class BasePress:
    """
    Base class for Pythia/GPT-NeoX KV cache compression.

    It registers forward hooks on GPT-NeoX attention layers.
    Each hook reads the current layer's KV cache, calls compress(...),
    and writes compressed KV tensors back into the cache.
    """

    mode: PressMode = "prefill"

    def post_init_from_model(self, model):
        pass

    def _get_hidden_states(self, inputs, kwargs):
        """
        In GPT-NeoX/Pythia, hidden_states is usually passed as positional input,
        not as kwargs. So we support both.
        """
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is not None:
            return hidden_states

        if inputs is not None and len(inputs) > 0:
            return inputs[0]

        return None

    def should_compress(self, module, hidden_states, cache, kwargs) -> bool:
        q_len = hidden_states.shape[1]

        if self.mode == "prefill":
            return q_len > 1

        if self.mode == "decode":
            return q_len == 1

        if self.mode == "both":
            return True

        raise ValueError(f"Unknown press mode: {self.mode}")

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        raise NotImplementedError

    def forward_hook(self, module, inputs, kwargs, output):
        hidden_states = self._get_hidden_states(inputs, kwargs)
        if hidden_states is None:
            return output

        cache = get_cache_from_kwargs(kwargs)
        if cache is None:
            return output

        if not self.should_compress(module, hidden_states, cache, kwargs):
            return output

        layer_idx = module.layer_idx
        keys, values = get_layer_kv(cache, layer_idx)

        attentions = None
        if isinstance(output, (tuple, list)) and len(output) > 1:
            attentions = output[1]

        new_keys, new_values = self.compress(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=attentions,
            kwargs=kwargs,
        )

        set_layer_kv(cache, layer_idx, new_keys, new_values)
        return output

    def __call__(self, model):
        self.post_init_from_model(model)
        return register_attention_hooks(model, self.forward_hook)


def maybe_press(model, press):
    if press is None:
        return nullcontext()
    return press(model)