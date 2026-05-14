from contextlib import contextmanager


def iter_pythia_attention_modules(model):
    """
    Iterate over attention modules of Pythia / GPT-NeoX models.

    For AutoModelForCausalLM("EleutherAI/pythia-70m"), the module path is:
        model.gpt_neox.layers[i].attention
    """
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        for layer in model.gpt_neox.layers:
            yield layer.attention
        return

    raise ValueError(
        f"Unsupported model structure: {type(model)}. "
        "Expected a GPTNeoXForCausalLM-like model with model.gpt_neox.layers."
    )


@contextmanager
def register_attention_hooks(model, hook_fn):
    """
    Temporarily register a forward hook on every Pythia/GPT-NeoX attention layer.

    Usage:
        with register_attention_hooks(model, hook_fn):
            model(...)

    Hooks are removed automatically after leaving the context.
    """
    handles = []

    try:
        for attention_module in iter_pythia_attention_modules(model):
            handle = attention_module.register_forward_hook(
                hook_fn,
                with_kwargs=True,
            )
            handles.append(handle)

        yield

    finally:
        for handle in handles:
            handle.remove()