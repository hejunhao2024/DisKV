import contextlib
import math

import torch
import torch.nn.functional as F
from transformers import DynamicCache


@torch.no_grad()
def prefill_context(model, context_ids, prefill_press=None):
    """
    Run context prefill and optionally compress KV cache with a prefill press.

    Returns:
        outputs: model output
        past: outputs.past_key_values
    """
    cache = DynamicCache(config=model.config)

    if prefill_press is None:
        outputs = model(
            input_ids=context_ids,
            past_key_values=cache,
            use_cache=True,
        )
    else:
        with prefill_press(model):
            outputs = model(
                input_ids=context_ids,
                past_key_values=cache,
                use_cache=True,
            )

    return outputs, outputs.past_key_values


@torch.no_grad()
def continuation_ppl(
    model,
    input_ids,
    context_len: int,
    target_len: int,
    prefill_press=None,
    decoding_press=None,
    position_mode: str = "absolute",
    count_first_target: bool = False,
):
    """
    Teacher-forced continuation perplexity.

    Protocol:
      1. Split input_ids into context and target.
      2. Prefill context.
      3. Optionally compress context KV with prefill_press.
      4. Evaluate target with teacher forcing.
      5. Return exp(mean negative log-likelihood).

    Args:
        position_mode:
            "absolute":
                target token position ids continue from original context_len.
                This is the default for non-rerotation methods.

            "compressed":
                target token position ids continue from compressed cache length.
                Use only when key rerotation / position remapping is implemented.

        count_first_target:
            If True, include loss from prefill last logits -> target[0].
            If False, start from target[0] -> target[1], so every measured token
            is predicted after the compressed cache is already in use.
    """
    device = next(model.parameters()).device
    model.eval()

    total_len = context_len + target_len
    input_ids = input_ids[:, :total_len].to(device)

    if input_ids.shape[1] < total_len:
        raise ValueError(
            f"Not enough tokens: got {input_ids.shape[1]}, need {total_len}"
        )

    context_ids = input_ids[:, :context_len]
    target_ids = input_ids[:, context_len:total_len]

    outputs, past = prefill_context(
        model=model,
        context_ids=context_ids,
        prefill_press=prefill_press,
    )

    if position_mode == "absolute":
        position_base = context_len
    elif position_mode == "compressed":
        position_base = past.get_seq_length(0)
    else:
        raise ValueError(f"Unknown position_mode: {position_mode}")

    nll_sum = 0.0
    n_tokens = 0

    # Optional: include the first target token predicted by prefill logits.
    # This is generation-faithful, but this first loss is not affected by the
    # compressed cache. For KV-compression comparison, False is cleaner.
    if count_first_target:
        first_gt = target_ids[:, 0]
        logits = outputs.logits[:, -1, :]
        loss = F.cross_entropy(logits.float(), first_gt, reduction="sum")
        nll_sum += loss.item()
        n_tokens += first_gt.numel()

    decode_context = (
        decoding_press(model)
        if decoding_press is not None
        else contextlib.nullcontext()
    )

    with decode_context:
        # Feed target[i], predict target[i + 1].
        for i in range(target_ids.shape[1] - 1):
            step_input = target_ids[:, i:i + 1]
            gt = target_ids[:, i + 1]

            position_ids = torch.tensor(
                [[position_base + i]],
                dtype=torch.long,
                device=device,
            )

            outputs = model(
                input_ids=step_input,
                past_key_values=past,
                position_ids=position_ids,
                use_cache=True,
            )

            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            loss = F.cross_entropy(logits.float(), gt, reduction="sum")
            nll_sum += loss.item()
            n_tokens += gt.numel()

    if n_tokens == 0:
        return float("nan")

    return math.exp(nll_sum / n_tokens)