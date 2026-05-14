from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from pythia_kvpress.presses.base import BasePress
from pythia_kvpress.presses.snapkv import SnapKVPress


def parse_layer_rank_map(spec: str | None) -> dict[int, int] | None:
    """
    Parse per-layer rank config.

    Examples
    --------
    "0:48,1:48,2-5:32"
        layer 0 rank 48
        layer 1 rank 48
        layers 2,3,4,5 rank 32

    Layer indices are zero-based.
    """
    if spec is None or str(spec).strip() == "":
        return None

    rank_map: dict[int, int] = {}

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" not in item:
            raise ValueError(
                f"Invalid layer rank item: {item}. "
                "Expected format like '0:48' or '2-5:32'."
            )

        layer_part, rank_part = item.split(":", 1)
        rank = int(rank_part)

        if "-" in layer_part:
            start_s, end_s = layer_part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {layer_part}")

            for layer_idx in range(start, end + 1):
                rank_map[layer_idx] = rank
        else:
            rank_map[int(layer_part)] = rank

    return rank_map


@dataclass
class SVDValueKVPress(BasePress):
    """
    SVD-ValueKV: SVD-based Value Cache Compression.

    This is a dense-cache simulation implementation.

    Core idea
    ---------
    For each selected layer/head, keep K unchanged and approximate V by a
    low-rank SVD subspace:

        z = (v - mu) P^T
        v_hat = z P + mu

    In this file, v_hat is written back to the HuggingFace KV cache with the
    original head dimension. Therefore, this version evaluates quality loss
    from low-rank Value approximation, but it does not physically reduce
    HuggingFace cache memory.

    To fully realize memory reduction, one would need to store Z instead of
    V_hat and modify attention forward to compute:

        A V ~= (A Z) P + mu

    Usage
    -----
    For complete prefill + decode behavior, use the same object as both
    prefill_press and decoding_press:

        press = SVDValueKVPress(mode="both", rank=32, basis_method="attn")
        return press, press

    Parameters
    ----------
    rank:
        Default SVD rank for every selected layer.

    basis_method:
        "naive":
            vanilla SVD on V.

        "attn":
            temperature-smoothed attention-weighted SVD.
            Token weights are estimated from the last attn_obs_len prefill
            queries.

    center:
        Whether to subtract weighted mean before SVD.

    layer_rank_map:
        Optional per-layer rank map, e.g. {0: 48, 1: 48, 2: 32}.
        If a layer is not in the map, rank is used.

    layer_start, layer_end:
        Optional selected layer range [layer_start, layer_end).
        If both are None, compress all layers.
    """

    rank: int = 32
    basis_method: str = "attn"  # "attn" or "naive"
    center: bool = False
    svd_device: str = "cuda"

    # Attention-weighted SVD parameters.
    attn_obs_len: int = 128
    attn_weight_power: float = 2.0
    temperature: float = 4.0
    weight_cap_quantile: float = 0.0

    # Optional layer selection.
    layer_start: int | None = None
    layer_end: int | None = None

    # Optional per-layer rank map.
    layer_rank_map: dict[int, int] | None = None

    # Internal states fitted during prefill.
    bases: dict[int, dict] = field(default_factory=dict, init=False)
    fit_stats: list[dict] = field(default_factory=list, init=False)
    num_layers: int | None = field(default=None, init=False)
    head_dim: int | None = field(default=None, init=False)
    selected_layers: set[int] | None = field(default=None, init=False)

    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError("rank must be positive.")

        if self.basis_method not in {"attn", "naive"}:
            raise ValueError("basis_method must be 'attn' or 'naive'.")

        if self.svd_device not in {"cuda", "cpu"}:
            raise ValueError("svd_device must be 'cuda' or 'cpu'.")

        if self.attn_obs_len <= 0:
            raise ValueError("attn_obs_len must be positive.")

        if self.attn_weight_power <= 0:
            raise ValueError("attn_weight_power must be positive.")

        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

        if self.weight_cap_quantile < 0 or self.weight_cap_quantile >= 1:
            if self.weight_cap_quantile != 0:
                raise ValueError("weight_cap_quantile must be in [0, 1).")

    def post_init_from_model(self, model):
        """
        Called by BasePress.__call__(model).

        Do not clear bases here because the same object should keep prefill-fitted
        bases for decode compression.
        """
        if self.num_layers is None:
            self.num_layers = int(model.config.num_hidden_layers)

        if self.selected_layers is None:
            start = 0 if self.layer_start is None else int(self.layer_start)
            end = self.num_layers if self.layer_end is None else int(self.layer_end)

            start = max(0, start)
            end = min(self.num_layers, end)

            if end < start:
                raise ValueError(f"Invalid layer range: [{start}, {end})")

            self.selected_layers = set(range(start, end))

    def _should_compress_layer(self, layer_idx: int) -> bool:
        return self.selected_layers is None or layer_idx in self.selected_layers

    def _get_layer_rank(self, layer_idx: int, head_dim: int) -> int:
        if self.layer_rank_map is not None and layer_idx in self.layer_rank_map:
            r = int(self.layer_rank_map[layer_idx])
        else:
            r = int(self.rank)

        return max(1, min(r, head_dim))

    def _get_svd_work_device(self, original_device: torch.device) -> torch.device:
        if self.svd_device == "cpu":
            return torch.device("cpu")

        # If values are already on CUDA, use that device. Otherwise fall back to CPU.
        if original_device.type == "cuda":
            return original_device

        return torch.device("cpu")

    def _compute_attention_token_weights(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Compute token weights for attention-weighted SVD.

        Returns
        -------
        weights:
            [batch, heads, seq_len], mean-normalized per batch/head.
        """
        bsz, num_heads, k_len, head_dim = keys.shape

        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is None:
            raise ValueError(
                "SVDValueKVPress with basis_method='attn' requires "
                "position_embeddings in attention kwargs. This follows the same "
                "assumption as SnapKVPress in this codebase."
            )

        window = min(self.attn_obs_len, hidden_states.shape[1], k_len)

        query_states = SnapKVPress.compute_prerope_query_states(
            module=module,
            hidden_states=hidden_states[:, -window:, :],
        )

        query_states = SnapKVPress._apply_rope_to_query(
            query_states=query_states,
            position_embeddings=position_embeddings,
        )

        # [B, H, W, D] x [B, H, D, N] -> [B, H, W, N]
        attn_logits = torch.matmul(
            query_states.float(),
            keys.float().transpose(2, 3),
        ) / math.sqrt(head_dim)

        # Causal mask for last-W prefill queries.
        key_pos = torch.arange(k_len, device=keys.device)
        query_pos = torch.arange(k_len - window, k_len, device=keys.device)

        causal_mask = (
            key_pos.view(1, 1, 1, k_len)
            > query_pos.view(1, 1, window, 1)
        )

        attn_logits = attn_logits.masked_fill(
            causal_mask,
            torch.finfo(attn_logits.dtype).min,
        )

        attn = torch.softmax(attn_logits, dim=-1)

        if self.attn_weight_power == 1.0:
            score = attn.sum(dim=2)  # [B, H, N]
        else:
            score = attn.clamp_min(0.0).pow(self.attn_weight_power).sum(dim=2)

        score = score.clamp_min(1e-12)

        # Temperature smoothing:
        # T < 1: sharper
        # T > 1: smoother
        weights = score.pow(1.0 / self.temperature)

        if self.weight_cap_quantile is not None and self.weight_cap_quantile > 0:
            caps = torch.quantile(
                weights,
                q=self.weight_cap_quantile,
                dim=-1,
                keepdim=True,
            )
            weights = torch.minimum(weights, caps)

        weights = weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-12)

        return weights.detach().float()

    @torch.no_grad()
    def _fit_svd_basis_for_values(
        self,
        values: torch.Tensor,
        rank: int,
        token_weights: torch.Tensor | None,
    ) -> dict:
        """
        Fit per-head SVD basis for one layer's Value cache.

        values:
            [batch, heads, seq_len, head_dim]

        token_weights:
            None for naive SVD, or [batch, heads, seq_len] for weighted SVD.

        Returns
        -------
        basis:
            {
                "mean": [heads, head_dim],
                "components": [heads, rank, head_dim],
                ...
            }
        """
        original_device = values.device
        bsz, num_heads, seq_len, head_dim = values.shape
        effective_rank = min(rank, head_dim)

        work_device = self._get_svd_work_device(original_device)

        means = []
        components = []
        energy_at_rank = []

        for h in range(num_heads):
            # [B, N, D] -> [B*N, D]
            x = values[:, h, :, :].detach().float().reshape(-1, head_dim)
            x = x.to(work_device)

            if token_weights is None:
                w = torch.ones(
                    x.shape[0],
                    dtype=torch.float32,
                    device=work_device,
                )
            else:
                # [B, H, N] -> [B*N]
                w = token_weights[:, h, :].detach().float().reshape(-1)
                w = w.to(work_device).clamp_min(1e-12)
                w = w / w.mean().clamp_min(1e-12)

            if self.center:
                mean = (x * w[:, None]).sum(dim=0) / w.sum().clamp_min(1e-12)
                x_work = x - mean
            else:
                mean = torch.zeros(
                    head_dim,
                    dtype=torch.float32,
                    device=work_device,
                )
                x_work = x

            # Weighted PCA objective:
            #   min_P sum_i w_i ||v_i - v_hat_i||^2
            #
            # Implementation:
            #   SVD on sqrt(w_i) * (v_i - mean)
            x_weighted = x_work * torch.sqrt(w[:, None])

            _, singular_values, vh = torch.linalg.svd(
                x_weighted,
                full_matrices=False,
            )

            comp = vh[:effective_rank, :].contiguous()

            total_energy = torch.sum(singular_values ** 2).clamp_min(1e-12)
            kept_energy = torch.sum(singular_values[:effective_rank] ** 2)
            energy = float((kept_energy / total_energy).detach().cpu())

            means.append(mean.detach().to(original_device))
            components.append(comp.detach().to(original_device))
            energy_at_rank.append(energy)

        mean = torch.stack(means, dim=0)             # [H, D]
        component = torch.stack(components, dim=0)   # [H, R, D]

        return {
            "mean": mean,
            "components": component,
            "rank": effective_rank,
            "head_dim": head_dim,
            "seq_len_fit": seq_len,
            "avg_energy_at_rank": sum(energy_at_rank) / len(energy_at_rank),
            "min_energy_at_rank": min(energy_at_rank),
            "max_energy_at_rank": max(energy_at_rank),
        }

    @staticmethod
    @torch.no_grad()
    def reconstruct_values_with_basis(
        values_slice: torch.Tensor,
        basis: dict,
        center: bool,
    ) -> torch.Tensor:
        """
        Project values to low-rank latent z and reconstruct back to head_dim.

        values_slice:
            [batch, heads, n_tokens, head_dim]

        basis["components"]:
            [heads, rank, head_dim]
        """
        original_dtype = values_slice.dtype

        x = values_slice.detach().float()

        mean = basis["mean"].to(
            device=values_slice.device,
            dtype=torch.float32,
        )
        comp = basis["components"].to(
            device=values_slice.device,
            dtype=torch.float32,
        )

        if center:
            x_centered = x - mean[None, :, None, :]
        else:
            x_centered = x

        # z = x P^T
        coeff = torch.einsum("bhnd,hrd->bhnr", x_centered, comp)

        # v_hat = z P
        recon = torch.einsum("bhnr,hrd->bhnd", coeff, comp)

        if center:
            recon = recon + mean[None, :, None, :]

        return recon.to(dtype=original_dtype)

    @torch.no_grad()
    def _fit_and_reconstruct_prefill(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prefill stage:
          1. Fit SVD basis from full prefill Value cache.
          2. Reconstruct the whole prefill Value cache.
          3. Keep K unchanged.
        """
        layer_idx = int(module.layer_idx)

        if not torch.isfinite(values).all():
            raise RuntimeError(
                f"NaN/Inf found in Value cache before SVD at layer {layer_idx}"
            )

        _, _, seq_len, head_dim = values.shape

        rank = self._get_layer_rank(layer_idx, head_dim)

        if self.basis_method == "naive":
            token_weights = None
        elif self.basis_method == "attn":
            token_weights = self._compute_attention_token_weights(
                module=module,
                hidden_states=hidden_states,
                keys=keys,
                kwargs=kwargs,
            )
        else:
            raise ValueError(f"Unknown basis_method: {self.basis_method}")

        basis = self._fit_svd_basis_for_values(
            values=values,
            rank=rank,
            token_weights=token_weights,
        )

        recon_values = self.reconstruct_values_with_basis(
            values_slice=values,
            basis=basis,
            center=self.center,
        )

        self.bases[layer_idx] = basis
        self.head_dim = int(head_dim)

        layer_feature_kv_ratio = (head_dim + rank) / (2.0 * head_dim)

        self.fit_stats.append({
            "layer": layer_idx,
            "rank": int(rank),
            "head_dim": int(head_dim),
            "seq_len_fit": int(seq_len),
            "basis_method": self.basis_method,
            "center": bool(self.center),
            "attn_obs_len": int(self.attn_obs_len),
            "attn_weight_power": float(self.attn_weight_power),
            "temperature": float(self.temperature),
            "avg_energy_at_rank": float(basis["avg_energy_at_rank"]),
            "min_energy_at_rank": float(basis["min_energy_at_rank"]),
            "max_energy_at_rank": float(basis["max_energy_at_rank"]),
            "layer_feature_kv_ratio_if_materialized": float(layer_feature_kv_ratio),
        })

        return keys, recon_values.contiguous()

    @torch.no_grad()
    def _reconstruct_decode_value(
        self,
        module: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        q_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode stage:
          The model appends new full-dimensional V token(s).
          We project/reconstruct only those newly appended positions using the
          prefill-fitted basis.
        """
        layer_idx = int(module.layer_idx)

        if layer_idx not in self.bases:
            # This can happen if decode is used without prefill fitting.
            return keys, values

        seq_len = int(values.shape[2])
        start = max(0, seq_len - q_len)
        end = seq_len

        if start >= end:
            return keys, values

        new_values = values.clone()

        recon_slice = self.reconstruct_values_with_basis(
            values_slice=values[:, :, start:end, :],
            basis=self.bases[layer_idx],
            center=self.center,
        )

        new_values[:, :, start:end, :] = recon_slice

        return keys, new_values.contiguous()

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Called by BasePress.forward_hook after each attention forward.

        q_len > 1:
            prefill; fit SVD basis and reconstruct full prefill V.

        q_len == 1:
            decode; reconstruct newly appended V token.
        """
        layer_idx = int(module.layer_idx)

        if not self._should_compress_layer(layer_idx):
            return keys, values

        q_len = int(hidden_states.shape[1])

        if q_len > 1:
            return self._fit_and_reconstruct_prefill(
                module=module,
                hidden_states=hidden_states,
                keys=keys,
                values=values,
                kwargs=kwargs,
            )

        return self._reconstruct_decode_value(
            module=module,
            keys=keys,
            values=values,
            q_len=q_len,
        )

    def theoretical_feature_kv_ratio(self) -> float:
        """
        Theoretical KV feature ratio if latent Value cache were materialized.

        K remains full-dimensional: d
        V becomes low-rank latent: r

        Per compressed layer:
            ratio = (d + r) / (2d)

        This dense implementation keeps full tensor shape, so this value should
        be reported separately from measured peak memory.
        """
        if self.num_layers is None or self.head_dim is None:
            return 1.0

        total_ratio = 0.0

        for layer_idx in range(self.num_layers):
            if not self._should_compress_layer(layer_idx):
                total_ratio += 1.0
                continue

            r = self._get_layer_rank(layer_idx, self.head_dim)
            total_ratio += (self.head_dim + r) / (2.0 * self.head_dim)

        return total_ratio / self.num_layers

    def theoretical_feature_kv_saving(self) -> float:
        return 1.0 - self.theoretical_feature_kv_ratio()

    def get_fit_summary(self) -> dict:
        """
        Return lightweight summary for logging.
        """
        return {
            "num_fitted_layers": len(self.bases),
            "basis_method": self.basis_method,
            "rank": self.rank,
            "center": self.center,
            "theoretical_feature_kv_ratio": self.theoretical_feature_kv_ratio(),
            "theoretical_feature_kv_saving": self.theoretical_feature_kv_saving(),
        }
