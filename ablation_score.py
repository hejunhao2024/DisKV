from __future__ import annotations

import csv
import gc
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pythia_kvpress.eval import continuation_ppl
from pythia_kvpress.presses.scorer import ScorerPress
from pythia_kvpress.our_presses.stat_lagkv import StatLagKVPress

from run_eval_ours import (
    DEFAULT_DATASET_PATHS,
    get_dtype,
    load_text,
)


# ============================================================
# Config
# ============================================================

MODEL_PATH = "models/pythia-6.9b"
MODEL_NAME = "Pythia-6.9B"

DATASETS = ["pg19", "wikitext"]
DATASET_DISPLAY = {
    "pg19": "PG-19",
    "wikitext": "WikiText-2",
}

CONTEXT_LEN = 1536
TARGET_LEN = 512
TOTAL_LEN = CONTEXT_LEN + TARGET_LEN
NUM_WINDOWS = 5

LOAD_DTYPE = "float16"   # important: avoid PPL explosion
POSITION_MODE = "absolute"

TARGET_KV_RATIO = 0.6
BUDGET = int(round(CONTEXT_LEN * TARGET_KV_RATIO))  # 922 for 1536

N_SINK = 4
LAG_SIZE = 128
CROSS_SCORING = False

RANDOM_SEED = 2026

RAW_CSV = "results/ablation_score_raw.csv"
SUMMARY_CSV = "results/ablation_score_summary.csv"
LATEX_TXT = "results/ablation_score_table.txt"


RAW_FIELDS = [
    "Status",
    "Error",
    "Model",
    "ModelPath",
    "Dataset",
    "DatasetPath",
    "Variant",
    "Alpha",
    "OffsetIndex",
    "TokenOffset",
    "ContextLen",
    "TargetLen",
    "Budget",
    "LoadDType",
    "PositionMode",
    "PPL",
]

SUMMARY_FIELDS = [
    "Model",
    "Variant",
    "Alpha",
    "NumRunsPG19",
    "PG19PPL",
    "NumRunsWikiText",
    "WikiTextPPL",
]


# ============================================================
# Random-score press
# ============================================================

@dataclass
class RandomScorePress(ScorerPress):
    """
    Random importance scores for non-protected KV positions.

    This keeps the same budget/top-k mechanism as other ScorerPress methods.
    We assign very high scores to sink tokens and the recent tail so that the
    random baseline is not unfairly weakened by deleting protected positions.
    """
    n_sink: int = 4
    recent_size: int = 128

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        # keys shape is usually [batch, num_heads, seq_len, head_dim]
        bsz, num_heads, seq_len, _ = keys.shape

        scores = torch.rand(
            (bsz, num_heads, seq_len),
            device=keys.device,
            dtype=torch.float32,
        )

        high = torch.finfo(scores.dtype).max / 10.0

        if self.n_sink > 0:
            sink = min(self.n_sink, seq_len)
            scores[:, :, :sink] = high

        if self.recent_size > 0:
            start = max(0, seq_len - self.recent_size)
            scores[:, :, start:] = high

        return scores


# ============================================================
# Utilities
# ============================================================

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def append_csv(path: str, row: dict, fields: list[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def read_existing_ok_keys(path: str) -> set[tuple[str, str, int]]:
    path = Path(path)
    if not path.exists():
        return set()

    keys = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status") != "ok":
                continue
            keys.add((row["Dataset"], row["Variant"], int(row["OffsetIndex"])))
    return keys


def tokenize_all(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids


def make_offsets(total_tokens: int) -> list[int]:
    max_start = total_tokens - TOTAL_LEN
    if max_start < 0:
        raise ValueError(
            f"Not enough tokens: got {total_tokens}, need at least {TOTAL_LEN}."
        )

    # Prefer non-overlapping 2048-token windows.
    non_overlap = [i * TOTAL_LEN for i in range(NUM_WINDOWS)]
    if non_overlap[-1] <= max_start:
        return non_overlap

    # Fallback: evenly spread 5 windows.
    return [
        int(round(i * max_start / (NUM_WINDOWS - 1)))
        for i in range(NUM_WINDOWS)
    ]


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def variants():
    return [
        {
            "variant": "Full KV",
            "alpha": "--",
            "kind": "full",
        },
        {
            "variant": "Random score",
            "alpha": "--",
            "kind": "random",
        },
        {
            "variant": "Cosine only",
            "alpha": "0.0",
            "kind": "diskv",
            "alpha_value": 0.0,
        },
        {
            "variant": "DiSKV",
            "alpha": "0.1",
            "kind": "diskv",
            "alpha_value": 0.1,
        },
        {
            "variant": "Balanced",
            "alpha": "0.5",
            "kind": "diskv",
            "alpha_value": 0.5,
        },
        {
            "variant": "Z-score only",
            "alpha": "1.0",
            "kind": "diskv",
            "alpha_value": 1.0,
        },
    ]


def build_press_pair(exp: dict):
    kind = exp["kind"]

    if kind == "full":
        return None, None

    if kind == "random":
        return (
            RandomScorePress(
                mode="prefill",
                budget=BUDGET,
                n_sink=N_SINK,
                recent_size=LAG_SIZE,
            ),
            None,
        )

    if kind == "diskv":
        return (
            StatLagKVPress(
                mode="prefill",
                budget=BUDGET,
                n_sink=N_SINK,
                lag_size=LAG_SIZE,
                cross_scoring=CROSS_SCORING,
                alpha=float(exp["alpha_value"]),
            ),
            None,
        )

    raise ValueError(f"Unknown variant kind: {kind}")


def load_model_once():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = get_dtype(LOAD_DTYPE, device)

    print("=" * 100)
    print(f"Loading model once: {MODEL_PATH}")
    print(f"device: {device}")
    print(f"dtype:  {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
        )

    model = model.to(device).eval()
    return model, tokenizer, device


# ============================================================
# Run experiments
# ============================================================

def run_one(
    model,
    input_ids: torch.Tensor,
    dataset: str,
    dataset_path: str,
    exp: dict,
    offset_index: int,
    token_offset: int,
):
    cleanup_cuda()

    print("=" * 100)
    print(f"Model:   {MODEL_NAME}")
    print(f"Dataset: {DATASET_DISPLAY[dataset]}")
    print(f"Variant: {exp['variant']}")
    print(f"Alpha:   {exp['alpha']}")
    print(f"Offset:  {offset_index} / token_offset={token_offset}")
    print(f"Budget:  {BUDGET}")

    row_base = {
        "Status": "ok",
        "Error": "",
        "Model": MODEL_NAME,
        "ModelPath": MODEL_PATH,
        "Dataset": dataset,
        "DatasetPath": dataset_path,
        "Variant": exp["variant"],
        "Alpha": exp["alpha"],
        "OffsetIndex": offset_index,
        "TokenOffset": token_offset,
        "ContextLen": CONTEXT_LEN,
        "TargetLen": TARGET_LEN,
        "Budget": BUDGET,
        "LoadDType": LOAD_DTYPE,
        "PositionMode": POSITION_MODE,
    }

    try:
        # Make random baseline reproducible.
        # Non-random variants are deterministic, so this does not hurt.
        dataset_seed_offset = 0 if dataset == "pg19" else 1000
        set_seed(RANDOM_SEED + dataset_seed_offset + offset_index)

        prefill_press, decoding_press = build_press_pair(exp)

        print("Running PPL...")
        ppl = continuation_ppl(
            model=model,
            input_ids=input_ids,
            context_len=CONTEXT_LEN,
            target_len=TARGET_LEN,
            prefill_press=prefill_press,
            decoding_press=decoding_press,
            position_mode=POSITION_MODE,
            count_first_target=False,
        )

        del prefill_press, decoding_press
        cleanup_cuda()

        row = {
            **row_base,
            "PPL": ppl,
        }

        print("Done:")
        print(f"  PPL: {ppl}")

        return row

    except Exception as e:
        cleanup_cuda()
        print("FAILED:")
        print(traceback.format_exc())
        return {
            **row_base,
            "Status": "failed",
            "Error": repr(e),
            "PPL": "",
        }


def run_all():
    existing_ok = read_existing_ok_keys(RAW_CSV)

    model, tokenizer, device = load_model_once()

    for dataset in DATASETS:
        dataset_path = DEFAULT_DATASET_PATHS[dataset]

        print("=" * 100)
        print(f"Loading dataset: {dataset}")
        print(f"path: {dataset_path}")

        text = load_text(dataset, dataset_path)
        all_ids = tokenize_all(tokenizer, text)
        offsets = make_offsets(all_ids.shape[1])

        print(f"total tokens: {all_ids.shape[1]}")
        print(f"offsets: {offsets}")

        for offset_index, token_offset in enumerate(offsets):
            input_ids = all_ids[:, token_offset: token_offset + TOTAL_LEN].to(device)
            assert input_ids.shape[1] == TOTAL_LEN

            for exp in variants():
                key = (dataset, exp["variant"], offset_index)
                if key in existing_ok:
                    print(f"Skipping existing ok row: {key}")
                    continue

                row = run_one(
                    model=model,
                    input_ids=input_ids,
                    dataset=dataset,
                    dataset_path=dataset_path,
                    exp=exp,
                    offset_index=offset_index,
                    token_offset=token_offset,
                )

                append_csv(RAW_CSV, row, RAW_FIELDS)

                if row["Status"] == "ok":
                    existing_ok.add(key)

    del model, tokenizer
    cleanup_cuda()


# ============================================================
# Aggregate and generate LaTeX
# ============================================================

def to_float(x):
    try:
        if x == "" or x is None:
            return None
        return float(x)
    except Exception:
        return None


def aggregate_results():
    raw_path = Path(RAW_CSV)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    rows = []
    with raw_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status") == "ok":
                rows.append(row)

    summary_rows = []

    for exp in variants():
        variant = exp["variant"]
        alpha = exp["alpha"]

        pg_vals = []
        wiki_vals = []

        for row in rows:
            if row["Variant"] != variant:
                continue

            ppl = to_float(row.get("PPL", ""))
            if ppl is None or not math.isfinite(ppl):
                continue

            if row["Dataset"] == "pg19":
                pg_vals.append(ppl)
            elif row["Dataset"] == "wikitext":
                wiki_vals.append(ppl)

        summary_rows.append(
            {
                "Model": MODEL_NAME,
                "Variant": variant,
                "Alpha": alpha,
                "NumRunsPG19": len(pg_vals),
                "PG19PPL": mean(pg_vals) if pg_vals else "",
                "NumRunsWikiText": len(wiki_vals),
                "WikiTextPPL": mean(wiki_vals) if wiki_vals else "",
            }
        )

    out = Path(SUMMARY_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})

    return summary_rows


def fmt_ppl(x):
    if x == "" or x is None:
        return "--"
    try:
        x = float(x)
        if not math.isfinite(x):
            return "--"
        return f"{x:.2f}"
    except Exception:
        return "--"


def generate_latex(summary_rows):
    lines = []

    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(r"Ablation of the scoring function on Pythia-6.9B at target KV ratio 0.6.")
    lines.append(r"All variants use context length 1536, target length 512, the same cache budget, and five text offsets.")
    lines.append(r"}")
    lines.append(r"\label{tab:score_ablation}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & $\alpha$ & PG-19 PPL $\downarrow$ & WikiText-2 PPL $\downarrow$ \\")
    lines.append(r"\midrule")

    for row in summary_rows:
        variant = row["Variant"]
        alpha = row["Alpha"]
        pg = fmt_ppl(row["PG19PPL"])
        wiki = fmt_ppl(row["WikiTextPPL"])

        lines.append(f"{variant} & {alpha} & {pg} & {wiki} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    out = Path(LATEX_TXT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")

    return latex


def main():
    print("=" * 100)
    print("Running score ablation")
    print(f"Raw CSV:     {RAW_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"LaTeX TXT:   {LATEX_TXT}")
    print("Important: dtype is fixed to float16 to avoid PPL explosion.")

    run_all()

    print("=" * 100)
    print("Aggregating results")
    summary_rows = aggregate_results()

    print("=" * 100)
    print("Generating LaTeX table")
    latex = generate_latex(summary_rows)

    print("=" * 100)
    print(f"Saved raw CSV to:     {RAW_CSV}")
    print(f"Saved summary CSV to: {SUMMARY_CSV}")
    print(f"Saved LaTeX table to: {LATEX_TXT}")
    print()
    print(latex)


if __name__ == "__main__":
    main()
