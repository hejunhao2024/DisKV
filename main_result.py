from __future__ import annotations

import csv
import gc
import math
import traceback
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pythia_kvpress.eval import continuation_ppl
from pythia_kvpress.presses.lagkv import LagKVPress
from pythia_kvpress.our_presses.stat_lagkv import StatLagKVPress
from pythia_kvpress.our_presses.svd_stat_lagkv import SVDStatLagKVPress
from pythia_kvpress.our_presses.svd_value_kv import parse_layer_rank_map

from run_eval_ours import (
    DEFAULT_DATASET_PATHS,
    benchmark_latency_memory,
    get_dtype,
    load_text,
)


# ============================================================
# Fixed main-table experiment config
# ============================================================

MODELS = [
    "models/pythia-70m",
    "models/pythia-6.9b",
]

DATASETS = [
    "pg19",
    "wikitext",
]

DATASET_DISPLAY = {
    "pg19": "PG-19",
    "wikitext": "WikiText-2",
}

MODEL_DISPLAY = {
    "pythia-70m": "Pythia-70M",
    "pythia-6.9b": "Pythia-6.9B",
}

CONTEXT_LEN = 1536
TARGET_LEN = 512
TOTAL_LEN = CONTEXT_LEN + TARGET_LEN

NUM_WINDOWS = 5
TARGET_KV_RATIO = 0.6

LOAD_DTYPE = "float16"
POSITION_MODE = "absolute"

N_SINK = 4
LAG_SIZE = 128
LAG_ALPHA = 0.1
CROSS_SCORING = False

# DiSKV + 20% SVD:
# value dimension keep ratio = 0.8.
# Since K is not compressed, total feature ratio = (1 + 0.8) / 2 = 0.9.
# We adjust token budget so total KV ratio ~= 0.6.
SVD_VALUE_KEEP_RATIO = 0.8
SVD_TEMPERATURE = 4.0
SVD_ATTN_OBS_LEN = 128
SVD_ATTN_WEIGHT_POWER = 1.0
SVD_CENTER = False
SVD_DEVICE = "cuda"
SVD_WEIGHT_CAP_QUANTILE = 0.0

RAW_CSV = "results/main_result_raw.csv"
SUMMARY_CSV = "results/main_result_summary.csv"
LATEX_TXT = "results/main_result_table.txt"


RAW_FIELDS = [
    "Status",
    "Error",

    "Model",
    "ModelPath",
    "Dataset",
    "DatasetPath",
    "Method",
    "KVSetting",

    "OffsetIndex",
    "TokenOffset",
    "ContextLen",
    "TargetLen",
    "LoadDType",
    "PositionMode",

    "Budget",
    "LayerRanks",
    "SVDValueKeepRatio",
    "SVDTotalFeatureRatio",

    "PPL",
    "TTFT",
    "TPOT",
    "Throughput",
    "Memory",
    "PrefillCacheLenAvg",
    "FinalCacheLenAvg",
]


SUMMARY_FIELDS = [
    "Model",
    "Dataset",
    "Method",
    "KVSetting",
    "NumRuns",

    "PPL_mean",
    "TTFT_mean",
    "TPOT_mean",
    "Throughput_mean",
    "Memory_mean",

    "Budget",
    "LayerRanks",
    "SVDValueKeepRatio",
    "SVDTotalFeatureRatio",
]


# ============================================================
# Utilities
# ============================================================

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def short_model_name(model_path: str) -> str:
    return Path(model_path).name or model_path


def display_model_name(model_path: str) -> str:
    return MODEL_DISPLAY.get(short_model_name(model_path), short_model_name(model_path))


def append_csv(path: str, row: dict, fields: list[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def read_existing_ok_keys(path: str) -> set[tuple[str, str, str, int]]:
    path = Path(path)
    if not path.exists():
        return set()

    keys = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status") != "ok":
                continue
            keys.add(
                (
                    row["ModelPath"],
                    row["Dataset"],
                    row["Method"],
                    int(row["OffsetIndex"]),
                )
            )
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

    # Prefer non-overlapping windows: 0, 2048, 4096, ...
    non_overlap = [i * TOTAL_LEN for i in range(NUM_WINDOWS)]
    if non_overlap[-1] <= max_start:
        return non_overlap

    # Fallback: evenly spread 5 windows across available text.
    if NUM_WINDOWS == 1:
        return [0]

    return [
        int(round(i * max_start / (NUM_WINDOWS - 1)))
        for i in range(NUM_WINDOWS)
    ]


def infer_model_shape(model):
    cfg = model.config

    num_layers = getattr(cfg, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(cfg, "n_layer", None)

    num_heads = getattr(cfg, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(cfg, "n_head", None)

    hidden_size = getattr(cfg, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(cfg, "n_embd", None)

    if num_layers is None or num_heads is None or hidden_size is None:
        raise ValueError("Cannot infer model shape from config.")

    head_dim = hidden_size // num_heads

    return {
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "hidden_size": int(hidden_size),
        "head_dim": int(head_dim),
    }


def make_layer_rank_spec(num_layers: int, front_rank: int, back_rank: int) -> str:
    mid = num_layers // 2
    return f"0-{mid - 1}:{front_rank},{mid}-{num_layers - 1}:{back_rank}"


def svd_20_rank_policy(model_shape: dict):
    """
    20% value-dimension compression.

    Constraint:
        front half drops fewer dimensions:
        drop_front = 0.5 * drop_back

    Target:
        average value keep ratio ~= 0.8
    """
    d = model_shape["head_dim"]
    target_keep = SVD_VALUE_KEEP_RATIO

    # drop_front = a, drop_back = 2a
    # avg keep = ((d-a) + (d-2a)) / (2d)
    #          = 1 - 3a/(2d)
    # a = (2d/3) * (1 - target_keep)
    drop_front = int(round((2.0 * d / 3.0) * (1.0 - target_keep)))
    drop_back = int(round(2.0 * drop_front))

    drop_front = max(0, min(d - 1, drop_front))
    drop_back = max(0, min(d - 1, drop_back))

    front_rank = d - drop_front
    back_rank = d - drop_back

    actual_value_keep = (front_rank + back_rank) / (2.0 * d)
    total_feature_ratio = (1.0 + actual_value_keep) / 2.0

    layer_ranks = make_layer_rank_spec(
        num_layers=model_shape["num_layers"],
        front_rank=front_rank,
        back_rank=back_rank,
    )

    return {
        "front_rank": front_rank,
        "back_rank": back_rank,
        "layer_ranks": layer_ranks,
        "actual_value_keep": actual_value_keep,
        "total_feature_ratio": total_feature_ratio,
    }


def method_plan(model_shape: dict):
    token_budget_06 = int(round(CONTEXT_LEN * TARGET_KV_RATIO))

    svd_info = svd_20_rank_policy(model_shape)
    diskv_svd_budget = int(round(CONTEXT_LEN * TARGET_KV_RATIO / svd_info["total_feature_ratio"]))
    diskv_svd_budget = max(N_SINK + 2, min(CONTEXT_LEN, diskv_svd_budget))

    return [
        {
            "method": "Full KV",
            "kv_setting": "1.0",
            "internal": "fullkv",
            "budget": "",
            "layer_ranks": "",
            "svd_value_keep": "",
            "svd_total_feature_ratio": "",
        },
        {
            "method": "LagKV",
            "kv_setting": "0.6 token",
            "internal": "lagkv",
            "budget": token_budget_06,
            "layer_ranks": "",
            "svd_value_keep": "",
            "svd_total_feature_ratio": "",
        },
        {
            "method": "DiSKV",
            "kv_setting": "0.6 token",
            "internal": "diskv",
            "budget": token_budget_06,
            "layer_ranks": "",
            "svd_value_keep": "",
            "svd_total_feature_ratio": "",
        },
        {
            "method": "DiSKV + 20% SVD",
            "kv_setting": "0.6 total",
            "internal": "diskv_svd",
            "budget": diskv_svd_budget,
            "layer_ranks": svd_info["layer_ranks"],
            "svd_value_keep": svd_info["actual_value_keep"],
            "svd_total_feature_ratio": svd_info["total_feature_ratio"],
        },
    ]


def build_press_pair(exp: dict):
    internal = exp["internal"]

    if internal == "fullkv":
        return None, None

    if internal == "lagkv":
        return (
            LagKVPress(
                mode="prefill",
                budget=int(exp["budget"]),
                n_sink=N_SINK,
                lag_size=LAG_SIZE,
                cross_scoring=CROSS_SCORING,
            ),
            None,
        )

    if internal == "diskv":
        return (
            StatLagKVPress(
                mode="prefill",
                budget=int(exp["budget"]),
                n_sink=N_SINK,
                lag_size=LAG_SIZE,
                cross_scoring=CROSS_SCORING,
                alpha=LAG_ALPHA,
            ),
            None,
        )

    if internal == "diskv_svd":
        press = SVDStatLagKVPress(
            mode="both",
            rank=1,
            basis_method="attn",
            center=SVD_CENTER,
            svd_device=SVD_DEVICE,
            attn_obs_len=SVD_ATTN_OBS_LEN,
            attn_weight_power=SVD_ATTN_WEIGHT_POWER,
            temperature=SVD_TEMPERATURE,
            weight_cap_quantile=SVD_WEIGHT_CAP_QUANTILE,
            layer_rank_map=parse_layer_rank_map(exp["layer_ranks"]),
            budget=int(exp["budget"]),
            n_sink=N_SINK,
            lag_size=LAG_SIZE,
            cross_scoring=CROSS_SCORING,
            lag_alpha=LAG_ALPHA,
        )
        return press, press

    raise ValueError(f"Unknown internal method: {internal}")


# ============================================================
# Experiment
# ============================================================

def run_one(
    model,
    input_ids: torch.Tensor,
    model_path: str,
    dataset: str,
    dataset_path: str,
    exp: dict,
    offset_index: int,
    token_offset: int,
):
    cleanup_cuda()

    print("=" * 100)
    print(f"Model:   {display_model_name(model_path)}")
    print(f"Dataset: {DATASET_DISPLAY[dataset]}")
    print(f"Method:  {exp['method']}")
    print(f"Offset:  {offset_index} / token_offset={token_offset}")
    print(f"Budget:  {exp['budget']}")
    print(f"Ranks:   {exp['layer_ranks']}")

    row_base = {
        "Status": "ok",
        "Error": "",

        "Model": display_model_name(model_path),
        "ModelPath": model_path,
        "Dataset": dataset,
        "DatasetPath": dataset_path,
        "Method": exp["method"],
        "KVSetting": exp["kv_setting"],

        "OffsetIndex": offset_index,
        "TokenOffset": token_offset,
        "ContextLen": CONTEXT_LEN,
        "TargetLen": TARGET_LEN,
        "LoadDType": LOAD_DTYPE,
        "PositionMode": POSITION_MODE,

        "Budget": exp["budget"],
        "LayerRanks": exp["layer_ranks"],
        "SVDValueKeepRatio": exp["svd_value_keep"],
        "SVDTotalFeatureRatio": exp["svd_total_feature_ratio"],
    }

    try:
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

        prefill_press, decoding_press = build_press_pair(exp)

        print("Running latency/memory benchmark...")
        perf = benchmark_latency_memory(
            model=model,
            input_ids=input_ids,
            context_len=CONTEXT_LEN,
            target_len=TARGET_LEN,
            prefill_press=prefill_press,
            decoding_press=decoding_press,
            position_mode=POSITION_MODE,
        )

        row = {
            **row_base,
            "PPL": ppl,
            "TTFT": perf.get("ttft_s", ""),
            "TPOT": perf.get("tpot_ms", ""),
            "Throughput": perf.get("throughput_tok_s", ""),
            "Memory": perf.get("peak_mem_mb", ""),
            "PrefillCacheLenAvg": perf.get("prefill_cache_len_avg", ""),
            "FinalCacheLenAvg": perf.get("final_cache_len_avg", ""),
        }

        del prefill_press, decoding_press
        cleanup_cuda()

        print("Done:")
        print(f"  PPL:        {row['PPL']}")
        print(f"  TTFT:       {row['TTFT']}")
        print(f"  TPOT:       {row['TPOT']}")
        print(f"  Throughput: {row['Throughput']}")
        print(f"  Memory:     {row['Memory']}")

        return row

    except Exception as e:
        cleanup_cuda()
        print("FAILED:")
        print(traceback.format_exc())
        return {
            **row_base,
            "Status": "failed",
            "Error": repr(e),
        }


def load_model_once(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = get_dtype(LOAD_DTYPE, device)

    print("=" * 100)
    print(f"Loading model once: {model_path}")
    print(f"device: {device}")
    print(f"dtype:  {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )

    model = model.to(device).eval()

    return model, tokenizer, device


def run_all():
    existing_ok = read_existing_ok_keys(RAW_CSV)

    for model_path in MODELS:
        model, tokenizer, device = load_model_once(model_path)
        model_shape = infer_model_shape(model)
        plan = method_plan(model_shape)

        print("Model shape:", model_shape)
        print("Plan:")
        for exp in plan:
            print(exp)

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

                for exp in plan:
                    key = (model_path, dataset, exp["method"], offset_index)
                    if key in existing_ok:
                        print(f"Skipping existing ok row: {key}")
                        continue

                    row = run_one(
                        model=model,
                        input_ids=input_ids,
                        model_path=model_path,
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
# Aggregation and LaTeX generation
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

    groups = {}

    with raw_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status") != "ok":
                continue

            key = (
                row["Model"],
                row["Dataset"],
                row["Method"],
            )

            groups.setdefault(key, []).append(row)

    summary_rows = []

    # Preserve desired order.
    for model_path in MODELS:
        model = display_model_name(model_path)
        for dataset in DATASETS:
            for method in [
                "Full KV",
                "LagKV",
                "DiSKV",
                "DiSKV + 20% SVD",
            ]:
                key = (model, dataset, method)
                rows = groups.get(key, [])

                if not rows:
                    continue

                def metric_mean(name: str):
                    vals = [to_float(r.get(name, "")) for r in rows]
                    vals = [v for v in vals if v is not None and math.isfinite(v)]
                    return mean(vals) if vals else ""

                first = rows[0]

                summary_rows.append(
                    {
                        "Model": model,
                        "Dataset": dataset,
                        "Method": method,
                        "KVSetting": first.get("KVSetting", ""),
                        "NumRuns": len(rows),

                        "PPL_mean": metric_mean("PPL"),
                        "TTFT_mean": metric_mean("TTFT"),
                        "TPOT_mean": metric_mean("TPOT"),
                        "Throughput_mean": metric_mean("Throughput"),
                        "Memory_mean": metric_mean("Memory"),

                        "Budget": first.get("Budget", ""),
                        "LayerRanks": first.get("LayerRanks", ""),
                        "SVDValueKeepRatio": first.get("SVDValueKeepRatio", ""),
                        "SVDTotalFeatureRatio": first.get("SVDTotalFeatureRatio", ""),
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


def fmt_metric(value, digits=2):
    if value == "" or value is None:
        return "--"
    try:
        x = float(value)
        if not math.isfinite(x):
            return "--"
        return f"{x:.{digits}f}"
    except Exception:
        return "--"


def find_summary(summary_rows, model: str, dataset: str, method: str):
    for row in summary_rows:
        if row["Model"] == model and row["Dataset"] == dataset and row["Method"] == method:
            return row
    return None


def latex_row(summary_rows, model: str, method: str):
    pg = find_summary(summary_rows, model, "pg19", method)
    wk = find_summary(summary_rows, model, "wikitext", method)

    if pg is None:
        pg_vals = ["--", "--", "--", "--"]
        kv_setting = "--"
    else:
        kv_setting = pg["KVSetting"]
        pg_vals = [
            fmt_metric(pg["PPL_mean"], 2),
            fmt_metric(pg["TTFT_mean"], 4),
            fmt_metric(pg["TPOT_mean"], 2),
            fmt_metric(pg["Throughput_mean"], 2),
        ]

    if wk is None:
        wk_vals = ["--", "--", "--", "--"]
    else:
        wk_vals = [
            fmt_metric(wk["PPL_mean"], 2),
            fmt_metric(wk["TTFT_mean"], 4),
            fmt_metric(wk["TPOT_mean"], 2),
            fmt_metric(wk["Throughput_mean"], 2),
        ]

    return (
        f"{method}\n"
        f"& {kv_setting}\n"
        f"& {pg_vals[0]} & {pg_vals[1]} & {pg_vals[2]} & {pg_vals[3]}\n"
        f"& {wk_vals[0]} & {wk_vals[1]} & {wk_vals[2]} & {wk_vals[3]} \\\\"
    )


def generate_latex(summary_rows):
    methods = [
        "Full KV",
        "LagKV",
        "DiSKV",
        "DiSKV + 20% SVD",
    ]

    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(r"Main results at target KV ratio 0.6 on PG-19 and WikiText-2.")
    lines.append(r"We use context length 1536 and target length 512.")
    lines.append(r"For DiSKV+SVD, 20\% value-dimension compression is applied and the token budget is adjusted to match the same overall KV ratio.")
    lines.append(r"}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{llrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Method} & \multirow{2}{*}{KV Setting}")
    lines.append(r"& \multicolumn{4}{c}{PG-19}")
    lines.append(r"& \multicolumn{4}{c}{WikiText-2} \\")
    lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    lines.append(r"&")
    lines.append(r"& PPL $\downarrow$ & TTFT $\downarrow$ & TPOT $\downarrow$ & Throughput $\uparrow$")
    lines.append(r"& PPL $\downarrow$ & TTFT $\downarrow$ & TPOT $\downarrow$ & Throughput $\uparrow$ \\")
    lines.append(r"\midrule")

    for block_i, model_path in enumerate(MODELS):
        model = display_model_name(model_path)

        if block_i > 0:
            lines.append(r"\midrule")

        lines.append(rf"\multicolumn{{10}}{{c}}{{{model}}} \\")
        lines.append(r"\midrule")

        for method in methods:
            lines.append(latex_row(summary_rows, model, method))
            lines.append("")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)

    out = Path(LATEX_TXT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")

    return latex


def main():
    print("=" * 100)
    print("Running main-result experiments")
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
