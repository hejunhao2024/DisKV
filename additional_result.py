from __future__ import annotations

import csv
import gc
import importlib
import inspect
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pythia_kvpress.eval import continuation_ppl
from pythia_kvpress.presses.scorer import ScorerPress
from pythia_kvpress.presses.lagkv import LagKVPress
from pythia_kvpress.our_presses.stat_lagkv import StatLagKVPress

from run_eval_ours import (
    DEFAULT_DATASET_PATHS,
    benchmark_latency_memory,
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

# Important: use float16, not bfloat16, to avoid PPL explosion on Pythia.
LOAD_DTYPE = "float16"
POSITION_MODE = "absolute"

TARGET_KV_RATIO = 0.6
BUDGET = int(round(CONTEXT_LEN * TARGET_KV_RATIO))  # 922

N_SINK = 4
LAG_SIZE = 128
LAG_ALPHA = 0.1
CROSS_SCORING = False

SNAP_WINDOW_SIZE = 32
SNAP_KERNEL_SIZE = 7

PYRAMID_BETA = 20.0

RAW_CSV = "results/additional_result_raw.csv"
SUMMARY_CSV = "results/additional_result_summary.csv"
LATEX_TXT = "results/additional_result_table.txt"


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
    "Budget",
    "LoadDType",
    "PositionMode",

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
    "Method",
    "KVSetting",
    "NumRunsPG19",
    "PG19_PPL",
    "PG19_TTFT",
    "PG19_TPOT",
    "PG19_Throughput",
    "PG19_Memory",
    "NumRunsWikiText",
    "WikiText_PPL",
    "WikiText_TTFT",
    "WikiText_TPOT",
    "WikiText_Throughput",
    "WikiText_Memory",
]


# ============================================================
# Import helpers
# ============================================================

def import_first(candidates: list[tuple[str, str]], required: bool = True):
    errors = []
    for module_name, class_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            print(f"[Import] {class_name} from {module_name}")
            return cls
        except Exception as e:
            errors.append(f"{module_name}.{class_name}: {repr(e)}")

    if required:
        msg = "Could not import required press class. Tried:\n" + "\n".join(errors)
        raise ImportError(msg)

    print("[Import] optional class not found. Tried:")
    for e in errors:
        print("  ", e)
    return None


def make_press(cls, **kwargs):
    """
    Instantiate a press class while only passing supported keyword arguments.
    This makes the script robust to small constructor-name differences.
    """
    sig = inspect.signature(cls)
    params = sig.parameters

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return cls(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return cls(**filtered)


SnapKVPress = import_first(
    [
        ("pythia_kvpress.presses.snapkv", "SnapKVPress"),
        ("pythia_kvpress.presses.snap_kv", "SnapKVPress"),
    ],
    required=True,
)

PyramidKVPress = import_first(
    [
        ("pythia_kvpress.presses.pyramidkv", "PyramidKVPress"),
        ("pythia_kvpress.presses.pyramid_kv", "PyramidKVPress"),
    ],
    required=True,
)

ExternalStreamingLLMPress = import_first(
    [
        ("pythia_kvpress.presses.streamingllm", "StreamingLLMPress"),
        ("pythia_kvpress.presses.streaming_llm", "StreamingLLMPress"),
        ("pythia_kvpress.presses.streaming", "StreamingLLMPress"),
    ],
    required=False,
)


# ============================================================
# Fallback StreamLLM implementation
# ============================================================

@dataclass
class FallbackStreamingLLMPress(ScorerPress):
    """
    StreamLLM-style sink + recent-window cache policy.

    This is prefill-only here:
    - during prefill, keep sink tokens + recent tokens;
    - during decode, no further compression is applied.
    """
    n_sink: int = 4
    recent_size: int = 918

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_heads, seq_len, _ = keys.shape

        scores = torch.zeros(
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
            keys.add((row["Dataset"], row["Method"], int(row["OffsetIndex"])))
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

    # Fallback: evenly spread 5 windows across available text.
    return [
        int(round(i * max_start / (NUM_WINDOWS - 1)))
        for i in range(NUM_WINDOWS)
    ]


def method_plan():
    return [
        {
            "method": "Full KV",
            "kv_setting": "1.0 ref.",
            "kind": "full",
        },
        {
            "method": "SnapKV",
            "kv_setting": "0.6 token",
            "kind": "snapkv",
        },
        {
            "method": "StreamLLM",
            "kv_setting": "0.6 token",
            "kind": "streamllm",
        },
        {
            "method": "LagKV",
            "kv_setting": "0.6 token",
            "kind": "lagkv",
        },
        {
            "method": "PyramidKV",
            "kv_setting": "0.6 token",
            "kind": "pyramidkv",
        },
        {
            "method": "DiSKV",
            "kv_setting": "0.6 token",
            "kind": "diskv",
        },
    ]


def build_press_pair(exp: dict):
    """
    All compression methods are prefill-only:
        return prefill_press, None

    During decoding, new KV tokens are appended normally and are not compressed.
    """
    kind = exp["kind"]

    if kind == "full":
        return None, None

    if kind == "snapkv":
        press = make_press(
            SnapKVPress,
            mode="prefill",
            budget=BUDGET,
            window_size=SNAP_WINDOW_SIZE,
            kernel_size=SNAP_KERNEL_SIZE,
            observation_window=SNAP_WINDOW_SIZE,
            pooling_kernel=SNAP_KERNEL_SIZE,
            n_sink=N_SINK,
        )
        return press, None

    if kind == "streamllm":
        recent_size = max(1, BUDGET - N_SINK)

        if ExternalStreamingLLMPress is not None:
            press = make_press(
                ExternalStreamingLLMPress,
                mode="prefill",
                budget=BUDGET,
                n_sink=N_SINK,
                recent_size=recent_size,
                window_size=recent_size,
            )
        else:
            press = FallbackStreamingLLMPress(
                mode="prefill",
                budget=BUDGET,
                n_sink=N_SINK,
                recent_size=recent_size,
            )

        return press, None

    if kind == "lagkv":
        return (
            LagKVPress(
                mode="prefill",
                budget=BUDGET,
                n_sink=N_SINK,
                lag_size=LAG_SIZE,
                cross_scoring=CROSS_SCORING,
            ),
            None,
        )

    if kind == "pyramidkv":
        press = make_press(
            PyramidKVPress,
            mode="prefill",
            budget=BUDGET,
            n_sink=N_SINK,
            window_size=SNAP_WINDOW_SIZE,
            kernel_size=SNAP_KERNEL_SIZE,
            observation_window=SNAP_WINDOW_SIZE,
            pooling_kernel=SNAP_KERNEL_SIZE,
            beta=PYRAMID_BETA,
        )
        return press, None

    if kind == "diskv":
        return (
            StatLagKVPress(
                mode="prefill",
                budget=BUDGET,
                n_sink=N_SINK,
                lag_size=LAG_SIZE,
                cross_scoring=CROSS_SCORING,
                alpha=LAG_ALPHA,
            ),
            None,
        )

    raise ValueError(f"Unknown method kind: {kind}")


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
# Run
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
    print(f"Method:  {exp['method']}")
    print(f"Offset:  {offset_index} / token_offset={token_offset}")
    print(f"Budget:  {BUDGET}")
    print("Compression mode: prefill-only; decode compression disabled")

    row_base = {
        "Status": "ok",
        "Error": "",

        "Model": MODEL_NAME,
        "ModelPath": MODEL_PATH,
        "Dataset": dataset,
        "DatasetPath": dataset_path,
        "Method": exp["method"],
        "KVSetting": exp["kv_setting"],

        "OffsetIndex": offset_index,
        "TokenOffset": token_offset,
        "ContextLen": CONTEXT_LEN,
        "TargetLen": TARGET_LEN,
        "Budget": BUDGET,
        "LoadDType": LOAD_DTYPE,
        "PositionMode": POSITION_MODE,
    }

    try:
        prefill_press, decoding_press = build_press_pair(exp)
        assert decoding_press is None, "This script should use prefill-only compression."

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
        assert decoding_press is None, "This script should use prefill-only compression."

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
            "PPL": "",
            "TTFT": "",
            "TPOT": "",
            "Throughput": "",
            "Memory": "",
            "PrefillCacheLenAvg": "",
            "FinalCacheLenAvg": "",
        }


def run_all():
    existing_ok = read_existing_ok_keys(RAW_CSV)

    model, tokenizer, device = load_model_once()
    plan = method_plan()

    print("Method plan:")
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
                key = (dataset, exp["method"], offset_index)
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
# Aggregate and LaTeX
# ============================================================

def to_float(x):
    try:
        if x == "" or x is None:
            return None
        return float(x)
    except Exception:
        return None


def mean_metric(rows: list[dict], name: str):
    vals = [to_float(r.get(name, "")) for r in rows]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return mean(vals) if vals else ""


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

    for exp in method_plan():
        method = exp["method"]
        kv_setting = exp["kv_setting"]

        pg_rows = [r for r in rows if r["Method"] == method and r["Dataset"] == "pg19"]
        wk_rows = [r for r in rows if r["Method"] == method and r["Dataset"] == "wikitext"]

        summary_rows.append(
            {
                "Model": MODEL_NAME,
                "Method": method,
                "KVSetting": kv_setting,

                "NumRunsPG19": len(pg_rows),
                "PG19_PPL": mean_metric(pg_rows, "PPL"),
                "PG19_TTFT": mean_metric(pg_rows, "TTFT"),
                "PG19_TPOT": mean_metric(pg_rows, "TPOT"),
                "PG19_Throughput": mean_metric(pg_rows, "Throughput"),
                "PG19_Memory": mean_metric(pg_rows, "Memory"),

                "NumRunsWikiText": len(wk_rows),
                "WikiText_PPL": mean_metric(wk_rows, "PPL"),
                "WikiText_TTFT": mean_metric(wk_rows, "TTFT"),
                "WikiText_TPOT": mean_metric(wk_rows, "TPOT"),
                "WikiText_Throughput": mean_metric(wk_rows, "Throughput"),
                "WikiText_Memory": mean_metric(wk_rows, "Memory"),
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


def fmt(x, digits=2):
    if x == "" or x is None:
        return "--"
    try:
        x = float(x)
        if not math.isfinite(x):
            return "--"
        return f"{x:.{digits}f}"
    except Exception:
        return "--"


def generate_latex(summary_rows: list[dict]):
    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{")
    lines.append(r"Additional baseline comparison on Pythia-6.9B at target KV ratio 0.6.")
    lines.append(r"We use context length 1536 and target length 512.")
    lines.append(r"All compression methods apply prefill-only KV compression, and no additional compression is applied during decoding.")
    lines.append(r"Results are averaged over five text offsets.")
    lines.append(r"}")
    lines.append(r"\label{tab:additional_baselines}")
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
    lines.append(r"\rowcolor{gray!10}")
    lines.append(r"\multicolumn{10}{c}{Pythia-6.9B} \\")
    lines.append(r"\midrule")

    for row in summary_rows:
        method = row["Method"]
        kv = row["KVSetting"]

        pg_ppl = fmt(row["PG19_PPL"], 2)
        pg_ttft = fmt(row["PG19_TTFT"], 4)
        pg_tpot = fmt(row["PG19_TPOT"], 2)
        pg_thr = fmt(row["PG19_Throughput"], 2)

        wk_ppl = fmt(row["WikiText_PPL"], 2)
        wk_ttft = fmt(row["WikiText_TTFT"], 4)
        wk_tpot = fmt(row["WikiText_TPOT"], 2)
        wk_thr = fmt(row["WikiText_Throughput"], 2)

        lines.append(
            f"{method}\n"
            f"& {kv}\n"
            f"& {pg_ppl} & {pg_ttft} & {pg_tpot} & {pg_thr}\n"
            f"& {wk_ppl} & {wk_ttft} & {wk_tpot} & {wk_thr} \\\\"
        )
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
    print("Running additional baseline comparison")
    print(f"Model:       {MODEL_NAME}")
    print(f"Raw CSV:     {RAW_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"LaTeX TXT:   {LATEX_TXT}")
    print("Important: dtype is fixed to float16 to avoid PPL explosion.")
    print("All compression methods are prefill-only; decode compression is disabled.")

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
