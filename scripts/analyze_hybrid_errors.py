#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from htr_ocr.data.collate import collate_line_batch
from htr_ocr.data.dataset import IamLineDataset
from htr_ocr.data.transforms import make_image_transform
from htr_ocr.text.ctc_decode import ctc_beam_search_batch, ctc_greedy_decode_batch
from htr_ocr.train.hybrid_infer import load_checkpoint as load_hybrid_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid CTC error analysis export (tables for charts) on selected splits."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best Hybrid CTC checkpoint (*.pt).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed",
        help=f"Directory with split CSV files (default: {BASE_DIR / 'data' / 'processed'})",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated splits for analysis (default: val,test). Example: val,test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "hybrid_error_export",
        help=f"Output directory in project root (default: {BASE_DIR / 'hybrid_error_export'})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference: cuda/cpu (default: cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--decode-method",
        type=str,
        default="beam",
        choices=["beam", "greedy"],
        help="Decoding method (default: beam).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=50,
        help="Beam width for beam-search decoding (default: 50).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Top-k symbols per step for beam-search decoding (default: 20).",
    )
    parser.add_argument(
        "--examples-count",
        type=int,
        default=3,
        help="How many worst examples to save in CSV for visual inspection (default: 3).",
    )
    parser.add_argument(
        "--top-word-errors",
        type=int,
        default=12,
        help="Top-N word mismatch pairs for console output (default: 12).",
    )
    return parser.parse_args()


def _safe_bool(value, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _safe_int(value, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _parse_splits_arg(raw_splits: str) -> list[str]:
    allowed = {"train", "val", "test"}
    splits: list[str] = []
    for item in str(raw_splits).split(","):
        split = item.strip().lower()
        if not split:
            continue
        if split not in allowed:
            raise ValueError(f"Unknown split: {split}. Allowed: train,val,test")
        if split not in splits:
            splits.append(split)
    if not splits:
        raise ValueError("No valid splits provided.")
    return splits


def _resolve_image_path(path_value: str, csv_path: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate_from_csv = (csv_path.parent / path).resolve()
        if candidate_from_csv.exists():
            return candidate_from_csv
        candidate_from_project = (BASE_DIR / path).resolve()
        if candidate_from_project.exists():
            return candidate_from_project
    return path


def load_preprocess_cfg(checkpoint_path: Path) -> dict[str, int | bool]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    preprocess = {}
    if isinstance(ckpt, dict):
        cfg = ckpt.get("cfg")
        if isinstance(cfg, dict):
            preprocess = cfg.get("preprocess") or {}

    tight_cfg = preprocess.get("tight_crop") if isinstance(preprocess, dict) else {}
    if not isinstance(tight_cfg, dict):
        tight_cfg = {}

    return {
        "height": _safe_int(preprocess.get("height"), 128),
        "keep_aspect": _safe_bool(preprocess.get("keep_aspect"), True),
        "pad_value": _safe_int(preprocess.get("pad_value"), 255),
        "tight_crop_enabled": _safe_bool(tight_cfg.get("enabled"), False),
        "tight_crop_threshold": _safe_int(tight_cfg.get("threshold"), 245),
        "tight_crop_margin": _safe_int(tight_cfg.get("margin"), 2),
    }


def build_dataloader(
    *,
    processed_dir: Path,
    split: str,
    preprocess_cfg: dict[str, int | bool],
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, Path]:
    csv_path = processed_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    transform = make_image_transform(
        height=int(preprocess_cfg["height"]),
        keep_aspect=bool(preprocess_cfg["keep_aspect"]),
        tight_crop_enabled=bool(preprocess_cfg["tight_crop_enabled"]),
        tight_crop_threshold=int(preprocess_cfg["tight_crop_threshold"]),
        tight_crop_margin=int(preprocess_cfg["tight_crop_margin"]),
        augment_cfg=None,
        is_train=False,
        fill=int(preprocess_cfg["pad_value"]),
        to_float_tensor=True,
    )

    dataset = IamLineDataset(
        csv_path=csv_path,
        transform=transform,
        target_height=int(preprocess_cfg["height"]),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        collate_fn=lambda batch: collate_line_batch(batch, pad_value=float(preprocess_cfg["pad_value"]) / 255.0),
    )
    return dataloader, csv_path


def _levenshtein_ops(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, list[dict[str, str]]]:
    """Return distance and one optimal op sequence converting ref -> hyp."""
    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "eq"
                continue

            sub_cost = dp[i - 1][j - 1] + 1
            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1
            best = min(sub_cost, del_cost, ins_cost)
            dp[i][j] = best

            # stable tie-break: substitution -> deletion -> insertion
            if best == sub_cost:
                back[i][j] = "sub"
            elif best == del_cost:
                back[i][j] = "del"
            else:
                back[i][j] = "ins"

    ops: list[dict[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "eq":
            i -= 1
            j -= 1
            continue
        if move == "sub":
            ops.append({"op": "sub", "src": str(ref[i - 1]), "dst": str(hyp[j - 1])})
            i -= 1
            j -= 1
            continue
        if move == "del":
            ops.append({"op": "del", "src": str(ref[i - 1]), "dst": ""})
            i -= 1
            continue
        if move == "ins":
            ops.append({"op": "ins", "src": "", "dst": str(hyp[j - 1])})
            j -= 1
            continue
        raise RuntimeError("Failed to backtrace Levenshtein path.")

    ops.reverse()
    return dp[n][m], ops


def _cer_from_ops(ref_text: str, hyp_text: str) -> tuple[float, int, list[dict[str, str]]]:
    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)
    dist, ops = _levenshtein_ops(ref_chars, hyp_chars)
    denom = len(ref_chars)
    if denom == 0:
        return (0.0 if len(hyp_chars) == 0 else 1.0), dist, ops
    return dist / denom, dist, ops


def _wer_from_ops(ref_text: str, hyp_text: str) -> tuple[float, int, list[dict[str, str]]]:
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    dist, ops = _levenshtein_ops(ref_words, hyp_words)
    denom = len(ref_words)
    if denom == 0:
        return (0.0 if len(hyp_words) == 0 else 1.0), dist, ops
    return dist / denom, dist, ops


def run_inference_table(
    *,
    checkpoint_path: Path,
    dataloader: DataLoader,
    csv_path: Path,
    device_str: str,
    decode_method: str,
    beam_width: int,
    topk: int,
) -> pd.DataFrame:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_hybrid_checkpoint(checkpoint_path, device)
    model.eval()

    rows: list[dict[str, object]] = []
    decode_mode = decode_method.lower().strip()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="hybrid-test-infer"):
            x = batch["pixel_values"].to(device)
            widths = batch["widths"]
            targets = batch["texts"]
            meta = batch["meta"]

            token_lengths = model.token_lengths_from_widths(widths).to(device)
            log_probs = model(x, token_lengths=token_lengths)

            if decode_mode == "beam":
                preds = ctc_beam_search_batch(
                    log_probs,
                    tokenizer,
                    beam_width=int(beam_width),
                    topk=int(topk),
                )
            elif decode_mode == "greedy":
                preds = ctc_greedy_decode_batch(log_probs, tokenizer)
            else:
                raise ValueError(f"Unknown decode method: {decode_mode}")

            for target_text, pred_text, meta_row in zip(targets, preds, meta, strict=True):
                cer_value, char_distance, char_ops = _cer_from_ops(target_text, pred_text)
                wer_value, word_distance, word_ops = _wer_from_ops(target_text, pred_text)
                image_path_raw = str(meta_row.get("image_path", ""))
                image_path = _resolve_image_path(image_path_raw, csv_path)

                rows.append(
                    {
                        "image_path": str(image_path),
                        "target_text": str(target_text),
                        "predicted_text": str(pred_text),
                        "CER": float(cer_value),
                        "WER": float(wer_value),
                        "char_errors": int(char_distance),
                        "word_errors": int(word_distance),
                        "edit_ops": json.dumps(char_ops, ensure_ascii=False),
                        "word_edit_ops": json.dumps(word_ops, ensure_ascii=False),
                    }
                )

    return pd.DataFrame(rows)


def aggregate_error_stats(table: pd.DataFrame) -> dict[str, object]:
    char_subst = Counter()
    char_insert = Counter()
    char_delete = Counter()
    char_ops_count = Counter()

    word_subst = Counter()
    word_insert = Counter()
    word_delete = Counter()
    word_ops_count = Counter()

    for row in table.itertuples(index=False):
        for op in json.loads(row.edit_ops):
            op_type = str(op["op"])
            char_ops_count[op_type] += 1
            if op_type == "sub":
                char_subst[(str(op["src"]), str(op["dst"]))] += 1
            elif op_type == "del":
                char_delete[str(op["src"])] += 1
            elif op_type == "ins":
                char_insert[str(op["dst"])] += 1

        for op in json.loads(row.word_edit_ops):
            op_type = str(op["op"])
            word_ops_count[op_type] += 1
            if op_type == "sub":
                word_subst[(str(op["src"]), str(op["dst"]))] += 1
            elif op_type == "del":
                word_delete[str(op["src"])] += 1
            elif op_type == "ins":
                word_insert[str(op["dst"])] += 1

    return {
        "char_subst": char_subst,
        "char_insert": char_insert,
        "char_delete": char_delete,
        "char_ops_count": char_ops_count,
        "word_subst": word_subst,
        "word_insert": word_insert,
        "word_delete": word_delete,
        "word_ops_count": word_ops_count,
    }


def print_console_summary(table: pd.DataFrame, stats: dict[str, object], top_word_errors: int) -> None:
    n = len(table)
    avg_cer = float(table["CER"].mean()) if n else 0.0
    avg_wer = float(table["WER"].mean()) if n else 0.0
    total_char_errors = int(table["char_errors"].sum()) if n else 0
    total_word_errors = int(table["word_errors"].sum()) if n else 0

    char_ops_count = stats["char_ops_count"]
    char_subst = stats["char_subst"]
    char_delete = stats["char_delete"]
    char_insert = stats["char_insert"]
    word_subst = stats["word_subst"]

    print(f"Rows analyzed: {n}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Total character errors: {total_char_errors}")
    print(f"Total word errors: {total_word_errors}")
    print()

    print("Character-level operations:")
    print(f"  substitutions: {int(char_ops_count.get('sub', 0))}")
    print(f"  deletions: {int(char_ops_count.get('del', 0))}")
    print(f"  insertions: {int(char_ops_count.get('ins', 0))}")
    print()

    print("Top character substitutions (truth -> prediction):")
    if not char_subst:
        print("  <none>")
    else:
        for (src, dst), count in char_subst.most_common(15):
            print(f"  {src} -> {dst}: {count}")
    print()

    print("Most frequent deleted characters:")
    if not char_delete:
        print("  <none>")
    else:
        for ch, count in char_delete.most_common(10):
            print(f"  {ch}: {count}")
    print()

    print("Most frequent inserted characters:")
    if not char_insert:
        print("  <none>")
    else:
        for ch, count in char_insert.most_common(10):
            print(f"  {ch}: {count}")
    print()

    print("Top word mismatches (truth -> prediction):")
    if not word_subst:
        print("  <none>")
    else:
        for (src, dst), count in word_subst.most_common(int(top_word_errors)):
            print(f"  {src} -> {dst}: {count}")


def _pair_counter_to_df(counter: Counter, left_col: str, right_col: str) -> pd.DataFrame:
    rows = [
        {
            left_col: str(left),
            right_col: str(right),
            "count": int(count),
        }
        for (left, right), count in counter.most_common()
    ]
    return pd.DataFrame(rows, columns=[left_col, right_col, "count"])


def _single_counter_to_df(counter: Counter, item_col: str) -> pd.DataFrame:
    rows = [
        {
            item_col: str(item),
            "count": int(count),
        }
        for item, count in counter.most_common()
    ]
    return pd.DataFrame(rows, columns=[item_col, "count"])


def _ops_counter_to_df(counter: Counter) -> pd.DataFrame:
    rows = [
        {"operation": "sub", "count": int(counter.get("sub", 0))},
        {"operation": "del", "count": int(counter.get("del", 0))},
        {"operation": "ins", "count": int(counter.get("ins", 0))},
    ]
    return pd.DataFrame(rows, columns=["operation", "count"])


def export_split_graph_data(
    *,
    split_name: str,
    split_dir: Path,
    table: pd.DataFrame,
    stats: dict[str, object],
    examples_count: int,
) -> dict[str, float | int | str]:
    split_dir.mkdir(parents=True, exist_ok=True)

    table_out = split_dir / f"{split_name}_predictions_with_errors.csv"
    table.to_csv(table_out, index=False)

    char_subst_df = _pair_counter_to_df(stats["char_subst"], "truth_char", "pred_char")
    char_subst_df.to_csv(split_dir / "char_substitutions.csv", index=False)

    char_del_df = _single_counter_to_df(stats["char_delete"], "truth_char")
    char_del_df.to_csv(split_dir / "char_deletions.csv", index=False)

    char_ins_df = _single_counter_to_df(stats["char_insert"], "pred_char")
    char_ins_df.to_csv(split_dir / "char_insertions.csv", index=False)

    char_ops_df = _ops_counter_to_df(stats["char_ops_count"])
    char_ops_df.to_csv(split_dir / "char_operation_counts.csv", index=False)

    word_subst_df = _pair_counter_to_df(stats["word_subst"], "truth_word", "pred_word")
    word_subst_df.to_csv(split_dir / "word_substitutions.csv", index=False)

    word_del_df = _single_counter_to_df(stats["word_delete"], "truth_word")
    word_del_df.to_csv(split_dir / "word_deletions.csv", index=False)

    word_ins_df = _single_counter_to_df(stats["word_insert"], "pred_word")
    word_ins_df.to_csv(split_dir / "word_insertions.csv", index=False)

    word_ops_df = _ops_counter_to_df(stats["word_ops_count"])
    word_ops_df.to_csv(split_dir / "word_operation_counts.csv", index=False)

    worst_examples = (
        table.sort_values(by=["char_errors", "CER", "WER"], ascending=[False, False, False])
        .head(max(1, int(examples_count)))
        .copy()
    )
    worst_examples.to_csv(split_dir / "worst_examples_by_char_errors.csv", index=False)

    summary_row = {
        "split": split_name,
        "rows": int(len(table)),
        "avg_CER": float(table["CER"].mean()) if len(table) else 0.0,
        "avg_WER": float(table["WER"].mean()) if len(table) else 0.0,
        "total_char_errors": int(table["char_errors"].sum()) if len(table) else 0,
        "total_word_errors": int(table["word_errors"].sum()) if len(table) else 0,
        "char_substitutions": int(stats["char_ops_count"].get("sub", 0)),
        "char_deletions": int(stats["char_ops_count"].get("del", 0)),
        "char_insertions": int(stats["char_ops_count"].get("ins", 0)),
        "word_substitutions": int(stats["word_ops_count"].get("sub", 0)),
        "word_deletions": int(stats["word_ops_count"].get("del", 0)),
        "word_insertions": int(stats["word_ops_count"].get("ins", 0)),
        "predictions_csv": str(table_out),
    }
    pd.DataFrame([summary_row]).to_csv(split_dir / "summary_metrics.csv", index=False)
    return summary_row


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    splits = _parse_splits_arg(args.splits)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = load_preprocess_cfg(checkpoint_path)
    all_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for split in splits:
        print()
        print(f"=== Processing split: {split} ===")
        dataloader, csv_path = build_dataloader(
            processed_dir=args.processed_dir.resolve(),
            split=split,
            preprocess_cfg=preprocess_cfg,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )

        table = run_inference_table(
            checkpoint_path=checkpoint_path,
            dataloader=dataloader,
            csv_path=csv_path,
            device_str=str(args.device),
            decode_method=str(args.decode_method),
            beam_width=int(args.beam_width),
            topk=int(args.topk),
        )
        table.insert(0, "split", split)
        all_tables.append(table)

        stats = aggregate_error_stats(table)
        print_console_summary(table, stats, top_word_errors=int(args.top_word_errors))

        split_dir = output_dir / split
        summary_rows.append(
            export_split_graph_data(
                split_name=split,
                split_dir=split_dir,
                table=table,
                stats=stats,
                examples_count=int(args.examples_count),
            )
        )

    if not all_tables:
        raise RuntimeError("No data processed for any split.")

    combined = pd.concat(all_tables, ignore_index=True)
    combined_stats = aggregate_error_stats(combined)
    combined_dir = output_dir / "combined"
    summary_rows.append(
        export_split_graph_data(
            split_name="combined",
            split_dir=combined_dir,
            table=combined,
            stats=combined_stats,
            examples_count=int(args.examples_count),
        )
    )

    combined.to_csv(output_dir / "all_splits_predictions_with_errors.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "all_splits_summary_metrics.csv", index=False)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "splits": splits,
        "processed_dir": str(args.processed_dir.resolve()),
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "decode_method": str(args.decode_method),
        "beam_width": int(args.beam_width),
        "topk": int(args.topk),
        "examples_count": int(args.examples_count),
        "preprocess_cfg_from_checkpoint": preprocess_cfg,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Saved export root: {output_dir}")
    print(f"Saved combined table: {output_dir / 'all_splits_predictions_with_errors.csv'}")
    print(f"Saved split summaries: {output_dir / 'all_splits_summary_metrics.csv'}")
    for split in splits + ["combined"]:
        print(f"Saved split data dir: {output_dir / split}")


if __name__ == "__main__":
    main()
