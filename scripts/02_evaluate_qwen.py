"""Evaluate one Qwen arm on a split: pretrained zero-shot / few-shot / thinking, or a LoRA adapter."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

from zhenju.data import ERROR_TYPES, read_jsonl, write_jsonl  # noqa: E402
from zhenju.metrics import score_predictions  # noqa: E402
from zhenju.qwen import (  # noqa: E402
    diagnose_samples, gpu_peak_gb, limit_vram, load_policy, load_tokenizer, seed_everything,
)


def pick_shots(pool: list[dict], seed: int = 7) -> list[dict]:
    """One short exemplar for each error type plus one clean sentence."""
    rng = random.Random(seed)
    short = [s for s in pool if len(s["sentence"]) <= 45]
    rng.shuffle(short)
    shots = [next(s for s in short if s["types"] == [code]) for code in ERROR_TYPES]
    shots.append(next(s for s in short if s["error_flag"] == 0))
    return shots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", required=True, type=Path, help="directory holding the split jsonl files")
    parser.add_argument("--split", default="test")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--shots", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    args = parser.parse_args()

    seed_everything(0)
    limit_vram()
    samples = read_jsonl(args.data / f"{args.split}.jsonl")
    if args.limit:
        samples = samples[:args.limit]
    shots = pick_shots(read_jsonl(args.data / "main.jsonl")) if args.shots else None

    tokenizer = load_tokenizer(args.base)
    model = load_policy(args.base, args.adapter)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    predictions, timing = diagnose_samples(
        model, tokenizer, samples, shots=shots, thinking=args.thinking,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
    )
    report = score_predictions(samples, predictions)
    report.update(arm=args.arm, split=args.split, timing=timing, peak_gpu_gb=gpu_peak_gb())

    write_jsonl(args.results / "predictions" / f"{args.arm}.{args.split}.jsonl", predictions)
    metrics_dir = args.results / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{args.arm}.{args.split}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("arm", "detection", "well_formed_rate")}, ensure_ascii=False))
    print("type", {k: round(v, 4) for k, v in report["type"].items() if k != "per_class_f1"})
    print("correction", {k: round(v, 4) if isinstance(v, float) else v for k, v in report["correction"].items()})
    print("timing", timing, "peak GB", round(report["peak_gpu_gb"], 2))
    for prediction in predictions[:3]:
        print(repr(prediction["raw"][:300]))


if __name__ == "__main__":
    main()
