"""Collect every arm's test predictions into tidy tables with bootstrap CIs and paired tests.

Writes results/summary.csv, results/per_type_f1.csv and results/paired_tests.csv. Figures and
the README read these files, so every number shown can be traced back to one row here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from zhenju.data import TYPE_CODES, read_jsonl  # noqa: E402
from zhenju.metrics import EditCounts, best_reference_counts, is_exact, mcnemar_exact  # noqa: E402

N_BOOT = 2000


def macro_f1_from_matrix(gold: np.ndarray, pred: np.ndarray) -> float:
    tp = (gold & pred).sum(0).astype(float)
    denom = gold.sum(0) + pred.sum(0)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def type_matrix(rows: list[list[str]]) -> np.ndarray:
    index = {c: i for i, c in enumerate(TYPE_CODES)}
    out = np.zeros((len(rows), len(TYPE_CODES)), dtype=bool)
    for r, codes in enumerate(rows):
        for code in codes:
            out[r, index[code]] = True
    return out


class Arm:
    """Per-sentence outcomes for one arm, ready for resampling."""

    def __init__(self, name: str, samples: list[dict], flags=None, types=None, corrections=None):
        self.name = name
        self.samples = samples
        gold = np.array([s["error_flag"] for s in samples])
        self.flags = None if flags is None else np.asarray(flags)
        self.correct = None if flags is None else (self.flags == gold)
        self.gold_types = type_matrix([s["types"] for s in samples if s["error_flag"]])
        self.pred_types = None if types is None else type_matrix(types)
        self.edit = None
        self.exact = None
        if corrections is not None:
            counts = [best_reference_counts(s["sentence"], c, s["references"]) for s, c in zip(samples, corrections)]
            self.edit = np.array(counts)
            self.exact = np.array([is_exact(c, s["references"]) for s, c in zip(samples, corrections)])
            self.faulty_mask = gold == 1

    def type_macro(self, rows=None) -> float:
        rows = slice(None) if rows is None else rows
        return macro_f1_from_matrix(self.gold_types[rows], self.pred_types[rows])

    def edit_f05(self, rows=None) -> float:
        rows = slice(None) if rows is None else rows
        tp, fp, fn = self.edit[rows].sum(0)
        return EditCounts(int(tp), int(fp), int(fn)).scores()["f0.5"]


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_arm(arm: Arm, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(arm.samples)
    n_faulty = len(arm.gold_types)
    out: dict = {}
    if arm.correct is not None:
        picks = rng.integers(0, n, size=(N_BOOT, n))
        out["detection_ci"] = ci(arm.correct[picks].mean(1))
    if arm.pred_types is not None:
        picks = rng.integers(0, n_faulty, size=(N_BOOT, n_faulty))
        out["type_ci"] = ci(np.array([arm.type_macro(row) for row in picks]))
    if arm.edit is not None:
        picks = rng.integers(0, n, size=(N_BOOT, n))
        out["edit_ci"] = ci(np.array([arm.edit_f05(row) for row in picks]))
    return out


def paired_diff(a: Arm, b: Arm, metric: str, seed: int = 1) -> dict:
    """Bootstrap the paired difference a - b by resampling the same sentences for both arms."""
    rng = np.random.default_rng(seed)
    if metric == "detection":
        n = len(a.samples)
        picks = rng.integers(0, n, size=(N_BOOT, n))
        diffs = a.correct[picks].mean(1) - b.correct[picks].mean(1)
        point = float(a.correct.mean() - b.correct.mean())
    elif metric == "type":
        n = len(a.gold_types)
        picks = rng.integers(0, n, size=(N_BOOT, n))
        diffs = np.array([a.type_macro(r) - b.type_macro(r) for r in picks])
        point = a.type_macro() - b.type_macro()
    else:
        n = len(a.samples)
        picks = rng.integers(0, n, size=(N_BOOT, n))
        diffs = np.array([a.edit_f05(r) - b.edit_f05(r) for r in picks])
        point = a.edit_f05() - b.edit_f05()
    low, high = ci(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": point, "ci_low": low, "ci_high": high, "p_boot": float(max(p, 1 / N_BOOT))}


def load_qwen_arm(name: str, samples: list[dict], results: Path) -> Arm | None:
    path = results / "predictions" / f"{name}.test.jsonl"
    if not path.exists():
        return None
    predictions = read_jsonl(path)
    flags = [p["flag"] for p in predictions]
    types = [p["types"] for p, s in zip(predictions, samples) if s["error_flag"]]
    arm = Arm(name, samples, flags, types, [p["correction"] for p in predictions])
    arm.well_formed = float(np.mean([p["well_formed"] for p in predictions]))
    return arm


def load_bert_arm(strategy: str, seed: int, samples: list[dict], results: Path) -> Arm | None:
    detect = results / "predictions" / f"bert_{strategy}_detect_s{seed}.test.jsonl"
    typed = results / "predictions" / f"bert_{strategy}_type_s{seed}.test.jsonl"
    flags = types = None
    if detect.exists():
        flags = (np.array([r["prob"] for r in read_jsonl(detect)]) >= 0.5).astype(int)
    if typed.exists():
        probs = np.array([r["prob"] for r in read_jsonl(typed)])
        types = [[TYPE_CODES[j] for j in range(len(TYPE_CODES)) if p[j] >= 0.5] or [TYPE_CODES[int(np.argmax(p))]]
                 for p in probs]
    if flags is None and types is None:
        return None
    arm = Arm(f"bert_{strategy}_s{seed}", samples, flags, types, None)
    arm.well_formed = None
    return arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    args = parser.parse_args()

    samples = read_jsonl(args.data / "test.jsonl")
    qwen_names = sorted(p.name.split(".test.")[0] for p in (args.results / "metrics").glob("*.test.json")
                        if not p.name.startswith(("bert_", "_")))
    arms: dict[str, Arm] = {}
    for name in qwen_names:
        arm = load_qwen_arm(name, samples, args.results)
        if arm:
            arms[name] = arm
    for strategy in ("head", "top2", "full", "layerwise"):
        arm = load_bert_arm(strategy, 42, samples, args.results)
        if arm:
            arms[arm.name] = arm

    rows, type_rows = [], []
    for name, arm in arms.items():
        boots = bootstrap_arm(arm)
        row: dict = {"arm": name}
        if arm.correct is not None:
            gold = np.array([s["error_flag"] for s in samples])
            tp = int(((arm.flags == 1) & (gold == 1)).sum())
            fp = int(((arm.flags == 1) & (gold == 0)).sum())
            fn = int(((arm.flags == 0) & (gold == 1)).sum())
            row.update(detection=float(arm.correct.mean()), detection_low=boots["detection_ci"][0],
                       detection_high=boots["detection_ci"][1],
                       precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0,
                       false_alarm=fp / max(1, int((gold == 0).sum())))
        if arm.pred_types is not None:
            row.update(type_macro_f1=arm.type_macro(), type_low=boots["type_ci"][0], type_high=boots["type_ci"][1])
            f1s: dict = {"arm": name}
            for j, code in enumerate(TYPE_CODES):
                tp = int((arm.gold_types[:, j] & arm.pred_types[:, j]).sum())
                denom = int(arm.gold_types[:, j].sum() + arm.pred_types[:, j].sum())
                f1s[code] = 2 * tp / denom if denom else 0.0
            type_rows.append(f1s)
        if arm.edit is not None:
            scores = EditCounts(*[int(v) for v in arm.edit.sum(0)]).scores()
            row.update(edit_f05=scores["f0.5"], edit_precision=scores["precision"], edit_recall=scores["recall"],
                       edit_low=boots["edit_ci"][0], edit_high=boots["edit_ci"][1],
                       exact_faulty=float(arm.exact[arm.faulty_mask].mean()),
                       keep_clean=float(arm.exact[~arm.faulty_mask].mean()))
        if arm.well_formed is not None:
            row["well_formed"] = arm.well_formed
        rows.append(row)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "arm", k))
    with open(args.results / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(args.results / "per_type_f1.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["arm", *TYPE_CODES])
        writer.writeheader()
        writer.writerows(type_rows)

    contrasts = [
        ("sft_s42", "pretrained_zero"), ("sft_s42", "pretrained_few"), ("sft_s42", "bert_full_s42"),
        ("bert_full_s42", "bert_head_s42"), ("dpo_beta0.1", "sft_s42"),
        ("ppo_rubric", "sft_s42"), ("ppo_naive", "sft_s42"),
    ]
    tests = []
    for a_name, b_name in contrasts:
        a, b = arms.get(a_name), arms.get(b_name)
        if a is None or b is None:
            continue
        row = {"a": a_name, "b": b_name}
        if a.correct is not None and b.correct is not None:
            row.update({f"detection_{k}": v for k, v in paired_diff(a, b, "detection").items()})
            row.update({f"mcnemar_{k}": v for k, v in mcnemar_exact(a.correct, b.correct).items()})
        if a.pred_types is not None and b.pred_types is not None:
            row.update({f"type_{k}": v for k, v in paired_diff(a, b, "type").items()})
        if a.edit is not None and b.edit is not None:
            row.update({f"edit_{k}": v for k, v in paired_diff(a, b, "edit").items()})
        tests.append(row)
    fields = sorted({k for r in tests for k in r}, key=lambda k: (k not in ("a", "b"), k))
    with open(args.results / "paired_tests.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tests)

    for row in rows:
        print(row["arm"], {k: round(row[k], 3) for k in ("detection", "type_macro_f1", "edit_f05") if k in row})


if __name__ == "__main__":
    main()
