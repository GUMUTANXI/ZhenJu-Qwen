"""Scoring for the three FCGEC tiers: detection, error-type identification and correction.

Correction uses exact match plus a character-level span-edit F0.5 in the spirit of
ChERRANT: edits come from a minimal Levenshtein alignment after punctuation removal,
consecutive edits merge into spans, and each sentence is scored against its best
reference. ChERRANT itself adds semantic and phonetic costs to the alignment, so these
numbers are comparable between arms here but not directly with published CGEC tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .data import TYPE_CODES, strip_punct


def binary_metrics(gold: list[int], pred: list[int]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(gold, pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
    }


def _type_matrix(rows: list[list[str]]) -> np.ndarray:
    index = {code: i for i, code in enumerate(TYPE_CODES)}
    matrix = np.zeros((len(rows), len(TYPE_CODES)), dtype=int)
    for r, codes in enumerate(rows):
        for code in codes:
            matrix[r, index[code]] = 1
    return matrix


def type_metrics(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """Multi-label type scores on sentences that are truly faulty."""
    keep = [i for i, g in enumerate(gold) if g]
    y_true = _type_matrix([gold[i] for i in keep])
    y_pred = _type_matrix([pred[i] for i in keep])
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": float(per_class.mean()),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "per_class_f1": {code: float(v) for code, v in zip(TYPE_CODES, per_class)},
    }


def edit_spans(source: str, target: str) -> set[tuple[int, int, str]]:
    spans: list[list] = []
    for op in Levenshtein.opcodes(source, target):
        if op.tag == "equal":
            continue
        text = target[op.dest_start:op.dest_end]
        if spans and spans[-1][1] == op.src_start:
            spans[-1][1] = op.src_end
            spans[-1][2] += text
        else:
            spans.append([op.src_start, op.src_end, text])
    return {(a, b, c) for a, b, c in spans}


@dataclass
class EditCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def scores(self, beta: float = 0.5) -> dict:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        b2 = beta * beta
        denominator = b2 * precision + recall
        fbeta = (1 + b2) * precision * recall / denominator if denominator else 0.0
        return {"precision": precision, "recall": recall, "f0.5": fbeta}


def best_reference_counts(source: str, hypothesis: str, references: list[str]) -> tuple[int, int, int]:
    """Counts (tp, fp, fn) against whichever reference gives this sentence the best F0.5."""
    src = strip_punct(source)
    hyp_edits = edit_spans(src, strip_punct(hypothesis))
    best: tuple[float, int, int, int] | None = None
    for reference in references:
        ref_edits = edit_spans(src, strip_punct(reference))
        tp = len(hyp_edits & ref_edits)
        fp = len(hyp_edits - ref_edits)
        fn = len(ref_edits - hyp_edits)
        score = EditCounts(tp, fp, fn).scores()["f0.5"]
        key = (score, tp, -fp, -fn)
        if best is None or key > best:
            best = key
    assert best is not None
    return best[1], -best[2], -best[3]


def is_exact(hypothesis: str, references: list[str]) -> int:
    clean = strip_punct(hypothesis)
    return int(any(clean == strip_punct(reference) for reference in references))


def correction_metrics(samples: list[dict], hypotheses: list[str]) -> dict:
    totals = EditCounts()
    exact_error: list[int] = []
    exact_clean: list[int] = []
    for sample, hypothesis in zip(samples, hypotheses):
        tp, fp, fn = best_reference_counts(sample["sentence"], hypothesis, sample["references"])
        totals.tp += tp
        totals.fp += fp
        totals.fn += fn
        hit = is_exact(hypothesis, sample["references"])
        (exact_error if sample["error_flag"] else exact_clean).append(hit)
    out = totals.scores()
    out.update({
        "exact_error_sentences": float(np.mean(exact_error)) if exact_error else 0.0,
        "keep_clean_sentences": float(np.mean(exact_clean)) if exact_clean else 0.0,
        "tp": totals.tp,
        "fp": totals.fp,
        "fn": totals.fn,
    })
    return out


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact McNemar test on paired per-sentence outcomes."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = float(stats.binomtest(min(only_a, only_b), n, 0.5).pvalue) if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def score_predictions(samples: list[dict], predictions: list[dict]) -> dict:
    """Score one arm; each prediction holds flag, types and correction."""
    gold_flags = [s["error_flag"] for s in samples]
    pred_flags = [p["flag"] for p in predictions]
    report = {
        "n": len(samples),
        "detection": binary_metrics(gold_flags, pred_flags),
        "type": type_metrics([s["types"] for s in samples], [p["types"] for p in predictions]),
        "correction": correction_metrics(samples, [p["correction"] for p in predictions]),
    }
    well = [p.get("well_formed", True) for p in predictions]
    report["well_formed_rate"] = float(np.mean(well))
    return report
