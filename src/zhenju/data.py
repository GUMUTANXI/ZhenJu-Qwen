"""FCGEC parsing: edit-operation replay, edit descriptions and homology-aware splitting.

FCGEC stores corrections as edit operations rather than target sentences, so the
references have to be reconstructed. The rules below follow the corpus README:

* ``Switch`` is a permutation of the source characters and is applied first.
* ``Delete`` / ``Insert`` / ``Modify`` positions index the (switched) sentence.
* ``Insert`` places text *after* ``pos``; ``pos == -1`` means sentence-initial.
* ``Modify`` with tag ``MOD_n`` replaces ``n`` characters starting at ``pos``.
* A list-valued ``label`` holds alternative fillers, which multiply out into
  several acceptable references.

Each entry in ``operation`` is an independent reference applied to the original
sentence, so a ``Switch`` in one reference must not leak into the next.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

ERROR_TYPES: dict[str, str] = {
    "IWO": "语序不当",
    "IWC": "搭配不当",
    "CM": "成分残缺",
    "CR": "成分赘余",
    "SC": "结构混乱",
    "ILL": "不合逻辑",
    "AM": "歧义",
}
TYPE_CODES: list[str] = list(ERROR_TYPES)
ZH_TO_CODE: dict[str, str] = {v: k for k, v in ERROR_TYPES.items()}
MAX_ALTERNATIVES = 16

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def strip_punct(text: str) -> str:
    """Remove whitespace and punctuation, as the FCGEC scorer requires before matching."""
    return _PUNCT.sub("", text)


def _as_list(label) -> list[str]:
    return [label] if isinstance(label, str) else list(label)


def _modify_span(tag: str) -> int:
    match = re.match(r"MOD_(\d+)", tag)
    return int(match.group(1)) if match else 1


def apply_reference(sentence: str, op: dict) -> list[str]:
    """Replay one reference; returns every sentence its alternative labels allow."""
    chars = list(sentence)
    if "Switch" in op:
        perm = op["Switch"]
        if sorted(perm) != list(range(len(chars))):
            raise ValueError(f"Switch is not a permutation of {len(chars)} characters")
        chars = [chars[i] for i in perm]

    deleted = set(op.get("Delete", []))
    slots: list[tuple[str, int, int, list[str]]] = []
    for item in op.get("Insert", []):
        slots.append(("ins", int(item["pos"]), 0, _as_list(item["label"])))
    for item in op.get("Modify", []):
        slots.append(("mod", int(item["pos"]), _modify_span(item["tag"]), _as_list(item["label"])))

    outputs: list[str] = []
    for choice in itertools.islice(itertools.product(*(s[3] for s in slots)), MAX_ALTERNATIVES):
        before: dict[int, str] = {}
        after: dict[int, str] = {}
        removed = set(deleted)
        for (kind, pos, span, _), filler in zip(slots, choice):
            if kind == "ins":
                after[pos] = after.get(pos, "") + filler
            else:
                before[pos] = before.get(pos, "") + filler
                removed.update(range(pos, pos + span))
        pieces = [after.get(-1, "")]
        for i, ch in enumerate(chars):
            pieces.append(before.get(i, ""))
            if i not in removed:
                pieces.append(ch)
            pieces.append(after.get(i, ""))
        outputs.append("".join(pieces))
    return outputs


def references_from_operations(sentence: str, operation: str | list) -> list[str]:
    ops = json.loads(operation) if isinstance(operation, str) else operation
    refs: list[str] = []
    for op in ops:
        for candidate in apply_reference(sentence, op):
            if candidate not in refs:
                refs.append(candidate)
    return refs


def describe_edit(source: str, target: str, op: dict | None = None) -> str:
    """Render a source→target change as the teacher-style note the model learns to write."""
    from rapidfuzz.distance import Levenshtein

    if source == target:
        return "无"
    if op is None or "Switch" not in op:
        spans = [o for o in Levenshtein.opcodes(source, target) if o.tag != "equal"]
        merged: list[list[int]] = []
        for o in spans:
            if merged and o.src_start - merged[-1][1] <= 1 and o.dest_start - merged[-1][3] <= 1:
                merged[-1][1], merged[-1][3] = o.src_end, o.dest_end
            else:
                merged.append([o.src_start, o.src_end, o.dest_start, o.dest_end])
        widest = max((m[1] - m[0]) + (m[3] - m[2]) for m in merged) if merged else 0
        if len(merged) <= 3 and widest <= 24:
            notes = []
            for s0, s1, d0, d1 in merged:
                old, new = source[s0:s1], target[d0:d1]
                if old and new:
                    notes.append(f"把“{old}”改为“{new}”")
                elif old:
                    notes.append(f"删去“{old}”")
                else:
                    anchor = source[max(0, s0 - 2):s0]
                    notes.append(f"在“{anchor}”后补上“{new}”" if anchor else f"在句首补上“{new}”")
            return "；".join(notes)
    # Word-order changes and widely scattered edits read better as one rewritten span.
    start = 0
    while start < min(len(source), len(target)) and source[start] == target[start]:
        start += 1
    end = 0
    while end < min(len(source), len(target)) - start and source[-1 - end] == target[-1 - end]:
        end += 1
    old = source[start:len(source) - end]
    new = target[start:len(target) - end]
    return f"把“{old}”调整为“{new}”"


@dataclass
class Sample:
    uid: str
    sentence: str
    error_flag: int
    types: list[str]
    references: list[str]
    edit_note: str
    split: str = ""

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "sentence": self.sentence,
            "error_flag": self.error_flag,
            "types": self.types,
            "references": self.references,
            "edit_note": self.edit_note,
            "split": self.split,
        }


def parse_types(raw: str) -> list[str]:
    if raw == "*":
        return []
    seen: list[str] = []
    for code in raw.split(";"):
        if code in ERROR_TYPES and code not in seen:
            seen.append(code)
    return seen


def load_fcgec(path: str | Path) -> tuple[list[Sample], dict]:
    """Load one labelled FCGEC split, tracking samples dropped as unusable."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    samples: list[Sample] = []
    audit = {"total": len(raw), "kept": 0, "dropped_no_reference": 0, "dropped_bad_op": 0,
             "flag0_with_type": 0, "multi_reference": 0}
    for uid, row in raw.items():
        sentence = row["sentence"]
        flag = int(row["error_flag"])
        types = parse_types(row["error_type"])
        if flag == 0:
            if types:
                audit["flag0_with_type"] += 1
            samples.append(Sample(uid, sentence, 0, [], [sentence], "无"))
            audit["kept"] += 1
            continue
        try:
            ops = json.loads(row["operation"]) if isinstance(row["operation"], str) else row["operation"]
            refs = [r for r in references_from_operations(sentence, ops) if r != sentence]
        except (ValueError, IndexError, KeyError, TypeError):
            audit["dropped_bad_op"] += 1
            continue
        if not refs or not types:
            audit["dropped_no_reference"] += 1
            continue
        audit["multi_reference"] += int(len(refs) > 1)
        first_op = next((op for op in ops if refs[0] in apply_reference(sentence, op)), None)
        samples.append(Sample(uid, sentence, 1, types, refs, describe_edit(sentence, refs[0], first_op)))
        audit["kept"] += 1
    return samples, audit


def homology_groups(sentences: list[str], k: int = 8) -> list[int]:
    """Union sentences sharing any punctuation-free character k-gram.

    FCGEC derives several items from one exam sentence (for instance a faulty
    version and its repaired twin), so a random split would leak near-copies.
    """
    parent = list(range(len(sentences)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, int] = {}
    for idx, sentence in enumerate(sentences):
        clean = strip_punct(sentence)
        grams = {clean[i:i + k] for i in range(max(1, len(clean) - k + 1))}
        for gram in grams:
            if gram in owner:
                parent[find(idx)] = find(owner[gram])
            else:
                owner[gram] = idx
    return [find(i) for i in range(len(sentences))]


def ngram_set(sentence: str, k: int = 8) -> set[str]:
    clean = strip_punct(sentence)
    return {clean[i:i + k] for i in range(max(1, len(clean) - k + 1))}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
