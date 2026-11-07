"""Mine preference pairs for alignment tuning.

The instruction-tuned model answers sentences from the held-out pool, which it never saw
in training, so its mistakes are genuine generalisation errors rather than noise.
chosen = the gold answer, rejected = a wrong answer the model actually produced.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

from zhenju.data import read_jsonl, write_jsonl  # noqa: E402
from zhenju.metrics import is_exact  # noqa: E402
from zhenju.prompts import parse_response  # noqa: E402
from zhenju.qwen import (  # noqa: E402
    encode_prompt, generate_texts, limit_vram, load_policy, load_tokenizer, sample_target, seed_everything,
)


def mistake_kind(sample: dict, text: str) -> str | None:
    """Label a reply's first error, or None if it is correct."""
    parsed = parse_response(text, sample["sentence"])
    if not parsed.well_formed:
        return "malformed"
    if parsed.flag != sample["error_flag"]:
        return "false_alarm" if parsed.flag else "missed_error"
    if sample["error_flag"] and set(parsed.types) != set(sample["types"]):
        return "wrong_type"
    if not is_exact(parsed.correction, sample["references"]):
        return "bad_correction"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=2000, help="pool sentences to mine")
    parser.add_argument("--samples", type=int, default=1, help="extra sampled replies per sentence")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-per-sentence", type=int, default=2)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    args = parser.parse_args()

    seed_everything(123)
    limit_vram()
    tokenizer = load_tokenizer(args.base)
    model = load_policy(args.base, args.adapter)
    model.eval()
    pool = read_jsonl(args.data / "pool.jsonl")[:args.limit]
    prompts =[encode_prompt(tokenizer, s["sentence"]) for s in pool]

    candidates: list[list[str]] = [[] for _ in pool]
    for i, text in enumerate(generate_texts(model, tokenizer, prompts)):
        candidates[i].append(text)
    for _ in range(args.samples):
        sampled = generate_texts(model, tokenizer, prompts, do_sample=True, temperature=args.temperature)
        for i, text in enumerate(sampled):
            candidates[i].append(text)
        torch.cuda.empty_cache()

    pairs = []
    kinds: collections.Counter = collections.Counter()
    greedy_correct = 0
    for sample, replies in zip(pool, candidates):
        greedy_correct += int(mistake_kind(sample, replies[0]) is None)
        taken: set[str] = set()
        for reply in replies:
            kind = mistake_kind(sample, reply)
            if kind is None or reply in taken or len(taken) >= args.max_per_sentence:
                continue
            taken.add(reply)
            kinds[kind] += 1
            pairs.append({"uid": sample["uid"], "sentence": sample["sentence"], "chosen": sample_target(sample),
                          "rejected": reply.strip(), "kind": kind})

    write_jsonl(args.data / "dpo_pairs.jsonl", pairs)
    report = {"pool": len(pool), "pairs": len(pairs), "sentences_with_pair": len({p["uid"] for p in pairs}),
              "greedy_fully_correct": greedy_correct / len(pool), "kinds": dict(kinds.most_common())}
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "dpo_pairs_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
