"""Build leakage-audited splits from FCGEC.

test  : the official validation split (the official test split ships without labels)
dev   : homology-grouped hold-out used only for checkpoint and hyperparameter selection
pool  : homology-grouped hold-out kept away from instruction tuning; it plays the role
        of post-deployment preference data for DPO and PPO
main  : everything else, used for all supervised arms
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from zhenju.data import homology_groups, load_fcgec, ngram_set, write_jsonl  # noqa: E402


def summarise(rows) -> dict:
    types = collections.Counter(code for r in rows for code in r.types)
    return {
        "n": len(rows),
        "faulty": sum(r.error_flag for r in rows),
        "clean": sum(1 - r.error_flag for r in rows),
        "types": dict(types.most_common()),
        "mean_chars": round(sum(len(r.sentence) for r in rows) / max(1, len(rows)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    parser.add_argument("--dev-size", type=int, default=1500)
    parser.add_argument("--pool-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    train, train_audit = load_fcgec(args.raw / "FCGEC_train_filtered.json")
    test, test_audit = load_fcgec(args.raw / "FCGEC_valid.json")

    # The official filter removes direct overlaps; check what homology is left, then drop it.
    test_grams = set().union(*(ngram_set(s.sentence) for s in test))
    survivors = [s for s in train if not (ngram_set(s.sentence) & test_grams)]
    leaked = len(train) - len(survivors)

    groups = homology_groups([s.sentence for s in survivors])
    members: dict[int, list] = collections.defaultdict(list)
    for sample, group in zip(survivors, groups):
        members[group].append(sample)

    # Oversized connected components come from shared boilerplate; they stay in main.
    ordered = list(members.values())
    random.Random(args.seed).shuffle(ordered)
    dev, pool, main_split = [], [], []
    for block in ordered:
        if len(block) > 20:
            main_split.extend(block)
        elif len(dev) + len(block) <= args.dev_size:
            dev.extend(block)
        elif len(pool) + len(block) <= args.pool_size:
            pool.extend(block)
        else:
            main_split.extend(block)

    splits = {"main": main_split, "dev": dev, "pool": pool, "test": test}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        for row in rows:
            row.split = name
        write_jsonl(args.out / f"{name}.jsonl", (r.to_dict() for r in rows))

    report = {
        "source": {"train_filtered": train_audit, "official_valid_as_test": test_audit},
        "homology_audit": {
            "train_items_sharing_8gram_with_test": leaked,
            "removed_from_training": leaked,
            "groups": len(members),
            "largest_group": max(len(v) for v in members.values()),
        },
        "splits": {name: summarise(rows) for name, rows in splits.items()},
        "seed": args.seed,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
