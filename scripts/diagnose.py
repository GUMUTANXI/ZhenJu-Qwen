"""Diagnose sentences from the command line.

    python -I scripts/diagnose.py --base models/Qwen3-1.7B --adapter adapters/zhenju-sft "从这篇文章中，给了我很大的启发。"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from zhenju.qwen import encode_prompt, generate_texts, load_policy, load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Qwen3-1.7B weights")
    parser.add_argument("--adapter", default=str(ROOT / "adapters" / "zhenju-sft"))
    parser.add_argument("sentences", nargs="+")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.base)
    model = load_policy(args.base, args.adapter)
    prompts = [encode_prompt(tokenizer, s) for s in args.sentences]
    for sentence, reply in zip(args.sentences, generate_texts(model, tokenizer, prompts)):
        print(f"句子：{sentence}\n{reply}\n")


if __name__ == "__main__":
    main()
