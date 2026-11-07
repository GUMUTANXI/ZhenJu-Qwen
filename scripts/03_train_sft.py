"""Instruction tuning: LoRA supervised fine-tuning on the four-line diagnosis instruction.

The same script also runs the data-efficiency sweep (--fraction) and the continued-SFT
control arm (--adapter + --train-split pool).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from zhenju.data import read_jsonl  # noqa: E402
from zhenju.qwen import (  # noqa: E402
    JsonlLogger, attach_lora, cosine_lambda, count_parameters, encode_prompt, evaluate_dev, gpu_peak_gb,
    length_grouped_batches, limit_vram, load_base, load_tokenizer, pad_batch, response_loss, sample_target,
    seed_everything, tokenize_pair,
)


def stratified_fraction(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    if fraction >= 1.0:
        return rows
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(";".join(row["types"]) or "clean", []).append(row)
    chosen: list[dict] = []
    for group in buckets.values():
        rng.shuffle(group)
        chosen.extend(group[:max(1, round(len(group) * fraction))])
    return chosen


@torch.no_grad()
def dev_loss(model, encoded: list[dict], pad_id: int, batch_size: int = 32) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(encoded), batch_size):
        batch = pad_batch(encoded[start:start + batch_size], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = response_loss(model, batch)
        total += loss.item() * len(batch["input_ids"])
        count += len(batch["input_ids"])
    model.train()
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--train-split", default="main")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--adapter", default=None, help="continue from an existing adapter")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=3000, help="padded-token budget per batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early; useful for speed tests")
    parser.add_argument("--dev-limit", type=int, default=800)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    limit_vram()
    tokenizer = load_tokenizer(args.base)
    rows = stratified_fraction(read_jsonl(args.data / f"{args.train_split}.jsonl"), args.fraction, args.seed)
    dev = read_jsonl(args.data / "dev.jsonl")

    def encode(items: list[dict]) -> list[dict]:
        out = []
        for item in items:
            pair = tokenize_pair(tokenizer, encode_prompt(tokenizer, item["sentence"]),
                                 sample_target(item), args.max_length)
            if pair is not None:
                out.append(pair)
        return out

    train_encoded = encode(rows)
    dev_encoded = encode(dev)
    print(f"train {len(train_encoded)}/{len(rows)}  dev {len(dev_encoded)}/{len(dev)}", flush=True)

    model = load_base(args.base)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        model = attach_lora(model, args.rank, args.alpha)
    model.config.use_cache = False
    params = count_parameters(model)
    print(params, flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), fused=True)
    rng = random.Random(args.seed)
    steps_per_epoch = len(length_grouped_batches(train_encoded, args.batch_size, random.Random(0), args.max_tokens))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lambda(total_steps, warmup=max(10, int(total_steps * 0.03))))

    logger = JsonlLogger(args.results / "logs" / f"{args.run_name}.jsonl")
    best = {"composite": -1.0}
    step = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    stop = False
    for epoch in range(args.epochs):
        for batch_rows in length_grouped_batches(train_encoded, args.batch_size, rng, args.max_tokens):
            batch = pad_batch(batch_rows, tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = response_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == 1:
                elapsed = time.time() - started
                logger.log(event="train", step=step, epoch=epoch, loss=loss.item(),
                           lr=scheduler.get_last_lr()[0], elapsed=elapsed,
                           allocated_gb=gpu_peak_gb(), reserved_gb=torch.cuda.memory_reserved() / 1024 ** 3)
            if step % 200 == 0:
                print(f"step {step}/{total_steps} loss {loss.item():.4f} "
                      f"{step / (time.time() - started):.2f} it/s peak {gpu_peak_gb():.1f}GB", flush=True)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        torch.cuda.empty_cache()
        model.config.use_cache = True
        val_loss = dev_loss(model, dev_encoded, tokenizer.pad_token_id)
        report = evaluate_dev(model, tokenizer, dev, limit=args.dev_limit)
        model.config.use_cache = False
        torch.cuda.empty_cache()
        logger.log(event="dev", step=step, epoch=epoch, dev_loss=val_loss, composite=report["composite"],
                   detection=report["detection"]["accuracy"], type_macro_f1=report["type"]["macro_f1"],
                   edit_f05=report["correction"]["f0.5"], well_formed=report["well_formed_rate"],
                   elapsed=time.time() - started)
        print(f"epoch {epoch} dev_loss {val_loss:.4f} composite {report['composite']:.4f} "
              f"det {report['detection']['accuracy']:.4f} type {report['type']['macro_f1']:.4f} "
              f"edit {report['correction']['f0.5']:.4f}", flush=True)
        if report["composite"] > best["composite"]:
            best = {"composite": report["composite"], "epoch": epoch, "step": step, "dev_loss": val_loss}
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out, safe_serialization=True)
        if stop:
            break

    summary = {
        "run": args.run_name, "seed": args.seed, "fraction": args.fraction, "n_train": len(train_encoded),
        "params": params, "steps": step, "seconds": time.time() - started, "peak_gpu_gb": gpu_peak_gb(),
        "best": best, "lr": args.lr, "rank": args.rank, "alpha": args.alpha, "batch_size": args.batch_size,
        "epochs": args.epochs,
    }
    logger.log(event="summary", **summary)
    logger.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
