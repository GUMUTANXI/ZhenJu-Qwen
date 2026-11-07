"""Alignment tuning with Direct Preference Optimization (Rafailov et al., 2023).

L = -log sigmoid( beta * [ (log pi(y_c|x) - log pi_ref(y_c|x)) - (log pi(y_r|x) - log pi_ref(y_r|x)) ] )

The log-ratios are logged per step because they expose DPO's known failure mode: the loss
only rewards the *gap*, so the policy can lower the probability of the good answer as long
as the bad answer falls faster.
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
from torch.nn import functional as F  # noqa: E402

from zhenju.data import read_jsonl  # noqa: E402
from zhenju.qwen import (  # noqa: E402
    JsonlLogger, cosine_lambda, count_parameters, encode_prompt, evaluate_dev, gpu_peak_gb, limit_vram,
    load_policy, load_tokenizer, pad_batch, seed_everything, sequence_logprobs, tokenize_pair,
)


def disable_dropout(model) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


@torch.no_grad()
def reference_logps(model, encoded: list[dict], pad_id: int, batch_size: int = 16) -> list[float]:
    model.eval()
    values: list[float] = []
    for start in range(0, len(encoded), batch_size):
        batch = pad_batch(encoded[start:start + batch_size], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values.extend(sequence_logprobs(model, batch).tolist())
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=380)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--dev-limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    limit_vram()
    tokenizer = load_tokenizer(args.base)
    dev = read_jsonl(args.data / "dev.jsonl")
    rows = read_jsonl(args.data / "dpo_pairs.jsonl")

    pairs = []
    for row in rows:
        prompt = encode_prompt(tokenizer, row["sentence"])
        chosen = tokenize_pair(tokenizer, prompt, row["chosen"], args.max_length)
        rejected = tokenize_pair(tokenizer, prompt, row["rejected"], args.max_length)
        if chosen and rejected:
            pairs.append({"chosen": chosen, "rejected": rejected, "kind": row["kind"]})
    print(f"pairs {len(pairs)}/{len(rows)}", flush=True)

    model = load_policy(args.base, args.adapter, trainable=True)
    disable_dropout(model)
    model.config.use_cache = False
    params = count_parameters(model)

    started = time.time()
    chosen_ref = reference_logps(model, [p["chosen"] for p in pairs], tokenizer.pad_token_id)
    rejected_ref = reference_logps(model, [p["rejected"] for p in pairs], tokenizer.pad_token_id)
    for pair, c, r in zip(pairs, chosen_ref, rejected_ref):
        pair["ref_c"], pair["ref_r"] = c, r
    ref_seconds = time.time() - started
    torch.cuda.empty_cache()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, fused=True)
    total_steps = int(len(pairs) * args.epochs / args.batch_pairs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda(total_steps, max(10, total_steps // 30)))

    logger = JsonlLogger(args.results / "logs" / f"{args.run_name}.jsonl")
    rng = random.Random(args.seed)
    order: list[int] = []
    model.train()
    disable_dropout(model)
    torch.cuda.reset_peak_memory_stats()
    best = {"composite": -1.0}
    started = time.time()

    baseline = evaluate_dev(model, tokenizer, dev, limit=args.dev_limit)
    logger.log(event="dev", step=0, composite=baseline["composite"], detection=baseline["detection"]["accuracy"],
               type_macro_f1=baseline["type"]["macro_f1"], edit_f05=baseline["correction"]["f0.5"],
               well_formed=baseline["well_formed_rate"], elapsed=0.0)
    print(f"step 0 composite {baseline['composite']:.4f}", flush=True)

    for step in range(1, total_steps + 1):
        if len(order) < args.batch_pairs:
            order = list(range(len(pairs)))
            rng.shuffle(order)
        picks = [pairs[order.pop()] for _ in range(args.batch_pairs)]
        rows_ = [p["chosen"] for p in picks] + [p["rejected"] for p in picks]
        batch = pad_batch(rows_, tokenizer.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logps = sequence_logprobs(model, batch)
        n = len(picks)
        ref_c = torch.tensor([p["ref_c"] for p in picks], device=logps.device, dtype=logps.dtype)
        ref_r = torch.tensor([p["ref_r"] for p in picks], device=logps.device, dtype=logps.dtype)
        ratio_c = logps[:n] - ref_c
        ratio_r = logps[n:] - ref_r
        margin = ratio_c - ratio_r
        loss = -F.logsigmoid(args.beta * margin).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == 1:
            logger.log(event="train", step=step, loss=loss.item(), chosen_logratio=ratio_c.mean().item(),
                       rejected_logratio=ratio_r.mean().item(), margin=margin.mean().item(),
                       pair_accuracy=(margin > 0).float().mean().item(), grad_norm=float(grad_norm),
                       chosen_logp=logps[:n].mean().item(), elapsed=time.time() - started)
        if step % args.eval_every == 0 or step == total_steps:
            torch.cuda.empty_cache()
            model.config.use_cache = True
            report = evaluate_dev(model, tokenizer, dev, limit=args.dev_limit)
            model.config.use_cache = False
            torch.cuda.empty_cache()
            model.train()
            disable_dropout(model)
            logger.log(event="dev", step=step, composite=report["composite"],
                       detection=report["detection"]["accuracy"], type_macro_f1=report["type"]["macro_f1"],
                       edit_f05=report["correction"]["f0.5"], well_formed=report["well_formed_rate"],
                       elapsed=time.time() - started)
            print(f"step {step}/{total_steps} loss {loss.item():.4f} chosen {ratio_c.mean().item():+.2f} "
                  f"rejected {ratio_r.mean().item():+.2f} composite {report['composite']:.4f} "
                  f"det {report['detection']['accuracy']:.4f} wf {report['well_formed_rate']:.3f}", flush=True)
            if report["composite"] > best["composite"]:
                best = {"composite": report["composite"], "step": step}
                args.out.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(args.out, safe_serialization=True)

    summary = {"run": args.run_name, "beta": args.beta, "lr": args.lr, "pairs": len(pairs), "steps": total_steps,
               "seconds": time.time() - started, "reference_seconds": ref_seconds, "peak_gpu_gb": gpu_peak_gb(),
               "baseline_composite": baseline["composite"], "best": best, "params": params,
               "sequences_per_step": args.batch_pairs * 2}
    logger.log(event="summary", **summary)
    logger.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
