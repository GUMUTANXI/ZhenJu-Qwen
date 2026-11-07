"""BERT baseline, reproducing the architecture of Wang & Lu (2022).

Network: BertModel -> BatchNorm + Dropout -> Linear(768, 200) -> Sigmoid -> Linear(200, n).

Strategies compared:
    head       encoder frozen, only the task head trains
    top2       top two encoder layers unfrozen
    full       every layer fine-tuned from the first epoch
    layerwise  layers unfrozen top-down across epochs (the paper's 逐层解冻)

Tasks: ``detect`` (binary, softmax) and ``type`` (seven-way multi-label, sigmoid per class,
trained on faulty sentences only). The paper targets a binary task, so the multi-label
output is our adaptation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoTokenizer, BertModel  # noqa: E402

from zhenju.data import TYPE_CODES, read_jsonl, write_jsonl  # noqa: E402
from zhenju.metrics import binary_metrics, type_metrics  # noqa: E402
from zhenju.qwen import JsonlLogger, cosine_lambda, gpu_peak_gb, limit_vram, seed_everything  # noqa: E402

DEVICE = "cuda"
N_LAYERS = 12


class RumorStyleClassifier(nn.Module):
    def __init__(self, path: str, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(path)
        self.norm = nn.BatchNorm1d(768)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(768, 200)
        self.act = nn.Sigmoid()
        self.out = nn.Linear(200, n_out)

    def encode(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        cls = self.encode(input_ids, attention_mask)
        return self.out(self.act(self.proj(self.drop(self.norm(cls)))))


def set_unfrozen_layers(model: RumorStyleClassifier, k: int) -> None:
    """Unfreeze the top k encoder layers; k == N_LAYERS also releases the embeddings."""
    for p in model.bert.parameters():
        p.requires_grad = False
    for layer in model.bert.encoder.layer[N_LAYERS - k:] if k else []:
        for p in layer.parameters():
            p.requires_grad = True
    if k >= N_LAYERS:
        for p in model.bert.parameters():
            p.requires_grad = True


def unfreeze_plan(strategy: str, epochs: int) -> list[int]:
    if strategy == "head":
        return [0] * epochs
    if strategy == "top2":
        return [2] * epochs
    if strategy == "full":
        return [N_LAYERS] * epochs
    ladder = [0, 2, 4, 8, N_LAYERS]
    return [ladder[min(i, len(ladder) - 1)] for i in range(epochs)]


def encode_rows(tokenizer, rows: list[dict], task: str, max_length: int) -> list[dict]:
    index = {code: i for i, code in enumerate(TYPE_CODES)}
    encoded = []
    for row in rows:
        ids = tokenizer(row["sentence"], truncation=True, max_length=max_length)["input_ids"]
        if task == "detect":
            label = row["error_flag"]
        else:
            label = [0.0] * len(TYPE_CODES)
            for code in row["types"]:
                label[index[code]] = 1.0
        encoded.append({"ids": ids, "label": label})
    return encoded


def batches(encoded: list[dict], batch_size: int, pad_id: int, task: str, order=None):
    order = order if order is not None else range(len(encoded))
    order = list(order)
    for start in range(0, len(order), batch_size):
        rows = [encoded[i] for i in order[start:start + batch_size]]
        longest = max(len(r["ids"]) for r in rows)
        input_ids = torch.tensor([r["ids"] + [pad_id] * (longest - len(r["ids"])) for r in rows], device=DEVICE)
        mask = torch.tensor([[1] * len(r["ids"]) + [0] * (longest - len(r["ids"])) for r in rows], device=DEVICE)
        dtype = torch.long if task == "detect" else torch.float32
        labels = torch.tensor([r["label"] for r in rows], dtype=dtype, device=DEVICE)
        yield input_ids, mask, labels


@torch.inference_mode()
def predict(model, encoded, pad_id, task, batch_size=128) -> np.ndarray:
    model.eval()
    chunks = []
    for input_ids, mask, _ in batches(encoded, batch_size, pad_id, task):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, mask)
        probs = torch.softmax(logits.float(), -1)[:, 1] if task == "detect" else torch.sigmoid(logits.float())
        chunks.append(probs.cpu().numpy())
    return np.concatenate(chunks)


@torch.inference_mode()
def cls_vectors(model, encoded, pad_id, task, batch_size=128) -> np.ndarray:
    model.eval()
    chunks = []
    for input_ids, mask, _ in batches(encoded, batch_size, pad_id, task):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            chunks.append(model.encode(input_ids, mask).float().cpu().numpy())
    return np.concatenate(chunks)


def score(task: str, rows: list[dict], probs: np.ndarray) -> dict:
    if task == "detect":
        report = binary_metrics([r["error_flag"] for r in rows], (probs >= 0.5).astype(int).tolist())
        report["select"] = report["accuracy"]
    else:
        predicted = [[TYPE_CODES[j] for j in range(len(TYPE_CODES)) if p[j] >= 0.5] or
                     [TYPE_CODES[int(np.argmax(p))]] for p in probs]
        report = type_metrics([r["types"] for r in rows], predicted)
        report["select"] = report["macro_f1"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--task", choices=["detect", "type"], default="detect")
    parser.add_argument("--strategy", choices=["head", "top2", "full", "layerwise"], default="full")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump-embeddings", type=Path, default=None)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    limit_vram()
    run = f"bert_{args.strategy}_{args.task}_s{args.seed}"
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    pad_id = tokenizer.pad_token_id

    def load(split: str) -> list[dict]:
        rows = read_jsonl(args.data / f"{split}.jsonl")
        return [r for r in rows if r["error_flag"]] if args.task == "type" else rows

    train_rows, dev_rows, test_rows = load("main"), load("dev"), load("test")
    train_enc = encode_rows(tokenizer, train_rows, args.task, args.max_length)
    dev_enc = encode_rows(tokenizer, dev_rows, args.task, args.max_length)
    test_enc = encode_rows(tokenizer, test_rows, args.task, args.max_length)

    n_out = 2 if args.task == "detect" else len(TYPE_CODES)
    model = RumorStyleClassifier(args.bert, n_out).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss() if args.task == "detect" else nn.BCEWithLogitsLoss()

    # Minimal pairs: each faulty test sentence against its own corrected form. Their distance
    # shows whether the representation became sensitive to grammar (cf. Table 1 of the paper).
    faulty_rows = [r for r in test_rows if r["error_flag"]]
    fixed_rows = [{**r, "sentence": r["references"][0]} for r in faulty_rows]
    faulty_enc = encode_rows(tokenizer, faulty_rows, args.task, args.max_length)
    fixed_enc = encode_rows(tokenizer, fixed_rows, args.task, args.max_length)

    def dump(tag: str) -> None:
        np.save(args.dump_embeddings / f"{run}_{tag}.npy", cls_vectors(model, test_enc, pad_id, args.task))
        a = cls_vectors(model, faulty_enc, pad_id, args.task)
        b = cls_vectors(model, fixed_enc, pad_id, args.task)
        cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        np.save(args.dump_embeddings / f"{run}_{tag}_paircos.npy", cosine)

    if args.dump_embeddings:
        args.dump_embeddings.mkdir(parents=True, exist_ok=True)
        dump("before")

    plan = unfreeze_plan(args.strategy, args.epochs)
    steps_per_epoch = (len(train_enc) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW([
        {"params": list(model.bert.parameters()), "lr": args.lr, "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("bert.")],
         "lr": args.head_lr, "weight_decay": 0.01},
    ], fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_lambda(total_steps, warmup=int(total_steps * 0.05)))

    logger = JsonlLogger(args.results / "logs" / f"{run}.jsonl")
    rng = np.random.default_rng(args.seed)
    best = {"select": -1.0}
    best_state = None
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    step = 0
    for epoch, k in enumerate(plan):
        set_unfrozen_layers(model, k)
        model.train()
        if k == 0:
            model.bert.eval()  # a frozen encoder must not apply dropout
        running = 0.0
        for input_ids, mask, labels in batches(train_enc, args.batch_size, pad_id, args.task,
                                               order=rng.permutation(len(train_enc)).tolist()):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(input_ids, mask).float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            step += 1
        dev_report = score(args.task, dev_rows, predict(model, dev_enc, pad_id, args.task))
        logger.log(event="dev", epoch=epoch, unfrozen=k, train_loss=running / steps_per_epoch,
                   select=dev_report["select"], elapsed=time.time() - started)
        print(f"{run} epoch {epoch} unfrozen {k} loss {running / steps_per_epoch:.4f} "
              f"dev {dev_report['select']:.4f} {time.time() - started:.0f}s", flush=True)
        if dev_report["select"] > best["select"]:
            best = {"select": dev_report["select"], "epoch": epoch}
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    probs = predict(model, test_enc, pad_id, args.task)
    report = score(args.task, test_rows, probs)
    trainable_last = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report.update(arm=run, task=args.task, strategy=args.strategy, seed=args.seed, best=best,
                  seconds=time.time() - started, peak_gpu_gb=gpu_peak_gb(), steps=step,
                  trainable_params_last_epoch=trainable_last,
                  total_params=sum(p.numel() for p in model.parameters()))
    logger.log(event="summary", **report)
    logger.close()

    if args.dump_embeddings:
        dump("after")

    metrics_dir = args.results / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{run}.test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.results / "predictions" / f"{run}.test.jsonl", (
        {"uid": r["uid"], "prob": p.tolist() if hasattr(p, "tolist") else float(p)}
        for r, p in zip(test_rows, probs)))
    print(json.dumps({k_: report[k_] for k_ in ("arm", "select", "seconds", "peak_gpu_gb")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
