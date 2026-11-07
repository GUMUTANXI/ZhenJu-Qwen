"""Qwen loading, chat encoding, batched generation and shared training utilities."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .metrics import score_predictions
from .prompts import build_messages, format_target, parse_response

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEVICE = "cuda"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_vram(fraction: float = 0.86) -> None:
    """Cap this process's VRAM so the caching allocator reclaims blocks instead of growing.

    Under Windows WDDM an allocation past dedicated VRAM pages to system RAM rather than
    failing. Variable-length batches fragment the cache, so without a cap training silently
    slows by an order of magnitude once the card fills.
    """
    torch.cuda.set_per_process_memory_fraction(fraction)


def load_tokenizer(path: str | Path, padding_side: str = "left"):
    tokenizer = AutoTokenizer.from_pretrained(str(path), padding_side=padding_side)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base(path: str | Path, dtype: torch.dtype = torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=dtype, attn_implementation="sdpa", device_map={"": 0}
    )
    model.config.use_cache = True
    return model


def attach_lora(model, rank: int, alpha: int, dropout: float = 0.05):
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
        bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_policy(base_path: str | Path, adapter: str | Path | None, trainable: bool = False):
    model = load_base(base_path)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    return model


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "ratio": trainable / total}


def encode_prompt(tokenizer, sentence: str, shots=None, thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(
        build_messages(sentence, shots), tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def sample_target(sample: dict) -> str:
    return format_target(sample["error_flag"], sample["types"], sample["edit_note"], sample["references"][0])


def tokenize_pair(tokenizer, prompt: str, response: str, max_length: int) -> dict | None:
    """Tokenize prompt+response, masking the prompt so loss covers only the response."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(response + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def pad_batch(rows: list[dict], pad_id: int, keys=("input_ids", "labels")) -> dict:
    """Right-pad for training; generation uses the tokenizer's left padding instead."""
    longest = max(len(r["input_ids"]) for r in rows)
    fill = {"input_ids": pad_id, "labels": -100}
    batch = {}
    for key in keys:
        batch[key] = torch.tensor(
            [r[key] + [fill[key]] * (longest - len(r[key])) for r in rows], dtype=torch.long, device=DEVICE
        )
    batch["attention_mask"] = torch.tensor(
        [[1] * len(r["input_ids"]) + [0] * (longest - len(r["input_ids"])) for r in rows],
        dtype=torch.long, device=DEVICE,
    )
    return batch


def length_grouped_batches(rows: list[dict], batch_size: int, rng: random.Random,
                           max_tokens: int = 3000) -> list[list[dict]]:
    """Length-bucketed batches capped by padded token count.

    A fixed batch of the longest sentences would overflow 24 GB of VRAM, and on Windows
    that pages to system RAM and slows training a hundredfold rather than raising OOM.
    """
    order = rows[:]
    rng.shuffle(order)
    bucket = batch_size * 50
    batches: list[list[dict]] = []
    for start in range(0, len(order), bucket):
        chunk = sorted(order[start:start + bucket], key=lambda r: len(r["input_ids"]))
        current: list[dict] = []
        for row in chunk:
            longest = max(len(row["input_ids"]), len(current[-1]["input_ids"]) if current else 0)
            if current and (len(current) >= batch_size or longest * (len(current) + 1) > max_tokens):
                batches.append(current)
                current = []
            current.append(row)
        if current:
            batches.append(current)
    rng.shuffle(batches)
    return batches


def unwrap_causal(model):
    """Return the bare CausalLM; LoRA layers are injected in place, so they stay active."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def response_logits(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project only response tokens through the 150k-way head; full-sequence logits exhaust VRAM.

    Returns (logits for selected positions, their labels, the boolean mask over shifted positions).
    """
    inner = unwrap_causal(model)
    hidden = inner.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         use_cache=False).last_hidden_state
    shift_labels = batch["labels"][:, 1:]
    mask = shift_labels != -100
    logits = inner.lm_head(hidden[:, :-1][mask])
    return logits, shift_labels[mask], mask


def response_loss(model, batch: dict) -> torch.Tensor:
    logits, targets, _ = response_logits(model, batch)
    return torch.nn.functional.cross_entropy(logits.float(), targets)


def sequence_logprobs(model, batch: dict) -> torch.Tensor:
    """Sum of response-token log-probabilities per sequence (the quantity DPO compares)."""
    logits, targets, mask = response_logits(model, batch)
    token_lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    rows = mask.nonzero(as_tuple=True)[0]
    return torch.zeros(mask.shape[0], device=token_lp.device, dtype=token_lp.dtype).index_add(0, rows, token_lp)


def cosine_lambda(total_steps: int, warmup: int):
    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule


@torch.inference_mode()
def generate_texts(model, tokenizer, prompts: list[str], *, batch_size: int = 48, max_new_tokens: int = 256,
                   do_sample: bool = False, temperature: float = 1.0, return_timing: bool = False):
    """Generate for many prompts, longest first so VRAM spikes surface early."""
    was_training = model.training
    model.eval()
    if do_sample:
        config = GenerationConfig(do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                                  max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=[tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id])
    else:
        config = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=[tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id])
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outputs: list[str] = [""] * len(prompts)
    started = time.time()
    produced = 0
    for start in range(0, len(order), batch_size):
        picks = order[start:start + batch_size]
        encoded = tokenizer([prompts[i] for i in picks], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(**encoded, generation_config=config)
        answers = generated[:, encoded["input_ids"].shape[1]:]
        produced += int((answers != tokenizer.pad_token_id).sum())
        for idx, text in zip(picks, tokenizer.batch_decode(answers, skip_special_tokens=True)):
            outputs[idx] = text
    if was_training:
        model.train()
    if return_timing:
        return outputs, {"seconds": time.time() - started, "new_tokens": produced}
    return outputs


def diagnose_samples(model, tokenizer, samples: list[dict], *, shots=None, thinking: bool = False,
                     batch_size: int = 48, max_new_tokens: int = 256) -> tuple[list[dict], dict]:
    prompts = [encode_prompt(tokenizer, s["sentence"], shots, thinking) for s in samples]
    texts, timing = generate_texts(model, tokenizer, prompts, batch_size=batch_size,
                                   max_new_tokens=max_new_tokens, return_timing=True)
    predictions = []
    for sample, text in zip(samples, texts):
        parsed = parse_response(text, sample["sentence"])
        predictions.append({
            "uid": sample["uid"], "flag": parsed.flag, "types": parsed.types, "note": parsed.note,
            "correction": parsed.correction, "well_formed": parsed.well_formed, "raw": text,
        })
    return predictions, timing


def composite_score(report: dict) -> float:
    """Selection score on dev: equal weight on the three FCGEC tiers."""
    return (report["detection"]["accuracy"] + report["type"]["macro_f1"] + report["correction"]["f0.5"]) / 3


def evaluate_dev(model, tokenizer, dev: list[dict], limit: int = 800, batch_size: int = 48) -> dict:
    subset = dev[:limit]
    predictions, _ = diagnose_samples(model, tokenizer, subset, batch_size=batch_size)
    report = score_predictions(subset, predictions)
    report["composite"] = composite_score(report)
    return report


def gpu_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024 ** 3


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w", encoding="utf-8", newline="\n")

    def log(self, **row) -> None:
        self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()
