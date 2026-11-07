"""Download the FCGEC corpus (Apache-2.0) and the base model weights. Scripts stay outside these folders."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FCGEC_RAW = "https://raw.githubusercontent.com/xlxwalex/FCGEC/main/data/{}"
FILES = ["FCGEC_train_filtered.json", "FCGEC_valid.json", "README.md"]
MODELS = {"Qwen/Qwen3-1.7B": "Qwen3-1.7B", "google-bert/bert-base-chinese": "bert-base-chinese"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/fcgec", type=Path)
    parser.add_argument("--model-dir", default="models", type=Path)
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = args.data_dir / name
        if not target.exists():
            print("fetch", name)
            urllib.request.urlretrieve(FCGEC_RAW.format(name), target)

    if args.skip_models:
        return
    from huggingface_hub import snapshot_download

    for repo, folder in MODELS.items():
        print("fetch", repo)
        snapshot_download(repo, local_dir=args.model_dir / folder,
                          allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt", "LICENSE*"],
                          ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot", "*.onnx"])


if __name__ == "__main__":
    main()
