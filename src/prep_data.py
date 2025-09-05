#!/usr/bin/env python3
"""
Simple data prep: load allenai/scitldr (AIC/Abstract/FullText) and write data/train.jsonl + data/val.jsonl.

Usage examples:
  # Use entire 'train' split (every row)
  python src/prep_data.py --config AIC --split train --use_all

  # Use all splits concatenated (train+validation+test) and then split into train/val
  python src/prep_data.py --config Abstract --split all --use_all

  # Use first 2000 examples of train for quick iteration
  python src/prep_data.py --config AIC --split train --max_examples 2000
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path("data")
OUT.mkdir(exist_ok=True)
RANDOM_SEED = 42


def local_scitldr_check():
    """If a local scitldr.py exists, datasets will try to use it and fail on new versions."""
    candidates = ["scitldr.py", "scitldr"]
    # check cwd and src/
    for p in ["./scitldr.py", "./src/scitldr.py"]:
        if Path(p).exists():
            return p
    return None


def write_jsonl(path: Path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def convert_scitldr_example(ex):
    """
    Extract abstract and tldr from a scitldr example defensively.
    Returns (abstract, tldr) or (None, None) if extraction fails.
    """
    abstract = ex.get("source") or ex.get("paper_abstract") or ex.get("abstract") or ""
    target = ex.get("target")
    if isinstance(target, list) and target:
        target = target[0]
    if target is None:
        target = ex.get("summary") or ex.get("tldr") or ""
    if not abstract or not target:
        return None, None
    return abstract.replace("\n", " "), str(target).replace("\n", " ")


def load_scitldr_all(config_name: str, split: str, max_examples: int = None, use_all: bool = False):
    """
    Load scitldr with the given config and split selection.

    - config_name: "AIC" | "Abstract" | "FullText"
    - split: "train" | "validation" | "test" | "all"
    - max_examples: integer limit per split (ignored if use_all=True)
    - use_all: if True, use every example from the requested split(s)
    """
    if split == "all":
        splits = ["train", "validation", "test"]
    else:
        splits = [split]

    examples = []
    for s in splits:
        # build slicing arg
        if use_all:
            slice_arg = s
        else:
            slice_arg = f"{s}[:{max_examples}]" if max_examples else s

        print(f"Loading allenai/scitldr config={config_name} split={slice_arg} ...")
        ds = load_dataset("allenai/scitldr", config_name, split=slice_arg)
        print(f" -> loaded {len(ds)} examples from {s}")

        for item in ds:
            abstract, tldr = convert_scitldr_example(item)
            if not abstract or not tldr:
                continue
            examples.append({
                "instruction": "Summarize the abstract in simple plain English for a non-expert (5th-grade level).",
                "input": abstract,
                "output": tldr,
            })

    return examples


def split_and_write(all_items, out_dir=OUT, val_frac=0.10):
    random.seed(RANDOM_SEED)
    random.shuffle(all_items)
    n = len(all_items)
    if n == 0:
        raise SystemExit("No examples extracted — aborting.")
    n_val = max(1, int(val_frac * n)) if n > 1 else 0
    train = all_items[:-n_val] if n_val > 0 else all_items
    val = all_items[-n_val:] if n_val > 0 else []
    print(f"Total examples collected: {n} -> train: {len(train)}, val: {len(val)}")
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    print(f"Wrote {out_dir/'train.jsonl'} and {out_dir/'val.jsonl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="AIC", help="scitldr config: AIC / Abstract / FullText")
    parser.add_argument("--split", type=str, default="train", help="split: train / validation / test / all")
    parser.add_argument("--max_examples", type=int, default=None, help="limit per split (ignored if --use_all)")
    parser.add_argument("--use_all", action="store_true", help="use all examples from the requested split(s)")
    args = parser.parse_args()

    # warn / abort if a local scitldr.py exists (this caused the earlier error)
    bad = local_scitldr_check()
    if bad:
        print(f"\nERROR: found local dataset script {bad}.")
        print("Please remove or move that file; it prevents loading the remote dataset.")
        raise SystemExit(1)

    # load only from allenai/scitldr (no fallbacks)
    examples = load_scitldr_all(args.config, args.split, max_examples=args.max_examples, use_all=args.use_all)

    # write train/val split
    split_and_write(examples)


if __name__ == "__main__":
    main()
