# src/prep_data.py
"""
Robust data preparation for ArXplain.

Usage:
    # Use the entire 'train' split of scitldr AIC config
    python src/prep_data.py --config AIC --split train --use_all

    # Use first 1000 examples from the 'train' split
    python src/prep_data.py --config Abstract --split train --max_examples 1000

    # If allenai/scitldr fails, script will fallback to scientific_papers:pubmed
"""
import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset, DatasetDict

OUT = Path("data")
OUT.mkdir(exist_ok=True)
RANDOM_SEED = 42


def local_scitldr_check():
    # If a local file named scitldr.py exists, datasets will try to use it and fail.
    candidates = ["scitldr.py", "scitldr", "scitldr.*"]
    cwd_files = [p.name for p in Path(".").glob("*")]
    if "scitldr.py" in cwd_files:
        return True
    # Also check src/ or repository root
    for p in ["src/scitldr.py", "./src/scitldr.py"]:
        if Path(p).exists():
            return True
    return False


def write_jsonl(path: Path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def convert_scitldr_example(ex):
    # scitldr configs have different field names; be defensive
    # prefer abstract text fields
    abstract = ex.get("source") or ex.get("paper_abstract") or ex.get("abstract") or ex.get("text") or ""
    # target sometimes a list, sometimes string
    target = ex.get("target")
    if isinstance(target, list) and target:
        target = target[0]
    if target is None:
        target = ex.get("summary") or ex.get("tldr") or ""
    return abstract, target


def prepare_from_scitldr(config_name, split, max_examples=None, use_all=False):
    """
    Loads allenai/scitldr with a config (AIC/Abstract/FullText).
    split can be "train", "validation", "test", or "all".
    """
    print(f"Loading allenai/scitldr config={config_name} split={split} (max_examples={max_examples}, use_all={use_all})")
    ds_splits = []
    if split == "all":
        requested = ["train", "validation", "test"]
    else:
        requested = [split]
    raw_examples = []
    for s in requested:
        # if use_all -> no slicing; else slice by max_examples if provided
        slice_arg = None if use_all else (f"{s}[:{max_examples}]" if max_examples else s)
        try:
            if slice_arg is None:
                ds = load_dataset("allenai/scitldr", config_name, split=s)
            else:
                ds = load_dataset("allenai/scitldr", config_name, split=slice_arg)
        except Exception as e:
            # bubble up exception so caller can fallback
            raise RuntimeError(f"Failed to load allenai/scitldr config={config_name} split={s}: {e}") from e

        print(f"Loaded {len(ds)} examples from scitldr {config_name} {s}")
        for ex in ds:
            abstract, target = convert_scitldr_example(ex)
            if not abstract or not target:
                continue
            raw_examples.append({
                "instruction": "Summarize the abstract in simple plain English for a non-expert (5th-grade level).",
                "input": abstract.replace("\n", " "),
                "output": target.replace("\n", " "),
            })

    return raw_examples


def prepare_from_scientific_papers(split, max_examples=None, use_all=False):
    """
    Fallback dataset: scientific_papers ('arxiv' or 'pubmed'). We'll use 'pubmed' if available.
    """
    print("Falling back to scientific_papers:pubmed (or arxiv if pubmed missing).")
    chosen = None
    try:
        chosen = "scientific_papers"
        # try pubmed first
        slice_arg = None if use_all else (f"train[:{max_examples}]" if max_examples else "train")
        ds = load_dataset("scientific_papers", "pubmed", split=slice_arg)
        print("Loaded scientific_papers:pubmed, size:", len(ds))
    except Exception as e:
        print("pubmed split failed:", e)
        try:
            ds = load_dataset("scientific_papers", "arxiv", split=slice_arg)
            print("Loaded scientific_papers:arxiv, size:", len(ds))
        except Exception as e2:
            raise RuntimeError("Both pubmed and arxiv splits failed to load.") from e2

    raw_examples = []
    for ex in ds:
        # fields differ between pubmed/arxiv: use what we can
        abstract = ex.get("article", {}).get("abstract") if isinstance(ex.get("article"), dict) else ex.get("abstract") or ""
        # try simple summary fields
        target = ex.get("highlights") or ex.get("summary") or ex.get("tldr") or ""
        if isinstance(target, list) and target:
            target = " ".join(target)
        if not abstract or not target:
            continue
        raw_examples.append({
            "instruction": "Summarize the abstract in simple plain English for a non-expert (5th-grade level).",
            "input": abstract.replace("\n", " "),
            "output": target.replace("\n", " "),
        })
    return raw_examples


def split_and_write(all_items, out_dir=OUT, val_frac=0.10):
    random.seed(RANDOM_SEED)
    random.shuffle(all_items)
    n = len(all_items)
    n_val = max(1, int(val_frac * n)) if n > 1 else 0
    train = all_items[:-n_val] if n_val > 0 else all_items
    val = all_items[-n_val:] if n_val > 0 else []
    print(f"Total examples: {n}; writing train={len(train)} val={len(val)}")
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="AIC", help="scitldr config: AIC / Abstract / FullText")
    parser.add_argument("--split", type=str, default="train", help="split: train / validation / test / all")
    parser.add_argument("--max_examples", type=int, default=None, help="limit per split (None => not limited)")
    parser.add_argument("--use_all", action="store_true", help="use all examples in split(s) (overrides max_examples)")
    args = parser.parse_args()

    # check for local scitldr script that trips datasets.new behavior
    if local_scitldr_check():
        print("\nERROR: Found a local 'scitldr.py' in your repo or working directory.")
        print("The 'datasets' library will try to use that and then fail (dataset scripts are deprecated).")
        print("Solution: delete or move the local file (e.g. `rm src/scitldr.py`), then re-run this script.")
        print("Alternatively, run this script from a different directory that does not contain scitldr.py.\n")
        raise SystemExit(1)

    # Try to load allenai/scitldr
    try:
        raw = prepare_from_scitldr(args.config, args.split, max_examples=args.max_examples, use_all=args.use_all)
    except Exception as e:
        print("allenai/scitldr failed to load or parse:", e)
        # fallback
        try:
            raw = prepare_from_scientific_papers(args.split, max_examples=args.max_examples, use_all=args.use_all)
        except Exception as e2:
            print("Fallback also failed:", e2)
            raise SystemExit(1)

    if not raw:
        print("No examples extracted from dataset. Exiting.")
        raise SystemExit(1)

    # If user asked to use 'all' across splits, we already collected across the requested splits; else
    # treat as gathered and do a train/val split.
    split_and_write(raw)

    print("Wrote data/train.jsonl and data/val.jsonl in data/")

if __name__ == "__main__":
    main()
