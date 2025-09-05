#!/usr/bin/env python3
"""
Universal data prep for ArXplain.

Features:
- Load one or more HF datasets (comma-separated)
- Optional dataset config names (comma-separated, or 'None')
- Auto-detect common abstract/summary fields or use explicit fields
- Use full split(s) or a limited number of examples
- Writes data/train.jsonl and data/val.jsonl (90/10 default)

Examples:
# single dataset (scitldr AIC), use entire train split
python src/prep_data.py --datasets allenai/scitldr --configs AIC --split train --use_all

# multiple datasets: first config for first dataset, blank for second
python src/prep_data.py --datasets allenai/scitldr,scientific_papers --configs AIC,pubmed --split train --use_all

# single dataset, specify fields manually
python src/prep_data.py --datasets some/ds --input_field abstract_text --target_field summary_text --max_examples 1000
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset

OUT = Path("data")
OUT.mkdir(exist_ok=True)
RANDOM_SEED = 42


def local_dataset_script_check():
    for p in ["./scitldr.py", "./src/scitldr.py"]:
        if Path(p).exists():
            return p
    return None


COMMON_INPUT_FIELDS = [
    "source", "paper_abstract", "abstract", "text", "article", "body", "article_abstract", "article_text", "document"
]
COMMON_TARGET_FIELDS = [
    "target", "summary", "tldr", "highlights", "abstract_summary", "summary_text"
]


def write_jsonl(path: Path, examples: List[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def detect_fields(example: dict, input_field: Optional[str], target_field: Optional[str]):
    # Returns (input_field, target_field) names to use for this example
    if input_field and target_field:
        return input_field, target_field

    # Try explicit detection from most likely candidates
    found_input = None
    found_target = None

    for f in COMMON_INPUT_FIELDS:
        if f in example:
            found_input = f
            break
    for f in COMMON_TARGET_FIELDS:
        if f in example:
            found_target = f
            break

    return found_input, found_target


def normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        # join list of primitives/dicts
        items = []
        for it in x:
            if isinstance(it, dict):
                # try to pick a value
                for v in it.values():
                    if isinstance(v, str):
                        items.append(v); break
            else:
                items.append(str(it))
        x = " ".join(items)
    elif isinstance(x, dict):
        # try to extract a likely text field
        if "text" in x:
            x = x["text"]
        elif "summary" in x:
            x = x["summary"]
        else:
            # join all string values
            x = " ".join([str(v) for v in x.values() if isinstance(v, str)])
    x = str(x).strip()
    x = " ".join(x.split())
    return x


def extract_examples_from_split(dataset_id: str, config: Optional[str], split: str,
                                max_examples: Optional[int], use_all: bool,
                                input_field: Optional[str], target_field: Optional[str]):
    """
    Loads one dataset split and yields (input, target) formatted dicts
    """
    slice_arg = None if use_all else (f"{split}[:{max_examples}]" if max_examples else split)
    print(f"Loading {dataset_id} config={config} split={slice_arg} ...")
    ds = load_dataset(dataset_id, config, split=slice_arg) if config else load_dataset(dataset_id, split=slice_arg)
    print(f" -> loaded {len(ds)} rows from {dataset_id} / {split}")

    extracted = []
    for item in ds:
        # Decide fields to use for this item
        inf, tf = detect_fields(item, input_field, target_field)

        # If detection failed, try to be more aggressive: check all keys
        if not inf:
            for k in item.keys():
                if any(sub in k.lower() for sub in ["abstract", "source", "paper", "body", "article"]):
                    inf = k; break
        if not tf:
            for k in item.keys():
                if any(sub in k.lower() for sub in ["tldr", "summary", "target", "highlights", "abstract"]):
                    tf = k; break

        if not inf or not tf:
            # can't extract from this row
            continue

        abstract = normalize_text(item.get(inf))
        target = normalize_text(item.get(tf))

        if not abstract or not target:
            continue

        extracted.append({
            "instruction": "Summarize the abstract in simple plain English for a non-expert (5th-grade level).",
            "input": abstract,
            "output": target
        })
    return extracted


def collect_from_datasets(datasets: List[str], configs: List[Optional[str]],
                          split: str, max_examples: Optional[int], use_all: bool,
                          input_field: Optional[str], target_field: Optional[str]):
    all_items = []
    for idx, ds_id in enumerate(datasets):
        conf = configs[idx] if idx < len(configs) else None
        items = extract_examples_from_split(ds_id, conf, split, max_examples, use_all, input_field, target_field)
        print(f"Extracted {len(items)} usable examples from {ds_id}")
        all_items.extend(items)
    return all_items


def split_and_write(args, all_items, out_dir=OUT, val_frac=0.10):
    random.seed(RANDOM_SEED)
    random.shuffle(all_items)
    n = len(all_items)
    if n == 0:
        raise SystemExit("No examples extracted — nothing to write.")
    n_val = max(1, int(val_frac * n)) if n > 1 else 0
    train = all_items[:-n_val] if n_val > 0 else all_items
    val = all_items[-n_val:] if n_val > 0 else []
    print(f"Total examples collected: {n} -> train: {len(train)}, val: {len(val)}")
    write_jsonl(out_dir / f"train_{args.file_name}.jsonl", train)
    write_jsonl(out_dir / f"val_{args.file_name}.jsonl", val)
    print(f"Wrote {out_dir/'train.jsonl'} and {out_dir/'val.jsonl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated HF dataset ids, e.g. 'allenai/scitldr' or 'allenai/scitldr,scientific_papers'")
    parser.add_argument("--configs", type=str, default="", help="Comma-separated configs aligning with datasets (use empty for none).")
    parser.add_argument("--split", type=str, default="train", help="split: train/validation/test/all")
    parser.add_argument("--max_examples", type=int, default=None, help="limit per split (ignored if --use_all)")
    parser.add_argument("--use_all", action="store_true", help="use whole split(s) instead of slicing")
    parser.add_argument("--input_field", type=str, default=None, help="explicit input field to read (optional)")
    parser.add_argument("--target_field", type=str, default=None, help="explicit target field to read (optional)")
    parser.add_argument("--file_name",type=str,required=True,help="Name of the train and validation JSON files")
    args = parser.parse_args()

    bad = local_dataset_script_check()
    if bad:
        print(f"ERROR: found local dataset script file {bad}. Move or remove it before running.")
        raise SystemExit(1)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    configs = [x.strip() if x.strip().lower() != "none" and x.strip() != "" else None for x in (args.configs.split(",") if args.configs else [])]

    items = collect_from_datasets(datasets, configs, args.split, args.max_examples, args.use_all, args.input_field, args.target_field)
    split_and_write(args,items)


if __name__ == "__main__":
    main()
