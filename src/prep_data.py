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


MAX_SHARD_SIZE_MB = 30
MAX_SHARD_SIZE_BYTES = MAX_SHARD_SIZE_MB * 1024 * 1024

def write_jsonl(path: Path, examples: List[dict]):
    """
    Write examples to JSONL, sharding into multiple files if file size >30MB.
    Output files are named path.stem + _part{n}.jsonl
    Example: train_xsum_part1.jsonl, train_xsum_part2.jsonl
    """
    base = path.stem   # e.g., "train_xsum"
    suffix = path.suffix or ".jsonl"

    shard_idx = 1
    shard_size = 0
    shard_file = open(path.with_name(f"{base}_part{shard_idx}{suffix}"), "w", encoding="utf-8")

    for ex in examples:
        line = json.dumps(ex, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        line_size = len(encoded)

        # roll over if adding this line would exceed max shard size
        if shard_size + line_size > MAX_SHARD_SIZE_BYTES:
            shard_file.close()
            print(f"  -> wrote shard {shard_idx} ({shard_size/1e6:.2f} MB)")
            shard_idx += 1
            shard_size = 0
            shard_file = open(path.with_name(f"{base}_part{shard_idx}{suffix}"), "w", encoding="utf-8")

        shard_file.write(line)
        shard_size += line_size

    shard_file.close()
    print(f"  -> wrote shard {shard_idx} ({shard_size/1e6:.2f} MB)")
    print(f"Shards written to {path.parent} with base '{base}'")


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
    Robust loader:
      - If split == "all": iterate ["train","validation","test"] and concatenate.
      - For each split, build a slice argument properly (not None) so load_dataset loads rows.
      - Handles dict rows and string rows (string rows must contain a separator).
      - Respects explicit input_field/target_field if provided.
    """
    def load_one(split_name):
        # build slice arg: if user wants all for this split, pass split_name; otherwise use slicing
        slice_arg = split_name if use_all else (f"{split_name}[:{max_examples}]" if max_examples else split_name)
        print(f"  loading {dataset_id} config={config} split={slice_arg} ...")
        if config:
            ds_part = load_dataset(dataset_id, config, split=slice_arg)
        else:
            ds_part = load_dataset(dataset_id, split=slice_arg)
        print(f"   -> {len(ds_part)} rows")
        return ds_part

    splits_to_load = []
    if split == "all":
        splits_to_load = ["train", "validation", "test"]
    else:
        splits_to_load = [split]

    extracted = []
    total_skipped = 0
    for sp in splits_to_load:
        ds_part = load_one(sp)

        # iterate rows in the split
        skipped = 0
        for raw_item in ds_part:
            # normalize raw_item
            if isinstance(raw_item, dict):
                item = raw_item
            else:
                # treat non-dict as string-like
                item = {"_single_text": str(raw_item)}

            # if explicit fields given, use them
            inf, tf = detect_fields(item, input_field, target_field)

            # if explicit fields not discovered and item is dict, try heuristics
            if not inf and isinstance(item, dict):
                for k in item.keys():
                    kl = k.lower()
                    if any(sub in kl for sub in ["abstract", "source", "paper", "body", "article", "document", "doc", "text"]):
                        inf = k; break
            if not tf and isinstance(item, dict):
                for k in item.keys():
                    kl = k.lower()
                    if any(sub in kl for sub in ["tldr", "summary", "target", "highlights", "headline", "summary_text"]):
                        tf = k; break

            # If item is a single text under _single_text, try parsing it as "input SEP output"
            if (not inf or not tf) and "_single_text" in item:
                s = item["_single_text"].strip()
                # try separators and newline split
                inp, outp = None, None
                for sep in ["\t", " ||| ", "|||", " ### ", "###", " || ", "||"]:
                    if sep in s:
                        parts = s.split(sep, 1)
                        inp = parts[0].strip(); outp = parts[1].strip()
                        break
                if not inp and "\n" in s:
                    head, rest = s.split("\n", 1)
                    inp = head.strip(); outp = rest.strip()
                if inp and outp:
                    abstract = normalize_text(inp)
                    target = normalize_text(outp)
                    if abstract and target:
                        ex_item = {
                            "instruction": f"[SOURCE={dataset_id}] Summarize the abstract in simple plain English for a non-expert (5th-grade level).",
                            "input": abstract,
                            "output": target,
                            "source": dataset_id,
                        }
                        extracted.append(ex_item)
                        continue
                # otherwise skip this row
                skipped += 1
                continue

            # last-resort: if item is dict and we still don't have fields, try first two keys
            if (not inf or not tf) and isinstance(item, dict) and len(item.keys()) >= 2:
                keys = list(item.keys())
                if not inf:
                    inf = keys[0]
                if not tf and len(keys) > 1:
                    tf = keys[1]

            if not inf or not tf:
                skipped += 1
                continue

            abstract = normalize_text(item.get(inf))
            target = normalize_text(item.get(tf))
            if not abstract or not target:
                skipped += 1
                continue

            ex_item = {
                "instruction": f"[SOURCE={dataset_id}] Summarize the abstract in simple plain English for a non-expert (5th-grade level).",
                "input": abstract,
                "output": target,
                "source": dataset_id,
            }
            extracted.append(ex_item)

        if skipped:
            print(f"   (skipped {skipped} unusable rows from {dataset_id}/{sp})")
        total_skipped += skipped

    if total_skipped:
        print(f"  (total skipped across splits: {total_skipped})")
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
