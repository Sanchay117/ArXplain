from datasets import load_dataset
from pathlib import Path
import json, random
OUT = Path("data")
OUT.mkdir(exist_ok=True)

def make_small_scitldr(max_examples=1000):
    ds = load_dataset("allenai/scitldr", "AIC", split=f"train[:{max_examples}]")
    lines=[]
    for ex in ds:
        abstract = ex.get("source") or ex.get("paper_abstract") or ex.get("abstract", "")
        target = None
        if "target" in ex and ex["target"]:
            target = ex["target"][0] if isinstance(ex["target"], list) else ex["target"]
        else:
            target = ex.get("target_text") or ex.get("summary") or ""
        if not abstract or not target: 
            continue
        lines.append({
            "instruction": "Summarize the abstract in simple plain English for a non-expert (5th-grade level).",
            "input": abstract.replace("\n"," "),
            "output": target.replace("\n"," ")
        })
    random.shuffle(lines)
    n = int(0.9*len(lines))
    for fn, chunk in [("train.jsonl", lines[:n]), ("val.jsonl", lines[n:])]:
        with open(OUT/fn, "w") as f:
            for ex in chunk:
                f.write(json.dumps(ex) + "\n")
    print("Wrote", OUT/"train.jsonl", OUT/"val.jsonl")

if __name__=="__main__":
    make_small_scitldr(max_examples=1000)
