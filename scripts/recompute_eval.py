#!/usr/bin/env python3
import os
import re
import json
import argparse
import sys

# Ensure project root on sys.path so that 'evaluations' can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from evaluations import (
    eval_ScienceQA,
    eval_MeetingBank,
    eval_PapyrusF,
    eval_CStance,
    eval_Py150,
    eval_FOMC,
    eval_NumGLUE_cm,
    eval_NumGLUE_ds,
    eval_20Minuten,
)


TASK2EVAL = {
    "ScienceQA": eval_ScienceQA,
    "MeetingBank": eval_MeetingBank,
    "Papyrus-f": eval_PapyrusF,
    "C-STANCE": eval_CStance,
    "Py150": eval_Py150,
    "FOMC": eval_FOMC,
    "NumGLUE-cm": eval_NumGLUE_cm,
    "NumGLUE-ds": eval_NumGLUE_ds,
    "20Minuten": eval_20Minuten,
}


def recompute_file(path: str, dry_run: bool = False) -> bool:
    base = os.path.basename(path)
    m = re.match(r"results-\d+-\d+-(.+)\.json$", base)
    if not m:
        return False
    task = m.group(1)
    if task not in TASK2EVAL:
        return False

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    res = d.get("results", [])
    labels = d.get("labels", [])

    # Some tasks require prompts as inputs (20Minuten SARI), maintain backward compatibility
    if task == "20Minuten":
        prompts = d.get("prompts", [])
        eval_out = TASK2EVAL[task].eval(prompts, res, labels)
    else:
        eval_out = TASK2EVAL[task].eval(res, labels)

    old = d.get("eval", {})
    d["eval"] = eval_out

    changed = json.dumps(old, sort_keys=True) != json.dumps(eval_out, sort_keys=True)
    if not dry_run and changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Directory containing results-*-*-<Task>.json files")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    total = 0
    changed = 0
    for fname in sorted(os.listdir(args.pred_dir)):
        if not fname.startswith("results-") or not fname.endswith(".json"):
            continue
        total += 1
        fpath = os.path.join(args.pred_dir, fname)
        try:
            if recompute_file(fpath, dry_run=args.dry_run):
                changed += 1
        except Exception as e:
            print(f"[WARN] failed to recompute {fpath}: {e}")

    print(f"Processed {total} files; updated {changed}.")


if __name__ == "__main__":
    main()


