#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import torch


#python clmm/TRACE/scripts/check_reft_alphas.py --output_dir /cluster/scratch/$USER/outputs_LLM-CL/cl/REFT-CL --rounds 0,1,2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ReFT-CL alpha parameters across saved rounds and report changes.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory where round subfolders (0,1,2,...) were saved.")
    parser.add_argument(
        "--rounds",
        type=str,
        default=None,
        help="Comma-separated list of round ids to inspect (e.g., 0,1,2). If omitted, autodetect numeric subfolders.")
    parser.add_argument(
        "--model_filename",
        type=str,
        default="pytorch_model.bin",
        help="Model filename inside each round subfolder.")
    parser.add_argument(
        "--max_print",
        type=int,
        default=64,
        help="Max alpha entries to print (to keep output readable).")
    return parser.parse_args()


def _discover_round_dirs(output_dir: str) -> List[Tuple[int, str]]:
    entries = []
    try:
        for name in os.listdir(output_dir):
            if re.fullmatch(r"\d+", name):
                entries.append((int(name), os.path.join(output_dir, name)))
    except FileNotFoundError:
        print(f"[ERR] Output directory not found: {output_dir}")
        sys.exit(1)
    return sorted(entries, key=lambda x: x[0])


def _parse_rounds_arg(output_dir: str, rounds_arg: str) -> List[Tuple[int, str]]:
    if not rounds_arg:
        return _discover_round_dirs(output_dir)
    ids = []
    for tok in rounds_arg.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            print(f"[ERR] Non-numeric round id: {tok}")
            sys.exit(1)
        ids.append(int(tok))
    return [(rid, os.path.join(output_dir, str(rid))) for rid in ids]


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[ERR] Failed to load state dict: {path}: {e}")
        sys.exit(1)


def _extract_alpha_tensors(state_dict: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """
    Extracts alpha parameters by matching names that start with either:
    - 'reftcl_alpha_bank.alphas.'
    - 'module.reftcl_alpha_bank.alphas.' (in case a wrapper prefixed names)
    Returns a map: alpha_index -> tensor
    """
    prefixes = ("reftcl_alpha_bank.alphas.", "module.reftcl_alpha_bank.alphas.")
    found: Dict[int, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        for pref in prefixes:
            if name.startswith(pref):
                idx_str = name.split(".")[-1]
                if idx_str.isdigit():
                    found[int(idx_str)] = tensor.detach().cpu()
                break
    return dict(sorted(found.items(), key=lambda kv: kv[0]))


def _format_float(x: float) -> str:
    # compact format but precise enough to see changes
    return f"{x:.8f}"


def main() -> None:
    args = parse_args()
    round_entries = _parse_rounds_arg(args.output_dir, args.rounds)
    if not round_entries:
        print("[ERR] No round directories found.")
        sys.exit(1)

    print(f"[INFO] Inspecting rounds: {[rid for rid, _ in round_entries]}")

    round_to_alphas: Dict[int, Dict[int, torch.Tensor]] = {}
    for rid, rdir in round_entries:
        model_path = os.path.join(rdir, args.model_filename)
        if not os.path.isfile(model_path):
            print(f"[WARN] Missing model file for round {rid}: {model_path}")
            continue
        sd = _load_state_dict(model_path)
        alphas = _extract_alpha_tensors(sd)
        if not alphas:
            print(f"[WARN] No alpha tensors found in round {rid} state dict.")
        round_to_alphas[rid] = alphas

    if not round_to_alphas:
        print("[ERR] No alpha tensors were found in any provided rounds.")
        sys.exit(1)

    # Determine union of alpha indices across all rounds
    all_indices = sorted({idx for alphas in round_to_alphas.values() for idx in alphas.keys()})
    print(f"[INFO] Detected alpha indices: {all_indices}")

    # Print values per round and diffs from previous round
    print("\n=== Alpha Values by Round ===")
    printed = 0
    for idx in all_indices:
        if printed >= args.max_print:
            print("[INFO] Output truncated by --max_print.")
            break
        line_vals = []
        prev = None
        changed_any = False
        for rid, _ in round_entries:
            t = round_to_alphas.get(rid, {}).get(idx)
            if t is None:
                line_vals.append("-")
                prev = None
                continue
            val = t.item() if t.numel() == 1 else float(t.reshape(-1)[0].item())
            if prev is not None and abs(val - prev) > 1e-12:
                changed_any = True
            line_vals.append(_format_float(val))
            prev = val
        status = "changed" if changed_any else "unchanged"
        print(f"alpha[{idx}]: {line_vals}  -> {status}")
        printed += 1

    # Detailed diffs
    print("\n=== Alpha Diffs (round_n - round_{n-1}) ===")
    printed = 0
    for idx in all_indices:
        if printed >= args.max_print:
            print("[INFO] Diff output truncated by --max_print.")
            break
        diffs_line = []
        prev = None
        for rid, _ in round_entries:
            t = round_to_alphas.get(rid, {}).get(idx)
            if t is None or t.numel() == 0:
                diffs_line.append("-")
                prev = None
                continue
            val = t.item() if t.numel() == 1 else float(t.reshape(-1)[0].item())
            if prev is None:
                diffs_line.append("-")
            else:
                diffs_line.append(_format_float(val - prev))
            prev = val
        print(f"alpha[{idx}] diffs: {diffs_line}")
        printed += 1

    print("\n[INFO] Done.")


if __name__ == "__main__":
    import torch, os
    ckpt = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL/5/pytorch_model.bin"
    sd = torch.load(ckpt, map_location="cpu")
    alphas = [sd[k].item() for k in sorted(sd) if k.startswith("reftcl_alpha_bank.alphas.")]
    print("num_alphas:", len(alphas), "values:", alphas)
    main()





