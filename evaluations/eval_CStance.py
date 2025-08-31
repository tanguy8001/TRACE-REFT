import json
import re
from metrics import caculate_accuracy


def _resolve_abc(pred_or_gt_list):
    """Normalize free-form generations to A/B/C for stance classification.

    Handles:
    - Chinese prompts with choices A.支持, B.反对, C.中立 mapping -> A/B/C
    - English words support/oppose/neutral mapping -> A/B/C when present
    - Patterns like "Answer: B" or lines containing only A/B/C
    """
    normalized = []
    for raw in pred_or_gt_list:
        text = "" if raw is None else str(raw)
        t = text.strip()
        low = t.lower()

        # 1) Chinese keywords
        if "支持" in t:
            normalized.append("A")
            continue
        if "反对" in t:
            normalized.append("B")
            continue
        if "中立" in t:
            normalized.append("C")
            continue

        # 2) English keywords
        if any(w in low for w in ["support", "favor", "favour"]):
            normalized.append("A")
            continue
        if any(w in low for w in ["oppose", "against"]):
            normalized.append("B")
            continue
        if "neutral" in low:
            normalized.append("C")
            continue

        # 3) Labeled patterns
        m = re.search(r"(?i)\b(answer|my answer|final answer)\b\s*[:\-]?\s*([ABC])\b", t)
        if not m:
            m = re.search(r"(?i)\bstance\b[^A-Za-z]{0,20}([ABC])\b", t)
        if not m:
            m = re.search(r"\b([ABC])\s*(?:[\.)\]:]|\b)", t)
        if m:
            normalized.append(m.group(2 if m.lastindex and m.lastindex >= 2 else 1).upper())
            continue

        # 4) Single-letter line
        picked = None
        for line in t.splitlines():
            s = line.strip()
            if len(s) == 1 and s.upper() in {"A", "B", "C"}:
                picked = s.upper()
                break
        if picked is not None:
            normalized.append(picked)
            continue

        # 5) Fallback: first non-space char
        first_char = t[:1].upper()
        normalized.append(first_char if first_char in {"A", "B", "C"} else "")

    return normalized


def eval(predicted_sequences, ground_truths):
    preds = _resolve_abc(predicted_sequences)
    gts = _resolve_abc(ground_truths)
    accuracy = caculate_accuracy(preds, gts)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
