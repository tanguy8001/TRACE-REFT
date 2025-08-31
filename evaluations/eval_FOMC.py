import json
import re
from metrics import caculate_accuracy


def _resolve_abc(pred_or_gt_list):
    """Normalize free-form generations to single-letter choices A/B/C.

    Heuristics handle cases like:
    - "Answer: B. hawkish"
    - "The monetary policy stance is C. neutral"
    - Lines containing only "A", "B", or "C"
    - Spelled-out words: dovish/hawkish/neutral -> A/B/C
    Falls back to first non-empty capital A/B/C at the start if present.
    """
    normalized = []
    for raw in pred_or_gt_list:
        text = "" if raw is None else str(raw)
        t = text.strip()
        low = t.lower()

        # 1) Spelled-out words
        if "dovish" in low:
            normalized.append("A")
            continue
        if "hawkish" in low:
            normalized.append("B")
            continue
        if "neutral" in low:
            normalized.append("C")
            continue

        # 2) Common labeled patterns (Answer:, My answer:, Stance is ...)
        m = re.search(r"(?i)\b(answer|my answer|final answer)\b\s*[:\-]?\s*([ABC])\b", t)
        if not m:
            m = re.search(r"(?i)\bstance\b[^A-Za-z]{0,20}([ABC])\b", t)
        if not m:
            m = re.search(r"(?i)\boption\b[^A-Za-z]{0,20}([ABC])\b", t)
        if m:
            normalized.append(m.group(2 if m.lastindex and m.lastindex >= 2 else 1).upper())
            continue

        # 3) Letter with punctuation, e.g., "B." or "C)"
        m = re.search(r"\b([ABC])\s*(?:[\.)\]:]|\b)", t)
        if m:
            normalized.append(m.group(1).upper())
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

        # 5) Fallback: if the very first non-space char is A/B/C
        first_char = t[:1].upper()
        normalized.append(first_char if first_char in {"A", "B", "C"} else "")

    return normalized


def eval(predicted_sequences, ground_truths):
    preds = _resolve_abc(predicted_sequences)
    gts = _resolve_abc(ground_truths)
    accuracy = caculate_accuracy(preds, gts)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
