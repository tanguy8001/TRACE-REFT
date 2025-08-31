import json
import re
from metrics import caculate_accuracy


def _normalize_number_strings(texts):
    normed = []
    for raw in texts:
        t = "" if raw is None else str(raw)
        t = t.strip()
        t = t.splitlines()[0] if t else ""
        m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
        normed.append(m.group(0) if m else t)
    return normed


def eval(predicted_sequences, ground_truths):
    preds = _normalize_number_strings(predicted_sequences)
    gts = _normalize_number_strings(ground_truths)
    accuracy = caculate_accuracy(preds, gts)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
