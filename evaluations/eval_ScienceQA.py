import json
from metrics import caculate_bleu, caculate_rouge, caculate_accuracy


# resolving answer and reasoning (robust to empty or malformed strings)
def resolve(dataset: list):
    answers = []
    reasonings = []
    for datium in dataset:
        text = datium if isinstance(datium, str) else ("" if datium is None else str(datium))
        text = text.strip()
        ans = text[0] if len(text) > 0 else ""
        reasoning = text[2:] if len(text) > 2 else (text[1:] if len(text) > 1 else "")
        answers.append(ans)
        reasonings.append(reasoning)
    outputs = {"answers": answers, "reasonings": reasonings}
    return outputs


def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    bleu_1 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
    bleu_4 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
    rouge = caculate_rouge(outputs["reasonings"], gts["reasonings"])
    accuracy = caculate_accuracy(outputs["answers"], gts["answers"])

    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result