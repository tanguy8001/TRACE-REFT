#!/usr/bin/env python3
"""
Simplified modular evaluator that:
- Uses a lightweight LLM (Llama 3.2 3B Instruct or compatible) to EXTRACT the final answer
  from free-form predictions for accuracy-based tasks, then computes exact-match accuracy.
- For non-accuracy tasks (MeetingBank, Py150, 20Minuten) calls the original evaluation code.

Important: No regex heuristics for parsing model predictions. We only parse the JSON the judge returns.

python3 evaluations/modular_evaluator.py   --pred_dir /cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_sysprompt/predictions   --judge_model_path /cluster/scratch/tdieudonne/initial_model/llama-3.2-3B-Instruct   --csv_out_dir /cluster/scratch/tdieudonne/csv_pairs_reft_sysprompt
"""

import json
import os
import csv
import logging
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from metrics import caculate_accuracy
from evaluations import eval_MeetingBank, eval_Py150, eval_20Minuten


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modular_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


ACC_TASKS = {"C-STANCE", "FOMC", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds"}
NON_ACC_TASKS = {"MeetingBank", "Py150", "20Minuten"}


class LLMExtractor:
    """LLM-based extractor that returns a single final answer via a JSON object.

    We use the chat-like prompt template consistent with download_model.py and expect the
    assistant to return ONLY a JSON object like {"answer": "<VALUE>"}.
    """

    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        # Use bf16 on GPU, fp32 on CPU
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _chat_wrap(instruction: str, text_to_parse: str) -> str:
        # Matches the template structure used in download_model.py
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{instruction}<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Text to parse:\n{text_to_parse}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

    def _build_instruction(self, task_type: str) -> str:
        if task_type in ("C-STANCE", "FOMC"):
            return (
                "Extract the final answer letter ONLY (uppercase).\n"
                "Rules:\n"
                "- Allowed answers: A, B, C.\n"
                "- Return ONLY a JSON object: {\"answer\": \"<LETTER>\"}.\n"
                "- Do NOT include any other text."
            )
        if task_type == "ScienceQA":
            return (
                "Extract the final answer letter ONLY (uppercase).\n"
                "Rules:\n"
                "- Allowed answers: A, B, C, D.\n"
                "- Return ONLY a JSON object: {\"answer\": \"<LETTER>\"}.\n"
                "- Do NOT include any other text."
            )
        if task_type in "NumGLUE-cm":
            return (
                "Extract the final answer ONLY.\n"
                "Rules:\n"
                "- If the answer is a number, return it exactly as written (including decimals).\n"
                "- If the final answer is not numeric but a single-word month name, return that single word exactly.\n"
                "- Return ONLY a JSON object: {\"answer\": \"<VALUE>\"}.\n"
                "- Do NOT include units or explanations."
            )
        if task_type in "NumGLUE-ds":
            return (
                "Extract the final number answer ONLY.\n"
                "Rules:\n"
                "- Return ONLY a JSON object: {\"answer\": \"<VALUE>\"}.\n"
                "- Do NOT include any other text, units, percentage, or explanations."
            )
        # Fallback (should not happen for acc tasks)
        return (
            "Extract the final answer ONLY. Return ONLY JSON: {\"answer\": \"<VALUE>\"}."
        )

    @staticmethod
    def _extract_json_answer(text: str) -> Optional[str]:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1 or end <= start:
                return None
            obj = json.loads(text[start:end+1])
            val = obj.get("answer")
            if val is None:
                return None
            # Coerce to string for uniform comparison
            return str(val).strip()
        except Exception:
            return None

    def extract(self, task_type: str, prediction_text: str) -> Optional[str]:
        instruction = self._build_instruction(task_type)
        prompt = self._chat_wrap(instruction, prediction_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Find assistant segment and strip the trailing <|eot_id|>
        assistant_tag = "<|start_header_id|>assistant<|end_header_id|>"
        idx = decoded.rfind(assistant_tag)
        if idx != -1:
            segment = decoded[idx + len(assistant_tag):].strip()
        else:
            # Fallback to entire decoded tail
            segment = decoded[len(prompt):].strip()
        segment = segment.replace("<|eot_id|>", "").strip()
        return self._extract_json_answer(segment)


class DatasetEvaluator:
    def __init__(self, extractor: LLMExtractor, csv_out_dir: str):
        self.extractor = extractor
        self.csv_out_dir = csv_out_dir
        os.makedirs(self.csv_out_dir, exist_ok=True)

    @staticmethod
    def detect_task_type(filename: str) -> str:
        for t in list(ACC_TASKS | NON_ACC_TASKS):
            if t in filename:
                return t
            raise ValueError(f"Unknown task type in filename: {filename}")
    
    @staticmethod
    def _parse_results_file(file_path: str) -> Tuple[List[str], List[str], List[str]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompts = data.get('prompts', [])
        results = data.get('results', [])
        labels = data.get('labels', [])
        n = min(len(prompts), len(results), len(labels) if labels else len(results))
        return prompts[:n], results[:n], labels[:n]

    def evaluate_file(self, file_path: str) -> Dict[str, Any]:
        filename = Path(file_path).name
        task = self.detect_task_type(filename)
        prompts, results, labels = self._parse_results_file(file_path)
        logger.info(f"Evaluating {filename} (task={task}) with {len(results)} samples")

        out: Dict[str, Any] = {"file": filename, "task": task}

        if task in ACC_TASKS:
            extracted: List[str] = []
            csv_rows: List[Tuple[int, str, str, str]] = []  # id, extracted, label, raw
            for i, pred in enumerate(results):
                ans = self.extractor.extract(task, pred)
                extracted.append(ans if ans is not None else "")
                # Save row for human inspection
                csv_rows.append((i + 1, ans if ans is not None else "", labels[i] if i < len(labels) else "", pred))

            # Prepare labels for ScienceQA per user's rule: take the letter at index 1
            if task == "ScienceQA":
                labels_prepared: List[str] = []
                for lbl in labels:
                    if isinstance(lbl, str) and len(lbl) >= 1:
                        labels_prepared.append(lbl[0])  # first character is the answer letter
                    elif isinstance(lbl, str) and len(lbl) == 1:
                        labels_prepared.append(lbl[0])
                    else:
                        labels_prepared.append("")
            else:
                labels_prepared = labels

            # Save CSV pairs
            csv_name = f"pairs_{filename.replace('.json','')}.csv"
            csv_path = os.path.join(self.csv_out_dir, csv_name)
            with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                writer.writerow(["id", "extracted_answer", "label", "raw_prediction"])
                writer.writerows(csv_rows)
            logger.info(f"Saved pairs CSV to {csv_path}")

            # Compute accuracy via original exact-match
            acc = caculate_accuracy(extracted, labels_prepared)
            out["metric"] = "accuracy"
            out["value"] = acc
            logger.info(f"Accuracy: {acc:.3f}")
        else:
            # Use original evaluation for non-accuracy tasks
            if task == "MeetingBank":
                ev = eval_MeetingBank.eval(results, labels)
                metric_key = "rouge-L"
            elif task == "Py150":
                ev = eval_Py150.eval(results, labels)
                metric_key = "similarity"
            elif task == "20Minuten":
                # 20Minuten expects sources/prompts as first arg
                ev = eval_20Minuten.eval(prompts, results, labels)
                metric_key = "sari"
            else:
                raise ValueError(f"Unhandled non-accuracy task: {task}")
            out["metric"], out["value"] = metric_key, ev.get(metric_key)
            logger.info(f"{metric_key}: {out['value']}")

        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Directory containing results-*-*-<Task>.json files")
    ap.add_argument("--judge_model_path", required=True, help="Path to Llama 3.2 3B Instruct (local) for extraction")
    ap.add_argument("--csv_out_dir", default="csv_pairs", help="Directory to save CSV (prediction,label) pairs")
    args = ap.parse_args()

    extractor = LLMExtractor(args.judge_model_path)
    evaluator = DatasetEvaluator(extractor, args.csv_out_dir)

    files = sorted([
        str(p) for p in Path(args.pred_dir).glob("results-*-*-*.json")
    ])
    if not files:
        logger.warning(f"No result json files found under {args.pred_dir}")
        return

    summary: Dict[str, Any] = {}
    for f in files:
        try:
            res = evaluator.evaluate_file(f)
            summary[Path(f).name] = res
        except Exception as e:
            logger.error(f"Failed to evaluate {f}: {e}")

    logger.info("\n==== Per-file Summary ====")
    for k, v in summary.items():
        logger.info(f"{k}: {v.get('metric')}={v.get('value')}")


if __name__ == "__main__":
    main()