#!/usr/bin/env python3
"""
Modular evaluator for multiple RefT datasets and tasks.
This script can evaluate different types of tasks with appropriate evaluation logic.
"""

import json
import os
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
import re

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modular_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LlamaJudge:
    """Helper to judge (prompt, answer, label) correctness using a Llama model."""

    def __init__(self, model_name: Optional[str] = None, tokenizer=None, model=None, device: Optional[str] = None):
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
            # Infer device from model if possible, else CPU/CUDA default
            self.device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        else:
            if model_name is None:
                raise ValueError("model_name must be provided if tokenizer/model are not passed")
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise ImportError("transformers not available to load Llama judge")
            self.device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.float16 if (torch and self.device == "cuda") else None),
                device_map=("auto" if (torch and self.device == "cuda") else None),
                low_cpu_mem_usage=True
            )
            if torch and self.device == "cpu":
                self.model = self.model.to(self.device)

    def _build_prompt(self, prompt: str, answer: str, label: str, task_type: str) -> str:
        instruction = (
            "You are a strict evaluator. Determine whether the model answer is coherent with (semantically consistent with) the expected label for this task. Base your decision on the label's meaning if provided; do not require an exact string match.\n"
            'Return ONLY a JSON object with this exact format: {"equal": true} or {"equal": false}.\n'
            "Do not include any other text."
        )
        return (
            f"{instruction}\n\n"
            f"Task: {task_type}\n"
            f"Prompt:\n{prompt}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Label:\n{label}\n"
        )


    def judge_equal(self, prompt: str, answer: str, label: str, task_type: str) -> bool:
        if torch is None:
            # transformers/torch not available, fail closed (not equal)
            return False
        text = self._build_prompt(prompt, answer, label, task_type)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = decoded[len(text):].strip()
        match = re.search(r'\{\s*"equal"\s*:\s*(true|false)\s*\}', generated, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        return False



class DatasetEvaluator:
    """Main evaluator that can handle multiple datasets and tasks."""
    
    def __init__(self, llama_judge: Optional[LlamaJudge] = None):
        self.llama_judge = llama_judge
    
    def detect_task_type(self, filename: str) -> str:
        """Detect task type from filename."""
        if 'C-STANCE' in filename:
            return 'C-STANCE'
        elif 'FOMC' in filename:
            return 'FOMC'
        elif 'MeetingBank' in filename:
            return 'MeetingBank'
        elif 'Py150' in filename:
            return 'Py150'
        elif 'ScienceQA' in filename:
            return 'ScienceQA'
        elif 'NumGLUE-cm' in filename:
            return 'NumGLUE-cm'
        elif 'NumGLUE-ds' in filename:
            return 'NumGLUE-ds'
        elif '20Minuten' in filename:
            return '20Minuten'
        else:
            raise ValueError(f"Unknown task type in filename: {filename}")
    
    def evaluate_dataset(self, file_path: str) -> Dict[str, Any]:
        """Evaluate a complete dataset."""
        logger.info(f"Starting evaluation of dataset: {file_path}")
        start_time = time.time()
        
        # Parse the data
        logger.info("Parsing results file...")
        parse_start = time.time()
        parsed_data = self._parse_results_file(file_path)
        parse_time = time.time() - parse_start
        logger.info(f"Data parsing completed in {parse_time:.2f} seconds")
        
        # Detect task type
        filename = Path(file_path).name
        task_type = self.detect_task_type(filename)
        evaluator = self.evaluators[task_type]
        
        logger.info(f"Evaluating {task_type} dataset: {filename}")
        logger.info(f"Task: {evaluator.get_task_name()}")
        logger.info(f"Total samples: {len(parsed_data)}")
        
        results = {
            'task_type': task_type,
            'task_name': evaluator.get_task_name(),
            'filename': filename,
            'total_samples': len(parsed_data),
            'correct_samples': 0,
            'incorrect_samples': 0,
            'overall_accuracy': 0.0,
            'detailed_analysis': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_evaluation_time': 0.0
        }
        
        logger.info("Processing classification tasks (with labels)...")
        # These are classification tasks with labels
        results['accuracy_by_label'] = {'A': {'total': 0, 'correct': 0}, 'B': {'total': 0, 'correct': 0}, 'C': {'total': 0, 'correct': 0}}
            
        for i, (prompt, answer, label) in enumerate(parsed_data):
            sample_start = time.time()
                
            if (i + 1) % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(parsed_data)} (Label: {label})...")
                
            # Evaluate this sample (prefer Llama judge if available)
            if self.llama_judge is not None and label:
                logger.debug(f"Using Llama judge for sample {i+1} with label {label}")
                judge_start = time.time()
                label_meaning = evaluator.get_label_meanings().get(label, '')
                label_for_judge = f"{label} ({label_meaning})" if label_meaning else label
                equal = self.llama_judge.judge_equal(prompt, answer, label_for_judge, task_type)
                judge_time = time.time() - judge_start
                logger.debug(f"Llama judge completed in {judge_time:.2f}s")
                    
                evaluation_result = {
                    'label': label,
                    'expected_meaning': evaluator.get_label_meanings().get(label, ''),
                    'is_correct': equal,
                    'score': 1 if equal else 0,
                    'judge': 'llama',
                    'judge_time': judge_time
                }
            else:
                logger.debug(f"Using pattern-based evaluation for sample {i+1} with label {label}")
                logger.info("NO LABEL DETECTED")
                 
            # Update counters
            if evaluation_result['is_correct']:
                results['correct_samples'] += 1
                logger.debug(f"  Sample {i+1}: CORRECT ✓")
            else:
                results['incorrect_samples'] += 1
                logger.debug(f"  Sample {i+1}: INCORRECT ✗")
                
            # Store detailed analysis
            sample_analysis = {
                'sample_id': i + 1,
                'prompt_snippet': prompt[:150] + "..." if len(prompt) > 150 else prompt,
                'answer_snippet': answer[:200] + "..." if len(answer) > 200 else answer,
                'evaluation': evaluation_result
            }
            results['detailed_analysis'].append(sample_analysis)
                
            # Update label-specific accuracy
            if label in results['accuracy_by_label']:
                results['accuracy_by_label'][label]['total'] += 1
                if evaluation_result['is_correct']:
                    results['accuracy_by_label'][label]['correct'] += 1
                
            # Update timing
            sample_time = time.time() - sample_start
            results['total_evaluation_time'] += sample_time
                
            # Progress update every 10 samples
            if (i + 1) % 10 == 0:
                progress = (i + 1) / len(parsed_data) * 100
                avg_time_per_sample = results['total_evaluation_time'] / (i + 1)
                estimated_remaining = avg_time_per_sample * (len(parsed_data) - i - 1)
                logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(parsed_data)}) - Avg: {avg_time_per_sample:.2f}s/sample - ETA: {estimated_remaining:.1f}s")
        
        # Calculate overall accuracy
        if results['total_samples'] > 0:
            results['overall_accuracy'] = (results['correct_samples'] / results['total_samples']) * 100
        
        results['end_time'] = datetime.now().isoformat()
        total_time = time.time() - start_time
        
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        logger.info(f"Total evaluation time: {results['total_evaluation_time']:.2f} seconds")
        logger.info(f"Average time per sample: {results['total_evaluation_time'] / len(parsed_data):.2f} seconds")
        
        return results
    
    def _parse_results_file(self, file_path: str) -> List[Tuple[str, str, str]]:
        """Parse a results JSON file to extract (prompt, answer, label) tuples."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = data.get('prompts', [])
        results = data.get('results', [])
        labels = data.get('labels', [])
        
        # Ensure all arrays have the same length
        min_length = min(len(prompts), len(results), len(labels) if labels else len(results))
        
        parsed_data = []
        for i in range(min_length):
            prompt = prompts[i]
            answer = results[i]
            label = labels[i] if labels and i < len(labels) else ""
            parsed_data.append((prompt, answer, label))
        
        return parsed_data
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of the evaluation results."""
        logger.info("\n" + "="*70)
        logger.info(f"EVALUATION SUMMARY: {results['task_name']}")
        logger.info("="*70)
        logger.info(f"Dataset: {results['filename']}")
        logger.info(f"Task Type: {results['task_type']}")
        logger.info(f"Total Samples: {results['total_samples']}")
        logger.info(f"Correct Samples: {results['correct_samples']}")
        logger.info(f"Incorrect Samples: {results['incorrect_samples']}")
        logger.info(f"Overall Accuracy: {results['overall_accuracy']:.1f}%")
        
        # Show label-specific accuracy for classification tasks
        if 'accuracy_by_label' in results:
            logger.info("\nAccuracy by Label:")
            evaluator = self.evaluators[results['task_type']]
            label_meanings = evaluator.get_label_meanings()
            
            for label in ['A', 'B', 'C']:
                label_stats = results['accuracy_by_label'][label]
                if label_stats['total'] > 0:
                    accuracy = (label_stats['correct'] / label_stats['total']) * 100
                    meaning = label_meanings[label]
                    logger.info(f"  Label {label} ({meaning}): {label_stats['correct']}/{label_stats['total']} = {accuracy:.1f}%")
        
        logger.info("="*70)
    
    def save_detailed_results(self, results: Dict[str, Any], output_file: str):
        """Save detailed evaluation results to a JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to: {output_file}")
    
    def show_incorrect_examples(self, results: Dict[str, Any], max_examples: int = 3):
        """Show examples of incorrect predictions."""
        incorrect_samples = [r for r in results['detailed_analysis'] if not r['evaluation']['is_correct']]
        
        if incorrect_samples:
            logger.info(f"\nExamples of Incorrect Predictions ({len(incorrect_samples)} total):")
            for i, sample in enumerate(incorrect_samples[:max_examples]):
                evaluation = sample['evaluation']
                logger.info(f"\n--- Incorrect Sample {i+1} ---")
                if 'label' in evaluation:
                    logger.info(f"Expected Label: {evaluation['label']} ({evaluation['expected_meaning']})")
                logger.info(f"Prompt: {sample['prompt_snippet']}")
                logger.info(f"Model Answer: {sample['answer_snippet']}")
                if 'label' in evaluation and evaluation['label']:
                    logger.info(f"Expected Answer: {evaluation['label']}")
                logger.info("-" * 60)

def main():
    """Main function to run evaluations on multiple datasets."""
    logger.info("Starting modular evaluation of multiple datasets...")
    evaluator = DatasetEvaluator()
    
    # Define datasets to evaluate
    datasets = [
        "predictions_o-lora/results-7-0-C-STANCE.json",
        "predictions_o-lora/results-7-1-FOMC.json",
        "predictions_o-lora/results-7-2-MeetingBank.json",
        "predictions_o-lora/results-7-3-Py150.json",
        "predictions_o-lora/results-7-4-ScienceQA.json",
        "predictions_o-lora/results-7-5-NumGLUE-cm.json",
        "predictions_o-lora/results-7-6-NumGLUE-ds.json",
        "predictions_o-lora/results-7-7-20Minuten.json"
    ]
    
    all_results = {}
    total_start_time = time.time()
    
    for i, dataset_path in enumerate(datasets):
        if os.path.exists(dataset_path):
            logger.info(f"\n{'='*70}")
            logger.info(f"EVALUATING DATASET {i+1}/{len(datasets)}: {dataset_path}")
            logger.info(f"{'='*70}")
            
            try:
                results = evaluator.evaluate_dataset(dataset_path)
                evaluator.print_evaluation_summary(results)
                
                # Save results
                output_file = f"evaluations/{Path(dataset_path).stem}_evaluation_results.json"
                evaluator.save_detailed_results(results, output_file)
                
                # Show some incorrect examples
                evaluator.show_incorrect_examples(results)
                
                all_results[dataset_path] = results
                
                # Progress update
                elapsed = time.time() - total_start_time
                remaining_datasets = len(datasets) - i - 1
                if remaining_datasets > 0:
                    avg_time_per_dataset = elapsed / (i + 1)
                    estimated_remaining = avg_time_per_dataset * remaining_datasets
                    logger.info(f"Dataset {i+1}/{len(datasets)} completed. Elapsed: {elapsed:.1f}s, ETA for remaining: {estimated_remaining:.1f}s")
                
            except Exception as e:
                logger.error(f"Error evaluating {dataset_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"Dataset not found: {dataset_path}")
    
    # Print overall summary
    if all_results:
        total_time = time.time() - total_start_time
        logger.info(f"\n{'='*70}")
        logger.info("OVERALL EVALUATION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total evaluation time: {total_time:.2f} seconds")
        for dataset_path, results in all_results.items():
            logger.info(f"{Path(dataset_path).name}: {results['overall_accuracy']:.1f}% accuracy")
        logger.info(f"{'='*70}")
    
    logger.info("All evaluations completed!")

if __name__ == "__main__":
    main()