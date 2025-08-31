#!/usr/bin/env python3
"""
Evaluate LLM predictions using Llama 3B model to verify answer correspondence with labels.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from parse_results import get_parsed_data, extract_prompt_components
import re
from typing import List, Tuple, Dict, Any
import numpy as np

class LlamaEvaluator:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the Llama evaluator.
        
        Args:
            model_name: Hugging Face model name for Llama
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def extract_label_from_answer(self, answer: str) -> str:
        """
        Extract the predicted label (A, B, or C) from the model's answer.
        
        Args:
            answer: The model's generated answer
            
        Returns:
            Extracted label (A, B, or C) or "UNKNOWN" if not found
        """
        # Look for patterns like "A.", "B.", "C." or "A", "B", "C"
        patterns = [
            r'[ABC]\.',  # A., B., C.
            r'[ABC]\s',  # A , B , C 
            r'[ABC]$',   # A, B, C at end
            r'选择\s*([ABC])',  # 选择 A, 选择 B, 选择 C
            r'答案是\s*([ABC])',  # 答案是 A, 答案是 B, 答案是 C
            r'([ABC])\s*[：:]\s*(支持|反对|中立)',  # A: 支持, B: 反对, C: 中立
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                if len(match.group()) == 1:
                    return match.group()
                else:
                    return match.group(1)
        
        # If no pattern found, try to find A, B, C in the text
        if 'A' in answer and 'B' not in answer and 'C' not in answer:
            return 'A'
        elif 'B' in answer and 'A' not in answer and 'C' not in answer:
            return 'B'
        elif 'C' in answer and 'A' not in answer and 'B' not in answer:
            return 'C'
        
        return "UNKNOWN"
    
    def evaluate_predictions(self, file_path: str) -> Dict[str, Any]:
        """
        Evaluate LLM predictions against ground truth labels.
        
        Args:
            file_path: Path to the JSON results file
            
        Returns:
            Dictionary containing evaluation results
        """
        # Parse the data
        parsed_data = get_parsed_data(file_path)
        
        results = {
            'total_samples': len(parsed_data),
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'unknown_predictions': 0,
            'accuracy': 0.0,
            'detailed_results': [],
            'label_accuracy': {'A': {'correct': 0, 'total': 0}, 'B': {'correct': 0, 'total': 0}, 'C': {'correct': 0, 'total': 0}}
        }
        
        print(f"Evaluating {len(parsed_data)} samples...")
        
        for i, (prompt, answer, true_label) in enumerate(parsed_data):
            # Extract predicted label from the answer
            predicted_label = self.extract_label_from_answer(answer)
            
            # Check if prediction is correct
            is_correct = predicted_label == true_label
            
            # Update counters
            if predicted_label == "UNKNOWN":
                results['unknown_predictions'] += 1
            elif is_correct:
                results['correct_predictions'] += 1
            else:
                results['incorrect_predictions'] += 1
            
            # Update label-specific accuracy
            if true_label in results['label_accuracy']:
                results['label_accuracy'][true_label]['total'] += 1
                if is_correct:
                    results['label_accuracy'][true_label]['correct'] += 1
            
            # Store detailed results
            sample_result = {
                'sample_id': i + 1,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'is_correct': is_correct,
                'prompt_snippet': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'answer_snippet': answer[:100] + "..." if len(answer) > 100 else answer
            }
            results['detailed_results'].append(sample_result)
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(parsed_data)} samples...")
        
        # Calculate overall accuracy
        valid_predictions = results['correct_predictions'] + results['incorrect_predictions']
        if valid_predictions > 0:
            results['accuracy'] = results['correct_predictions'] / valid_predictions
        
        # Calculate label-specific accuracy
        for label in results['label_accuracy']:
            if results['label_accuracy'][label]['total'] > 0:
                results['label_accuracy'][label]['accuracy'] = (
                    results['label_accuracy'][label]['correct'] / 
                    results['label_accuracy'][label]['total']
                )
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """
        Print a summary of the evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Samples: {results['total_samples']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        print(f"Incorrect Predictions: {results['incorrect_predictions']}")
        print(f"Unknown Predictions: {results['unknown_predictions']}")
        print(f"Overall Accuracy: {results['accuracy']:.2%}")
        
        print("\nLabel-specific Accuracy:")
        for label, stats in results['label_accuracy'].items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                print(f"  Label {label}: {stats['correct']}/{stats['total']} = {accuracy:.2%}")
        
        print("\n" + "="*60)
    
    def save_detailed_results(self, results: Dict[str, Any], output_file: str):
        """
        Save detailed evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Detailed results saved to: {output_file}")

def main():
    """Main function to run the evaluation."""
    # Initialize the evaluator
    print("Initializing Llama evaluator...")
    
    # You can change the model name here if needed
    # For Llama 3B, you might use: "meta-llama/Llama-2-3b-chat-hf" or similar
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change this to 3B if needed
    
    try:
        evaluator = LlamaEvaluator(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if you have access to the model and sufficient memory.")
        return
    
    # Evaluate the predictions
    file_path = "/cluster/home/lbarinka/trace/predictions_reftcl/results-0-0-C-STANCE.json"
    
    print(f"\nEvaluating file: {file_path}")
    results = evaluator.evaluate_predictions(file_path)
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    # Save detailed results
    output_file = "evaluations/llm_evaluation_results.json"
    evaluator.save_detailed_results(results, output_file)
    
    # Show some examples of incorrect predictions
    incorrect_samples = [r for r in results['detailed_results'] if not r['is_correct'] and r['predicted_label'] != "UNKNOWN"]
    
    if incorrect_samples:
        print(f"\nExamples of incorrect predictions ({len(incorrect_samples)} total):")
        for i, sample in enumerate(incorrect_samples[:3]):  # Show first 3
            print(f"\n--- Incorrect Sample {i+1} ---")
            print(f"True Label: {sample['true_label']}")
            print(f"Predicted: {sample['predicted_label']}")
            print(f"Prompt: {sample['prompt_snippet']}")
            print(f"Answer: {sample['answer_snippet']}")

if __name__ == "__main__":
    main()
