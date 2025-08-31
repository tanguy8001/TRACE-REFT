"""
Run modular evaluation on multiple RefT datasets.
"""

import os
import sys
import argparse
from pathlib import Path

# Prepare paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
evaluations_dir = project_root / "evaluations"

# Ensure evaluations can be imported
if str(evaluations_dir) not in sys.path:
    sys.path.insert(0, str(evaluations_dir))

from modular_evaluator import DatasetEvaluator, LlamaJudge

def main():
    parser = argparse.ArgumentParser(description="Run modular evaluation on RefT datasets")
    parser.add_argument("--datasets", "-d", nargs="+", 
                       default=["predictions_reftcl/results-1-0-C-STANCE.json", 
                               "predictions_reftcl/results-1-1-FOMC.json"],
                       help="Paths to datasets to evaluate")
    parser.add_argument("--output-dir", "-o", type=str, default="evaluations",
                       help="Output directory for results")
    parser.add_argument("--judge-llama", action="store_true",
                       help="Use a Llama model as judge for correctness of classification tasks")
    parser.add_argument("--llama-model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                       help="HF model name to load for the Llama judge")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    llama_judge = None
    if args.judge_llama:
        try:
            llama_judge = LlamaJudge(model_name=args.llama_model)
        except Exception as e:
            print(f"Failed to initialize Llama judge: {e}")
            llama_judge = None

    evaluator = DatasetEvaluator(llama_judge=llama_judge)
    all_results = {}
    
    for dataset_path in args.datasets:
        if os.path.exists(dataset_path):
            print(f"\n{'='*70}")
            print(f"EVALUATING DATASET: {dataset_path}")
            print(f"{'='*70}")
            
            try:
                results = evaluator.evaluate_dataset(dataset_path)
                evaluator.print_evaluation_summary(results)
                
                # Save results
                output_file = os.path.join(args.output_dir, f"{Path(dataset_path).stem}_evaluation_results.json")
                evaluator.save_detailed_results(results, output_file)
                
                # Show some incorrect examples
                evaluator.show_incorrect_examples(results)
                
                all_results[dataset_path] = results
                
            except Exception as e:
                print(f"Error evaluating {dataset_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Dataset not found: {dataset_path}")
    
    # Print overall summary
    if all_results:
        print(f"\n{'='*70}")
        print("OVERALL EVALUATION SUMMARY")
        print(f"{'='*70}")
        for dataset_path, results in all_results.items():
            print(f"{Path(dataset_path).name}: {results['overall_accuracy']:.1f}% accuracy")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()
