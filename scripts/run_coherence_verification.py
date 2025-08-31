#!/usr/bin/env python3
"""
Run Llama-based coherence verification.
This script loads the requested Llama model and runs verification using evaluations/verify_coherence.py.
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

from verify_coherence import LlamaCoherenceVerifier

def main():
    parser = argparse.ArgumentParser(description="Run Llama coherence verification")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON file with predictions")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output JSON file for results")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Hugging Face model name")
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load and run
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    try:
        verifier = LlamaCoherenceVerifier(model_name=args.model)
        results = verifier.analyze_all_samples(args.input)
        verifier.print_coherence_summary(results)
        verifier.save_detailed_results(results, args.output)
        verifier.show_incoherent_examples(results, max_examples=3)
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
