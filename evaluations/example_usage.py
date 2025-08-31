#!/usr/bin/env python3
"""
Example usage of the parse_results module.
"""

from parse_results import get_parsed_data, extract_prompt_components

def main():
    # Parse the specific file mentioned
    file_path = "/cluster/home/lbarinka/trace/predictions_reftcl/results-0-0-C-STANCE.json"
    
    # Get the parsed data
    parsed_data = get_parsed_data(file_path)
    
    print(f"Successfully parsed {len(parsed_data)} samples")
    
    # Example: Process the first few samples
    for i, (prompt, answer, label) in enumerate(parsed_data[:5]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Ground Truth Label: {label}")
        
        # Extract components from the prompt
        components = extract_prompt_components(prompt)
        
        if components.get('target_object'):
            print(f"Target Object: {components['target_object']}")
        
        # Show a snippet of the answer
        answer_snippet = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"Model Answer: {answer_snippet}")
        
        print("-" * 50)
    
    # Example: Count labels
    label_counts = {}
    for _, _, label in parsed_data:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabel Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples ({count/len(parsed_data)*100:.1f}%)")

if __name__ == "__main__":
    main()
