import json
import os
from typing import List, Tuple, Dict, Any
from pathlib import Path

def parse_results_file(file_path: str) -> List[Tuple[str, str, str]]:
    """
    Parse a results JSON file to extract (prompt, answer, label) tuples.
    
    Args:
        file_path: Path to the JSON results file
        
    Returns:
        List of tuples containing (prompt, answer, label)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = data.get('prompts', [])
    results = data.get('results', [])
    labels = data.get('labels', [])
    
    # Ensure all arrays have the same length
    min_length = min(len(prompts), len(results), len(labels))
    
    parsed_data = []
    for i in range(min_length):
        prompt = prompts[i]
        answer = results[i]
        label = labels[i]
        parsed_data.append((prompt, answer, label))
    
    return parsed_data

def get_parsed_data(file_path: str) -> List[Tuple[str, str, str]]:
    """
    Simple utility function to get parsed data from a results file.
    
    Args:
        file_path: Path to the JSON results file
        
    Returns:
        List of tuples containing (prompt, answer, label)
    """
    return parse_results_file(file_path)

def parse_all_results_in_directory(directory_path: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Parse all results JSON files in a directory.
    
    Args:
        directory_path: Path to directory containing results files
        
    Returns:
        Dictionary mapping filename to parsed data
    """
    results = {}
    directory = Path(directory_path)
    
    for json_file in directory.glob("*.json"):
        try:
            parsed_data = parse_results_file(str(json_file))
            results[json_file.name] = parsed_data
            print(f"Parsed {json_file.name}: {len(parsed_data)} samples")
        except Exception as e:
            print(f"Error parsing {json_file.name}: {e}")
    
    return results

def extract_prompt_components(prompt: str) -> Dict[str, str]:
    """
    Extract components from a Chinese prompt text.
    
    Args:
        prompt: The full prompt text
        
    Returns:
        Dictionary with extracted components
    """
    # The prompts follow a specific format in Chinese
    # We can extract the main text and the target object
    components = {}
    
    # Split by "对象：" to separate the main text from the target
    if "对象：" in prompt:
        parts = prompt.split("对象：")
        if len(parts) >= 2:
            components['main_text'] = parts[0].strip()
            components['target_object'] = parts[1].strip()
    
    # Split by "态度：" to get the instruction part
    if "态度：" in prompt:
        parts = prompt.split("态度：")
        if len(parts) >= 2:
            components['instruction'] = parts[0].strip()
            components['question'] = parts[1].strip()
    
    return components

def main():
    """Main function to demonstrate usage."""
    # Example usage with the specific file mentioned
    file_path = "/cluster/home/lbarinka/trace/predictions_reftcl/results-0-0-C-STANCE.json"
    
    if os.path.exists(file_path):
        print(f"Parsing file: {file_path}")
        parsed_data = parse_results_file(file_path)
        
        print(f"\nTotal samples parsed: {len(parsed_data)}")
        
        # Show first few examples
        print("\nFirst 3 examples:")
        for i, (prompt, answer, label) in enumerate(parsed_data[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Label: {label}")
            
            # Extract components from prompt
            components = extract_prompt_components(prompt)
            if components.get('main_text'):
                print(f"Main text: {components['main_text'][:100]}...")
            if components.get('target_object'):
                print(f"Target object: {components['target_object']}")
            
            print(f"Answer: {answer[:100]}...")
        
        # Count label distribution
        label_counts = {}
        for _, _, label in parsed_data:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nLabel distribution: {label_counts}")
        
    else:
        print(f"File not found: {file_path}")
        print("Please check the file path and try again.")

if __name__ == "__main__":
    main()
