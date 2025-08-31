"""
Utility functions for working with evaluation prompts.
"""

def format_prompt_for_llm(prompt_dict: dict, include_examples: bool = True) -> str:
    """
    Convert a prompt dictionary into a formatted text prompt for LLM input.
    
    Args:
        prompt_dict: Dictionary containing prompt components
        include_examples: Whether to include examples in the prompt
    
    Returns:
        Formatted prompt string ready for LLM input
    """
    lines = []
    
    # System message
    lines.append(prompt_dict["system"])
    lines.append("")
    
    # Instruction
    lines.append(prompt_dict["instruction"])
    lines.append("")
    
    # Input format
    lines.append("Given the following input as a JSON:")
    lines.append("{")
    for key, description in prompt_dict["input_format"].items():
        lines.append(f'  "{key}": "{description}"')
    lines.append("}")
    lines.append("")
    
    # Examples
    if include_examples and "examples" in prompt_dict:
        lines.append("Examples:")
        for i, example in enumerate(prompt_dict["examples"]):
            lines.append(f"Input: {example['input']}")
            lines.append(f"Output: {example['output']}")
            if i < len(prompt_dict["examples"]) - 1:
                lines.append("")
        lines.append("")
    
    # Output format
    lines.append(prompt_dict["output_format"])
    lines.append("")
    
    # Constraint
    lines.append(prompt_dict["constraint"])
    
    return "\n".join(lines)


def format_input_for_prompt(prompt: str, answer: str, label: str) -> str:
    """
    Format the input data into the JSON structure expected by the prompt.
    
    Args:
        prompt: The original question/task
        answer: The LLM's response
        label: The expected/correct answer
    
    Returns:
        Formatted JSON string
    """
    import json
    return json.dumps({
        "prompt": prompt,
        "answer": answer,
        "label": label
    }, ensure_ascii=False)


def get_full_prompt(dataset_name: str, prompt: str, answer: str, label: str, include_examples: bool = True) -> str:
    """
    Get a complete formatted prompt for a specific dataset and input data.
    
    Args:
        dataset_name: Name of the dataset (e.g., '20minuten', 'py150')
        prompt: The original question/task
        answer: The LLM's response
        label: The expected/correct answer
        include_examples: Whether to include examples in the prompt
    
    Returns:
        Complete formatted prompt string ready for LLM input
    """
    # Import here to avoid circular imports
    from .datasets import get_all_prompts
    
    all_prompts = get_all_prompts()
    if dataset_name not in all_prompts:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(all_prompts.keys())}")
    
    prompt_dict = all_prompts[dataset_name]
    formatted_prompt = format_prompt_for_llm(prompt_dict, include_examples)
    
    # Add the specific input data
    input_data = format_input_for_prompt(prompt, answer, label)
    
    return f"{formatted_prompt}\n\nInput: {input_data}"
