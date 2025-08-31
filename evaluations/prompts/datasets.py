"""
Dataset-specific evaluation prompts with relevant examples.
"""

# 20Minuten - German text simplification
PROMPTS_20MINUTEN = {
    "system": "You are evaluating whether an LLM's answer matches a given label for German text simplification tasks.",
    "instruction": "Determine if the LLM's simplified German text is semantically equivalent to the expected label. Consider different phrasings, synonyms, proper German grammar, and whether the simplification maintains the core meaning.",
    "input_format": {
        "prompt": "the original German text to be simplified",
        "answer": "the LLM's simplified version", 
        "label": "the expected simplified version"
    },
    "examples": [
        {
            "input": {
                "prompt": "Provide a simplified version of the following paragraph in German.\n\nParagraph:\nReto M. hat gestern Abend in der Stadt ein neues Restaurant entdeckt, das italienische Küche anbietet.",
                "answer": "Reto M. fand gestern ein neues italienisches Restaurant in der Stadt.",
                "label": "Reto M. entdeckte gestern ein italienisches Restaurant in der Stadt."
            },
            "output": {"equal": True}
        },
        {
            "input": {
                "prompt": "Vereinfache den folgenden Satz: 'Die Implementierung der neuen Technologie führte zu einer signifikanten Verbesserung der Effizienz.'",
                "answer": "Die neue Technologie machte alles effizienter.",
                "label": "Die neue Technologie verbesserte die Effizienz."
            },
            "output": {"equal": True}
        }
    ],
    "output_format": "Return ONLY a JSON object with this exact format: {\"equal\": true} or {\"equal\": false}",
    "constraint": "Do not include any other text, explanations, or formatting. Just the JSON object."
}

# Py150 - Python code completion
PROMPTS_PY150 = {
    "system": "You are evaluating whether an LLM's answer matches a given label for Python code completion tasks.",
    "instruction": "Determine if the LLM's completed Python code is functionally equivalent to the expected label. Consider different variable names, formatting, and coding styles as long as the logic and functionality are identical.",
    "input_format": {
        "prompt": "the incomplete Python code or context",
        "answer": "the LLM's completed code", 
        "label": "the expected completed code"
    },
    "examples": [
        {
            "input": {
                "prompt": "Complete the function:\ndef calculate_sum(numbers):\n    # TODO: implement sum calculation",
                "answer": "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total",
                "label": "def calculate_sum(numbers):\n    return sum(numbers)"
            },
            "output": {"equal": True}
        },
        {
            "input": {
                "prompt": "Complete: if x > 0:",
                "answer": "if x > 0:\n    print('positive')",
                "label": "if x > 0:\n    return True"
            },
            "output": {"equal": False}
        }
    ],
    "output_format": "Return ONLY a JSON object with this exact format: {\"equal\": true} or {\"equal\": false}",
    "constraint": "Do not include any other text, explanations, or formatting. Just the JSON object."
}

# ScienceQA - Science question answering
PROMPTS_SCIENCEQA = {
    "system": "You are evaluating whether an LLM's answer matches a given label for science question answering tasks.",
    "instruction": "Determine if the LLM's answer is semantically equivalent to the expected label. Consider different ways of expressing the same scientific concept, accuracy of scientific facts, and completeness of the response.",
    "input_format": {
        "prompt": "the science question",
        "answer": "the LLM's response", 
        "label": "the expected answer"
    },
    "examples": [
        {
            "input": {
                "prompt": "What is the chemical symbol for gold?",
                "answer": "The chemical symbol for gold is Au.",
                "label": "Au"
            },
            "output": {"equal": True}
        },
        {
            "input": {
                "prompt": "What causes rain?",
                "answer": "Rain is caused by water vapor condensing in clouds and falling due to gravity.",
                "label": "Water vapor condenses in clouds and falls as precipitation."
            },
            "output": {"equal": True}
        }
    ],
    "output_format": "Return ONLY a JSON object with this exact format: {\"equal\": true} or {\"equal\": false}",
    "constraint": "Do not include any other text, explanations, or formatting. Just the JSON object."
}

# FOMC - Federal Reserve monetary policy stance
PROMPTS_FOMC = {
    "system": "You are evaluating whether an LLM's answer matches a given label for Federal Reserve monetary policy stance classification tasks.",
    "instruction": "Determine if the LLM's classification of the FOMC stance is semantically equivalent to the expected label. Consider different ways of expressing the same monetary policy position.",
    "input_format": {
        "prompt": "the FOMC statement or context",
        "answer": "the LLM's stance classification", 
        "label": "the expected stance classification"
    },
    "examples": [
        {
            "input": {
                "prompt": "The FOMC statement indicates a dovish stance with emphasis on supporting economic recovery.",
                "answer": "Dovish",
                "label": "dovish"
            },
            "output": {"equal": True}
        },
        {
            "input": {
                "prompt": "The Committee decided to raise the target range for the federal funds rate.",
                "answer": "Hawkish monetary policy",
                "label": "Hawkish"
            },
            "output": {"equal": True}
        }
    ],
    "output_format": "Return ONLY a JSON object with this exact format: {\"equal\": true} or {\"equal\": false}",
    "constraint": "Do not include any other text, explanations, or formatting. Just the JSON object."
}

# C-STANCE - Chinese stance detection
PROMPTS_C_STANCE = {
    "system": "You are evaluating whether an LLM's answer matches a given label for Chinese stance detection tasks.",
    "instruction": "Determine if the LLM's stance classification is semantically equivalent to the expected label. Consider different ways of expressing the same stance in Chinese, including synonyms and paraphrasing.",
    "input_format": {
        "prompt": "the Chinese text or context",
        "answer": "the LLM's stance classification", 
        "label": "the expected stance classification"
    },
    "examples": [
        {
            "input": {
                "prompt": "这段文字表达了对新政策的支持态度。",
                "answer": "支持",
                "label": "支持"
            },
            "output": {"equal": True}
        },
        {
            "input": {
                "prompt": "作者对新规定持反对意见。",
                "answer": "反对",
                "label": "反对"
            },
            "output": {"equal": True}
        }
    ],
    "output_format": "Return ONLY a JSON object with this exact format: {\"equal\": true} or {\"equal\": false}",
    "constraint": "Do not include any other text, explanations, or formatting. Just the JSON object."
}


def get_all_prompts():
    """Get all available dataset prompts as a dictionary."""
    return {
        '20minuten': PROMPTS_20MINUTEN,
        'py150': PROMPTS_PY150,
        'scienceqa': PROMPTS_SCIENCEQA,
        'fomc': PROMPTS_FOMC,
        'c_stance': PROMPTS_C_STANCE
    }
