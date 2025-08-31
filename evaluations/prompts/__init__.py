"""
Evaluation prompts package for different datasets.
Import specific prompts or use get_all_prompts() to get all available prompts.
"""

from .base import BASE_EVALUATION_PROMPT
from .datasets import *
from .utils import get_full_prompt

__all__ = [
    'BASE_EVALUATION_PROMPT',
    'PROMPTS_20MINUTEN',
    'PROMPTS_PY150', 
    'PROMPTS_SCIENCEQA',
    'PROMPTS_FOMC',
    'PROMPTS_C_STANCE',
    'get_all_prompts',
    'get_full_prompt'
]
