#from transformers import T5ForConditionalGeneration, T5Tokenizer
#import os
#
#save_dir = "initial_model"
#os.makedirs(save_dir, exist_ok=True)
#
## Disable model parallelism checks that can trigger DeepSpeed logic
#import transformers.modeling_utils as modeling_utils
#modeling_utils.is_torch_available = lambda: True
#modeling_utils.unwrap_model = lambda x, **kwargs: x  # Bypass wrapping
#
## Load and save
#model = T5ForConditionalGeneration.from_pretrained("t5-large")
#tokenizer = T5Tokenizer.from_pretrained("t5-large")
#
#model.save_pretrained(save_dir)
#tokenizer.save_pretrained(save_dir)
#import os
#
## Must be set before importing transformers
#os.environ["HF_HOME"] = "/cluster/scratch/tdieudonne/hf_home"
#
#from transformers import AutoModelForCausalLM, AutoTokenizer
#
#save_dir = "/cluster/scratch/tdieudonne/initial_model/llama-7b-hf"
#repo_id = "yahma/llama-7b-hf"
#
#tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
#
#tok.save_pretrained(save_dir)
#model.save_pretrained(save_dir)

import os
import sys
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

REPO_ID = "meta-llama/Llama-2-7b-chat-hf"   # or 'yahma/llama-7b-hf'
SAVE_DIR = "/cluster/scratch/tdieudonne/initial_model/llama-2-7b-chat"


def set_cache_env_if_missing():
    # Prefer scratch for large HF downloads to avoid filling $HOME
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = "/cluster/scratch/tdieudonne/hf_home"
    if not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")


def download_and_save(repo_id: str, save_dir: str):
    print(f"Downloading tokenizer and model from {repo_id} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )

        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving to {save_dir} ...")
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        print("Saved model and tokenizer.")
    except Exception:
        print("Error while downloading/saving model:")
        traceback.print_exc()
        print("\nIf this is a gated/private repo, run 'huggingface-cli login' and try again.")
        sys.exit(1)


def test_generation(local_dir: str):
    print(f"\nLoading from local dir for sanity test: {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print("Generating output...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Generated text:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    set_cache_env_if_missing()
    if not os.path.isdir(SAVE_DIR) or not os.listdir(SAVE_DIR):
        download_and_save(REPO_ID, SAVE_DIR)
    else:
        print(f"Found existing local model at {SAVE_DIR}. Skipping download.")
    test_generation(SAVE_DIR)

