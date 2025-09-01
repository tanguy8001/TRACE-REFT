import os
import sys
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

REPO_ID ="meta-llama/Llama-2-7b-chat-hf"
SAVE_DIR = "/cluster/scratch/tdieudonne/initial_model/llama-2-7b-chat-hf"


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
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    instruction = (
            "Extract the final numeric answer ONLY from the text.\n"
            "Rules:\n"
            "- If the answer is a number, return it exactly as written (including decimals).\n"
            "- If the final answer is not numeric but a single-word label (e.g., a month name), return that single word exactly.\n"
            "- Return ONLY a JSON object: {\"answer\": \"<VALUE>\"}.\n"
            "- Do NOT include units or explanations.\n"
    )
    answer_text = "there are 3 frogs in the pond. How many frog eyes should he expect in the pond?\nAnswer:\nAxel should expect 6 frog eyes in the pond.\n\nExplanation:\nEach frog has 2 eyes, so there are 2 eyes per frog. Since there are 3 frogs in the pond, there are 3 x 2 = 6 eyes in the pond."
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{instruction}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Text to parse:
{answer_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print("Generating output...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Generated text:")
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_start_tag = "<|start_header_id|>assistant<|end_header_id|>"
    response_start_index = full_output.rfind(assistant_start_tag)
    if response_start_index != -1:
        # Add the length of the tag to get the index of the first token of the response
        response_text = full_output[response_start_index + len(assistant_start_tag):].strip()
        
        # Remove the <|eot_id|> token
        response_text = response_text.replace("<|eot_id|>", "").strip() # <-- This is the new line
        
        print("Generated JSON:")
        print(response_text)
    else:
        print("Could not find assistant tag in output.")
        print(full_output)

if __name__ == "__main__":
    set_cache_env_if_missing()
    test_generation(SAVE_DIR)

