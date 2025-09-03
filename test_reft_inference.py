#!/usr/bin/env python3
"""
Test REFT-CL inference loading and generation.
"""
import torch
import os
import sys
from transformers import AutoTokenizer

# Add TRACE to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from reft_inference_loading import load_reft_cl_model_for_inference, load_reft_config_from_saved_model

def test_reft_inference():
    print("=== Testing REFT-CL Inference ===")
    
    # Test paths
    base_model_path = "/cluster/scratch/tdieudonne/initial_model/llama-2-7b-chat"
    saved_model_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL/7"  # Round 7
    
    print(f"Base model: {base_model_path}")
    print(f"Saved model: {saved_model_path}")
    
    # Check if paths exist
    if not os.path.exists(base_model_path):
        print(f"❌ Base model path not found: {base_model_path}")
        return False
    if not os.path.exists(saved_model_path):
        print(f"❌ Saved model path not found: {saved_model_path}")
        return False
    
    # Load tokenizer
    print("\n--- Loading Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token or tokenizer.eos_token})
    tokenizer.padding_side = 'left'
    print("✅ Tokenizer loaded")
    
    # Load config
    print("\n--- Loading Config ---")
    reft_config = load_reft_config_from_saved_model(saved_model_path)
    print(f"Config: {reft_config}")
    
    # Load model
    print("\n--- Loading Model ---")
    try:
        model, tokenizer = load_reft_cl_model_for_inference(
            base_model_path=base_model_path,
            saved_model_path=saved_model_path,
            reft_config=reft_config,
            tokenizer=tokenizer
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Inputs moved to device: {device}")
    
    try:
        with torch.no_grad():
            # Use pyreft's forward API
            base_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
            outputs = model.forward(base_inputs)
            print(f"✅ Forward pass successful!")
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                print(f"Output shape: {outputs.logits.shape}")
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                print(f"Output type: tuple with {len(outputs)} elements")
                if hasattr(outputs[0], 'logits'):
                    print(f"Output shape: {outputs[0].logits.shape}")
                else:
                    print(f"First output type: {type(outputs[0])}")
            else:
                print(f"Output type: {type(outputs)}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test generation
    print("\n--- Testing Generation ---")
    try:
        with torch.no_grad():
            # Use pyreft's generation API
            base_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
            
            # Set up unit_locations for interventions
            seq_len = inputs["input_ids"].shape[1]
            num_interventions = len(model.interventions)
            unit_locations = [[list(range(seq_len))] for _ in range(num_interventions)]
            
            generation_args = {
                "base": base_inputs,
                "unit_locations": {"sources->base": (None, unit_locations)},
                "intervene_on_prompt": True,
                "max_new_tokens": 50,
                "temperature": 0.1,
                "do_sample": True,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            _, generated_ids = model.generate(**generation_args)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"✅ Generation successful!")
            print(f"Generated text: {generated_text}")
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test task activation
    print("\n--- Testing Task Activation ---")
    try:
        base_ref = model.module if hasattr(model, "module") else model
        for key, intervention in getattr(base_ref, "interventions", {}).items():
            if hasattr(intervention, "set_active_tasks"):
                intervention.set_active_tasks(1)  # Activate task 0
                print(f"✅ Set active tasks for {key} to 1")
                if intervention.active_tasks == 1:
                    print(f"  Current active tasks: {intervention.active_tasks}")
                else:
                    print(f"❌ Failed to set active tasks for {key}")
                    return False
    except Exception as e:
        print(f"❌ Task activation failed: {e}")
        return False
    
    print("\n✅ All tests passed! REFT-CL inference is working correctly.")
    return True

if __name__ == "__main__":
    if test_reft_inference():
        print("\n🎉 REFT-CL inference test PASSED")
    else:
        print("\n💥 REFT-CL inference test FAILED")
