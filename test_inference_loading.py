#!/usr/bin/env python3
"""
Test script to verify REFT-CL inference loading works with the new task-structured files.
"""
import os
import sys
sys.path.append('/cluster/home/tdieudonne/clmm/TRACE')

from reft_inference_loading import load_reft_cl_model_for_inference, load_reft_config_from_saved_model

def test_inference_loading():
    """Test loading a REFT-CL model for inference"""
    
    print("=== Testing REFT-CL Inference Loading ===")
    
    # Paths
    base_model_path = "/cluster/scratch/tdieudonne/initial_model/llama-2-7b-chat"
    saved_model_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_rank8_9layers/2"  # Round 2
    
    print(f"Base model: {base_model_path}")
    print(f"Saved model: {saved_model_path}")
    
    # Check if files exist
    if not os.path.exists(base_model_path):
        print(f"❌ Base model not found: {base_model_path}")
        return False
    
    if not os.path.exists(saved_model_path):
        print(f"❌ Saved model not found: {saved_model_path}")
        return False
    
    # Check for required files
    required_files = [
        "reft_config.json",
        "reftcl_alphas.bin"
    ]
    
    for file_name in required_files:
        file_path = os.path.join(saved_model_path, file_name)
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_name}")
    
    # Check for intervention files
    intervention_files = [f for f in os.listdir(saved_model_path) if f.startswith("intkey_") and f.endswith(".bin")]
    print(f"✅ Found {len(intervention_files)} intervention files")
    
    try:
        # Load config
        print("\n--- Loading Config ---")
        reft_config = load_reft_config_from_saved_model(saved_model_path)
        print(f"Config: {reft_config}")
        
        # Load model
        print("\n--- Loading Model ---")
        model, tokenizer = load_reft_cl_model_for_inference(
            base_model_path=base_model_path,
            saved_model_path=saved_model_path,
            reft_config=reft_config
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        # Test basic forward pass
        print("\n--- Testing Forward Pass ---")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use pyreft's forward API
            base_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
            outputs = model.forward(base_inputs)
            print(f"✅ Forward pass successful!")
            print(f"Output shape: {outputs.logits.shape}")
        
        # Test task activation
        print("\n--- Testing Task Activation ---")
        base_ref = model.module if hasattr(model, "module") else model
        for key, intervention in getattr(base_ref, "interventions", {}).items():
            if hasattr(intervention, "set_active_tasks"):
                print(f"Intervention {key}: current active_tasks = {intervention.active_tasks}")
                intervention.set_active_tasks(2)  # Activate 2 tasks
                print(f"Intervention {key}: new active_tasks = {intervention.active_tasks}")
        
        print(f"\n🎉 All tests passed! REFT-CL inference loading is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch
    success = test_inference_loading()
    if success:
        print(f"\n✅ REFT-CL inference loading test PASSED")
    else:
        print(f"\n❌ REFT-CL inference loading test FAILED")
