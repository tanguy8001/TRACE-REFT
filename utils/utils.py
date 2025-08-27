# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    """Load a tokenizer with a robust fallback from fast to slow implementations.

    - Prefer fast tokenizers when available
    - Fall back to slow (SentencePiece-based) tokenizers on parsing errors
    - Normalize llama detection to be case-insensitive
    """
    use_fast = bool(fast_tokenizer)
    tokenizer = None

    try:
        if "llama" in model_name_or_path.lower():
            # Let AutoTokenizer resolve to the correct LLaMA variant first
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, use_fast=use_fast, trust_remote_code=True
                )
            except Exception:
                # Fallback to slow LLaMA tokenizer
                from transformers.models.llama import LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained(
                    model_name_or_path, use_fast=False
                )
            # Ensure padding token
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token or tokenizer.eos_token})
            tokenizer.padding_side = 'left'
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, use_fast=use_fast, trust_remote_code=True
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, use_fast=False, trust_remote_code=True
                )
            # Ensure reasonable defaults
            if getattr(tokenizer, 'pad_token', None) is None and getattr(tokenizer, 'eos_token', None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            if getattr(tokenizer, 'bos_token', None) is None and getattr(tokenizer, 'eos_token', None) is not None:
                tokenizer.bos_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
    except Exception:
        # Last-resort generic fallback
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, trust_remote_code=True
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    tokenizer.truncation_side = "left"
    return tokenizer


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    # for key in list(save_dict.keys()):
    #     if "lora" in key:
    #         del save_dict[key]
    torch.save(save_dict, output_model_file)
    # Write config with robust fallback to handle non-JSON-serializable objects
    try:
        model_to_save.config.to_json_file(output_config_file)
    except Exception:
        import json
        def _safe(obj):
            try:
                import torch as _torch  # local import to avoid polluting namespace
                if isinstance(obj, (str, int, float, bool)) or obj is None:
                    return obj
                if isinstance(obj, _torch.dtype):
                    return str(obj)
            except Exception:
                pass
            if isinstance(obj, dict):
                return {k: _safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_safe(v) for v in obj]
            # Fallback: stringify unknown objects/types
            try:
                return str(obj)
            except Exception:
                return None
        cfg_dict = getattr(model_to_save, 'config', None)
        cfg_dict = cfg_dict.to_dict() if hasattr(cfg_dict, 'to_dict') else {}
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(_safe(cfg_dict), f, indent=2)
    # Tokenizer save
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
    alpha_lr=None,
):
    # Recognize alpha params both before and after wrappers (e.g., DeepSpeed), which prepend 'module.'
    alpha_name_prefixes = ("reftcl_alpha_bank.alphas", "module.reftcl_alpha_bank.alphas")
    def _is_alpha_name(name: str) -> bool:
        try:
            return any(name.startswith(prefix) for prefix in alpha_name_prefixes)
        except Exception:
            return False
    # Debug: inspect how parameters are named before grouping
    try:
        all_named_params = list(model.named_parameters())
        total_params = len(all_named_params)
        total_trainable = sum(1 for _, p in all_named_params if getattr(p, 'requires_grad', False))
        alpha_like = [n for n, _ in all_named_params if ("alpha" in n.lower() or "reft" in n.lower())]
        trainable_head = [n for n, p in all_named_params if getattr(p, 'requires_grad', False)][:30]
        print(f"[DEBUG] named_parameters: total={total_params}, trainable={total_trainable}")
        if alpha_like:
            preview = ", ".join(alpha_like[:20]) + (" ..." if len(alpha_like) > 20 else "")
            print(f"[DEBUG] params containing 'alpha' or 'reft' ({len(alpha_like)}): {preview}")
        else:
            print("[DEBUG] No params containing 'alpha' or 'reft' found in names.")
        if trainable_head:
            print(f"[DEBUG] first trainable params: {', '.join(trainable_head)}")
        # Try to introspect the alpha bank directly if present
        alpha_bank = getattr(model, 'reftcl_alpha_bank', None)
        if alpha_bank is not None and hasattr(alpha_bank, 'alphas'):
            try:
                alpha_flags = [p.requires_grad for p in alpha_bank.alphas]
                expected_names = [f"reftcl_alpha_bank.alphas.{i}" for i, _ in enumerate(alpha_bank.alphas)]
                print(f"[DEBUG] reftcl_alpha_bank present with {len(alpha_bank.alphas)} alphas; requires_grad={alpha_flags}")
                print(f"[DEBUG] expected alpha param names head: {', '.join(expected_names[:10])}{' ...' if len(expected_names) > 10 else ''}")
            except Exception as e:
                print(f"[DEBUG] reftcl_alpha_bank introspection error: {e}")
        else:
            print("[DEBUG] Model has no attribute 'reftcl_alpha_bank' (yet).")
    except Exception as _dbg_e:
        print(f"[DEBUG] Error while inspecting named_parameters: {_dbg_e}")

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad
                    and not any(nd in n for nd in lora_name_list)
                    and not _is_alpha_name(n))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad
                    and not _is_alpha_name(n))
            ],
            "weight_decay":
            0.0,
        },
    ]


    alpha_params = [
        p for n, p in model.named_parameters()
        if _is_alpha_name(n) and p.requires_grad
    ]
    # Fallback: if alpha params are hidden by a wrapper's named_parameters, pull directly from the alpha bank
    if not alpha_params:
        try:
            alpha_bank = getattr(model, 'reftcl_alpha_bank', None)
            if alpha_bank is not None and hasattr(alpha_bank, 'alphas'):
                fallback = [p for p in alpha_bank.alphas if getattr(p, 'requires_grad', False)]
                if fallback:
                    alpha_params = fallback
                    print(f"[DEBUG] Using alpha_bank fallback; collected {len(alpha_params)} alpha params for optimizer grouping.")
        except Exception as _alpha_fb_err:
            print(f"[DEBUG] alpha_bank fallback failed: {_alpha_fb_err}")
    # Debug: show how alpha params were resolved for grouping
    try:
        debug_alpha_names = [n for n, p in model.named_parameters() if _is_alpha_name(n)]
        print(f"Alpha params (requires_grad=True) count: {len(alpha_params)}")
        if debug_alpha_names:
            print(f"[DEBUG] all alpha param names: {', '.join(debug_alpha_names)}")
        else:
            print("[DEBUG] No parameters matched alpha prefixes via startswith.")
    except Exception:
        pass
    if alpha_params:
        optimizer_grouped_parameters.append({
            "params": alpha_params,
            "weight_decay": 0.0,
            "lr": alpha_lr,
        })
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)

    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0, sub_folder=""):
    zero_stage_3 = (zero_stage == 3)
    save_dir = os.path.join(save_dir, sub_folder)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        # 问题出在这里，不会保存不在named_parameters()里的参数，若是 model_to_save.state_dict() 则 OK
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict
