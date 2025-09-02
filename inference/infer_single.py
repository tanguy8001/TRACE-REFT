"""
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
"""

# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import deepspeed
import json
from typing import Dict, List, Tuple, Optional

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten # to be continued
from training.params import Method2Class, AllDatasetName

from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model

# dist.init_process_group(backend='nccl')


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset. A single data path.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--inference_model_path",
        type=str,
        help=
        "Path to inference model.",
        required=True,
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    # inference params
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generate temperature params.",
    )

    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    # TODO, add other inference params
    parser.add_argument(
        "--inference_tasks",
        type=list_of_strings,
        default='all',
        help='Datasets to be used.'
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")

    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # added by wangxiao
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    parser.add_argument('--CL_method',
            default=None,
            help='continual learning method used')
    # REFT-CL specific knobs (align with training)
    parser.add_argument('--reft_layers', type=str, default='3;9;18;24', help='Layers to intervene (e.g., 3;9;18;24 or all)')
    parser.add_argument('--reft_rank', type=int, default=4, help='Low-rank dimension r for REFT')
    parser.add_argument('--reft_eps', type=float, default=1e-6, help='Epsilon for direction normalization')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda")


    def _is_intervenable(model_obj) -> bool:
        underlying = model_obj.module if hasattr(model_obj, 'module') else model_obj
        return hasattr(underlying, 'interventions')

    def _pick_first_tensor(container):
        if torch.is_tensor(container):
            return container
        if isinstance(container, (tuple, list)):
            # Prefer non-None tensor with 2+ dims
            candidates = [t for t in container if torch.is_tensor(t)]
            if candidates:
                candidates.sort(key=lambda t: (t.ndim, t.numel()), reverse=True)
                return candidates[0]
        # Try attributes commonly used
        for attr in ("sequences", "ids", "output", "generated_ids"):
            val = getattr(container, attr, None)
            if torch.is_tensor(val):
                return val
        return None

    def _generate_with_model(model_obj, tokenizer, batch, args):
        # Returns a torch.Tensor of generated ids
        gen_kwargs = dict(
            max_new_tokens=args.max_ans_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.unk_token_id,
            temperature=args.temperature,
            do_sample=True,
            num_return_sequences=1,
            use_cache=True,
        )
        if _is_intervenable(model_obj):
            base_inputs = {"input_ids": batch["input_ids"], "attention_mask": batch.get("attention_mask")}
            out = model_obj.generate(base_inputs, **gen_kwargs)
        else:
            out = model_obj.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], **gen_kwargs)
        gen_ids = _pick_first_tensor(out)
        if gen_ids is None:
            # Minimal debug dump
            msg = f"[DEBUG] Unexpected generate() output type: {type(out).__name__}"
            try:
                if isinstance(out, (tuple, list)):
                    shapes = []
                    for t in out:
                        if torch.is_tensor(t):
                            shapes.append(f"Tensor{tuple(t.shape)}:{t.dtype}")
                        else:
                            shapes.append(type(t).__name__)
                    msg += " | elements: " + ", ".join(shapes[:6])
            except Exception:
                pass
            print_rank_0(msg, 0)
            raise RuntimeError("Could not extract generated ids from model output")
        return gen_ids

    def prediction(model, infer_dataloader):
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()
        for step, batch in enumerate(infer_dataloader):
            # TODO, add prompts, choosen, rejected
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            sources_sequences += batch['sources']
            ground_truths += batch['gts']
            del batch['sources']
            del batch['gts']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]
            # update progress bar
            progress_bar.update(1)
            description = f"Step {step}"
            progress_bar.set_description(description, refresh=False)
            with torch.no_grad():
                generate_ids = _generate_with_model(model, tokenizer, batch, args)
            sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            predicted_sequences += sequences
        return sources_sequences, predicted_sequences, ground_truths

    def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                ground_truths: list, round: int, i_task: int, task: str):
        # save as a json file
        df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
                'labels': ground_truths}
        if not os.path.exists(args.inference_output_path):
            os.makedirs(args.inference_output_path)
        with open(args.inference_output_path + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w+", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)


    # ===== Helpers for summary/averaging & BWT =====
    def _primary_metric_for_task(task: str) -> Tuple[str, float]:
        """Return (metric_name, scale) used for averaging.

        scale is multiplied to normalize metrics to [0,1].
        """
        if task in ("C-STANCE", "FOMC", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds"):
            return ("accuracy", 1.0)
        if task == "MeetingBank":
            return ("rouge-L", 1.0)
        if task == "Py150":
            return ("similarity", 1.0 / 100.0)  # fuzzy ratio 0-100
        if task == "20Minuten":
            return ("sari", 1.0 / 100.0)       # SARI 0-100
        return ("", 1.0)

    def _extract_primary_metric(task: str, evaluation_result: Dict[str, float]) -> Optional[float]:
        metric_name, scale = _primary_metric_for_task(task)
        if not metric_name:
            return None
        val = evaluation_result.get(metric_name)
        if val is None:
            return None
        return float(val) * float(scale)

    # round_summaries[round_index][task] = normalized_primary_metric
    round_summaries: List[Dict[str, float]] = []


    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    # TODO, modify left pad for training and inference
    inference_tasks = args.inference_tasks 
    task_num = len(inference_tasks)
    for round in range(task_num):   # load models and adapters of a new round in continual learning
        inference_model_path = os.path.join(args.inference_model_path, str(round))
        print_rank_0("Inference Model Path: " + inference_model_path, args.local_rank)

        model = create_hf_model(AutoModelForCausalLM,
                                args.model_name_or_path,
                                tokenizer,
                                ds_config=None,
                                )

        # REFT-CL: use pyreft's direct loading mechanism
        if args.CL_method == "REFT-CL":
            try:
                from pyvene import IntervenableModel
            except Exception:
                print_rank_0("[REFT-CL] Failed to import pyvene.IntervenableModel", args.local_rank)
                raise
            
            # Use IntervenableModel.load() which handles the full loading process
            try:
                model = IntervenableModel.load(inference_model_path, model)
                print_rank_0(f"[REFT-CL] Successfully loaded model from {inference_model_path}", args.local_rank)
            except Exception as load_error:
                print_rank_0(f"[REFT-CL] Direct load failed: {load_error}", args.local_rank)
                
                # Fallback: reconstruct the intervention structure and load manually
                print_rank_0("[REFT-CL] Attempting manual reconstruction and loading...", args.local_rank)
                from reft_loading_utils import load_reft_cl_model

                # Determine target layers
                layer_str = getattr(args, "reft_layers", "3;9;18;24")
                if layer_str.strip() == "all":
                    num_layers = model.config.num_hidden_layers
                    target_layers = list(range(num_layers))
                else:
                    target_layers = [int(x) for x in layer_str.split(";") if len(x) > 0]

                reft_config = {
                    'target_layers': target_layers,
                    'low_rank': int(getattr(args, "reft_rank", 4)),
                    'eps': float(getattr(args, "reft_eps", 1e-6)),
                    'num_tasks': len(inference_tasks)
                }
                
                model = load_reft_cl_model(model, inference_model_path, reft_config)
                print_rank_0(f"[REFT-CL] Successfully loaded interventions manually from {inference_model_path}", args.local_rank)
        
        # TODO: add adapters
        if args.CL_method == "LFPT5":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, inference_model_path)

        if args.CL_method == "O-LoRA":
            from utils.my_peft import PeftModel
            model = PeftModel.from_pretrained(model, inference_model_path)
            for name, param in model.named_parameters():
                if name.find("loranew_") != -1:
                    param.requires_grad = True
                elif name.find("lora_") != -1:
                    param.requires_grad = False

        if args.CL_method == "OGD":
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            for name, param in model.named_parameters():
                if name.find("lora") != -1:
                    param.requires_grad = True

        if args.CL_method=="PP" or args.CL_method=="L2P":
            if "opt" in args.model_name_or_path.lower():
                embed_tokens_shape = model.model.decoder.embed_tokens.weight.shape
                embed_tokens = model.model.decoder.embed_tokens
                args.embed_tokens_dim = embed_tokens_shape[1]
                args.embed_tokens_length = embed_tokens_shape[0]
                args.embed_tokens = embed_tokens
            elif "llama" in args.model_name_or_path.lower():
                embed_tokens_shape = model.model.embed_tokens.weight.shape
                embed_tokens = model.model.embed_tokens
                args.embed_tokens_dim = embed_tokens_shape[1]
                args.embed_tokens_length = embed_tokens_shape[0]
                args.embed_tokens = embed_tokens
            if args.CL_method=="PP":
                args.prefix_len = 20
                model = convert_PP_model(model, args)
            elif args.CL_method=="L2P":
                args.pool_size = 10
                args.prompt_length = 5
                args.prompt_init = "uniform"
                model = convert_L2P_model(model, args)
                for name, params in model.named_parameters():
                    if "prompt" not in name:
                        params.requires_grad=False

        if args.CL_method == "lora":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, inference_model_path)


        elif args.CL_method not in ("lora", "O-LoRA", "LFPT5", "REFT-CL"): 
            inference_model = torch.load(os.path.join(inference_model_path, "pytorch_model.bin"))
            for name, param in model.named_parameters():
                if name in inference_model:
                    param.data.copy_(inference_model[name])
            del inference_model

        model.to(device)

        for inference_task_id in range(round+1):    # evaluation for previous tasks in a single round
            inference_task = inference_tasks[inference_task_id]
            dataset_path = os.path.join(args.data_path, inference_task)
            # Prepare the data
            _, _, infer_dataset = create_prompt_dataset(
                args.local_rank,
                dataset_path,
                args.data_output_path,
                args.seed,
                distributed=False
            )

            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=args.max_prompt_len,
                max_ans_len=args.max_ans_len,
                pad_to_multiple_of=8,
                inference=True,
                task=inference_task
            )
            infer_sampler = SequentialSampler(infer_dataset)
            infer_dataloader = DataLoader(infer_dataset,
                                          collate_fn=inf_data_collator,
                                          sampler=infer_sampler,
                                          batch_size=args.inference_batch)
            progress_bar = tqdm(total=len(infer_dataloader), leave=True)

            # Inference !
            print_rank_0("***** Start inference *****", args.local_rank)
            # For REFT-CL, activate tasks up to current inference round
            if args.CL_method == "REFT-CL":
                base_ref = model.module if hasattr(model, "module") else model
                try:
                    for inter in getattr(base_ref, "interventions", {}).values():
                        if hasattr(inter, "set_active_tasks"):
                            inter.set_active_tasks(inference_task_id + 1)
                except Exception:
                    pass
            sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader)
            
            # Get Accuracy/ROUGE/BLEU/...
            # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
            if inference_task == "ScienceQA":
                evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
            elif inference_task == "MeetingBank":
                evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
            elif inference_task == "C-STANCE":
                evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
            elif inference_task == "Papyrus-f":
                evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
            elif inference_task == "Py150":
                evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
            elif inference_task == "FOMC":
                evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
            elif inference_task == "NumGLUE-cm":
                evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
            elif inference_task == "NumGLUE-ds":
                evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
            elif inference_task == "20Minuten":
                evaluation_result = eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)
            else:
                evaluation_result = {}

            # if args.global_rank <= 0:  # only one process is running
            print("***** Saving inference results *****")
            save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, inference_task_id, inference_task)

            # Update summary store
            if args.local_rank in (-1, 0):
                while len(round_summaries) <= round:
                    round_summaries.append({})
                v = _extract_primary_metric(inference_task, evaluation_result)
                if v is not None:
                    round_summaries[round][inference_task] = v

        # Per-round average across tasks seen so far
        if args.local_rank in (-1, 0):
            seen_tasks = [inference_tasks[i] for i in range(round + 1)]
            vals = [round_summaries[round].get(t) for t in seen_tasks]
            vals = [x for x in vals if isinstance(x, float)]
            if vals:
                avg_val = sum(vals) / len(vals)
                print(f"Round {round} average over {len(vals)} task(s): {avg_val:.3f}")

    # After all rounds, print compact table + final average + BWT
    if args.local_rank in (-1, 0) and len(round_summaries) == task_num:
        print("\n===== Continual Learning Summary (normalized primary metrics) =====")
        header = ["Task"] + [str(r) for r in range(task_num)]
        print("\t".join(header))
        for task in inference_tasks:
            row = [task]
            for r in range(task_num):
                v = round_summaries[r].get(task)
                row.append("" if v is None else f"{v:.3f}")
            print("\t".join(row))

        final_vals = [round_summaries[-1].get(t) for t in inference_tasks]
        final_vals = [x for x in final_vals if isinstance(x, float)]
        if final_vals:
            final_avg = sum(final_vals) / len(final_vals)
            print(f"Final average (round {task_num-1}): {final_avg:.3f}")

        # BWT formula: mean_{i<T} (R_T,i - R_i,i)
        diag = []
        final = []
        for i, task in enumerate(inference_tasks):
            if i < task_num - 1:
                r_ii = round_summaries[i].get(task)
                r_ti = round_summaries[-1].get(task)
                if isinstance(r_ii, float) and isinstance(r_ti, float):
                    diag.append(r_ii)
                    final.append(r_ti)
        if diag and final:
            bwt = sum((ft - di) for ft, di in zip(final, diag)) / len(diag)
            print("BWT (mean_i<T (R_T,i - R_i,i)): {:.3f}".format(bwt))

if __name__ == "__main__":
    main()
