import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
import time
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds # to be continued
from transformers import GenerationConfig
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)


class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        
        
    def _debug_log_training_tensors(self, batch, outputs, step, epoch):
        try:
            underlying = self.model.module if hasattr(self.model, 'module') else self.model
            model_name = underlying.__class__.__name__
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')
            labels = batch.get('labels')
            msg_lines = [
                f"[DEBUG] epoch={epoch+1} step={step}",
                f"  model={model_name}",
                f"  input_ids: shape={tuple(input_ids.shape) if hasattr(input_ids,'shape') else None} dtype={getattr(input_ids,'dtype',None)}",
                f"  attention_mask: shape={tuple(attention_mask.shape) if hasattr(attention_mask,'shape') else None} dtype={getattr(attention_mask,'dtype',None)}",
                f"  labels: shape={tuple(labels.shape) if hasattr(labels,'shape') else None} dtype={getattr(labels,'dtype',None)}",
            ]
            # Output introspection
            if hasattr(outputs, 'logits') and getattr(outputs, 'logits') is not None:
                logits = outputs.logits
                msg_lines.append(f"  logits: shape={tuple(logits.shape)} dtype={logits.dtype}")
            elif isinstance(outputs, tuple):
                shapes = []
                for t in outputs:
                    if torch.is_tensor(t):
                        shapes.append(f"Tensor{tuple(t.shape)}:{t.dtype}")
                    else:
                        shapes.append(type(t).__name__)
                msg_lines.append("  outputs(tuple): " + ", ".join(shapes[:6]) + (" ..." if len(shapes) > 6 else ""))
            else:
                msg_lines.append(f"  outputs type: {type(outputs).__name__}")

            # Small content preview
            try:
                if torch.is_tensor(input_ids):
                    msg_lines.append(f"  input_ids[0,:10]: {input_ids[0, :10].tolist()}")
                if torch.is_tensor(labels):
                    # show non -100 positions in first row
                    first_row = labels[0]
                    keep = (first_row != -100).nonzero(as_tuple=False).flatten()[:10]
                    preview = first_row[keep].tolist() if keep.numel() > 0 else []
                    msg_lines.append(f"  labels(non-ignored) preview: {preview}")
            except Exception:
                pass

            print_rank_0("\n".join(msg_lines), self.args.global_rank)
        except Exception:
            # Best-effort only
            pass

    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                # If wrapped by DeepSpeed, underlying module may be a pyreft IntervenableModel
                underlying = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(underlying, 'interventions'):
                    base_inputs = {"input_ids": batch["input_ids"], "attention_mask": batch.get("attention_mask")}
                    outputs = self.model(base_inputs, labels=batch.get("labels"))
                    # If wrapper returns (base_out, cf_out), prefer the element with logits/loss
                    if isinstance(outputs, (tuple, list)):
                        picked = None
                        for e in outputs:
                            if hasattr(e, 'loss') or hasattr(e, 'logits'):
                                picked = e
                                break
                        outputs = picked if picked is not None else outputs
                else:
                    outputs = self.model(**batch, use_cache=False)
            # Robust loss extraction (handles tuple returns or missing .loss)
            labels = batch.get("labels")
            loss = None
            if isinstance(outputs, tuple) and len(outputs) > 0 and torch.is_tensor(outputs[0]) and outputs[0].ndim == 0:
                loss = outputs[0]
            elif hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            elif labels is not None:
                logits = getattr(outputs, "logits", None)
                if logits is None and isinstance(outputs, tuple):
                    # Best-effort: pick first tensor with 3 dims as logits
                    for t in outputs:
                        if torch.is_tensor(t) and t.ndim >= 3:
                            logits = t
                            break
                if logits is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity


    def train_one_task(self, task, i_task, epochs):
        # 在单独某个任务上训练
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, device)
                # If wrapped by DeepSpeed, underlying module may be a pyreft IntervenableModel
                underlying = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(underlying, 'interventions'):
                    base_inputs = {"input_ids": batch["input_ids"], "attention_mask": batch.get("attention_mask")}
                    outputs = self.model(base_inputs, labels=batch.get("labels"))
                    if isinstance(outputs, (tuple, list)):
                        picked = None
                        for e in outputs:
                            if hasattr(e, 'loss') or hasattr(e, 'logits'):
                                picked = e
                                break
                        outputs = picked if picked is not None else outputs
                else:
                    outputs = self.model(**batch, use_cache=False)
                # Robust loss extraction
                labels = batch.get("labels")
                loss = None
                if isinstance(outputs, tuple) and len(outputs) > 0 and torch.is_tensor(outputs[0]) and outputs[0].ndim == 0:
                    loss = outputs[0]
                elif hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                elif labels is not None:
                    logits = getattr(outputs, "logits", None)
                    if logits is None and isinstance(outputs, tuple):
                        for t in outputs:
                            if torch.is_tensor(t) and t.ndim >= 3:
                                logits = t
                                break
                    if logits is not None:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )

                # If loss is still None or non-finite, print detailed debug info and raise
                if loss is None or not torch.isfinite(loss):
                    self._debug_log_training_tensors(batch, outputs, step, epoch)
                    raise RuntimeError("Training loss is None or non-finite; see debug log above.")
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    loss_str = f"{loss.item():.4f}" if torch.is_tensor(loss) else "None"
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss_str}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()


            # Evaluate perplexity on the validation set.
            # print_rank_0(
            #     f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs} *****",
            #     self.args.global_rank)
            # perplexity = self.perplexity_evaluation(eval_dataloader, device)
            # print_rank_0(f"ppl: {perplexity}", self.args.global_rank)
            # self.model.tput_timer.update_epoch_count()
    
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            self.save_model(i_task)

    
    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))

        if self.args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
        print_rank_0('Sucessful saving model after round {}'.format(round), self.args.global_rank)
        
