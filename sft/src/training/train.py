import os

import torch
from torch.nn import Linear
import transformers
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration, TrainerCallback
from training.trainer import LLamaVTrainer
from training.data import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, \
    safe_save_model_for_hf_trainer
import pathlib
from tqdm import tqdm
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import numpy as np

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.vision_model
    vision_tower.to(dtype=compute_dtype, device=device)

    img_projection_params = model.multi_modal_projector.parameters()
    set_requires_grad(img_projection_params, training_args.tune_img_projector)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    if training_args.bits in [4, 8]:
        model.multi_modal_projector.to(dtype=compute_dtype, device=device)


def configure_llm(model, training_args):
    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def _gather_full_param(param):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            return None
        with zero.GatheredParameters([param], modifier_rank=None):
            return param.detach().cpu().clone()
    return param.detach().cpu().clone()


keep_kw = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".mlp.down_proj.weight",)

from torch.utils.data import Dataset
import json, torch


class ConversationsEvalDataset(Dataset):
    """
    Loads an OOD eval file that looks like:
      [{"conversations":[{"from":"human","value":...},{"from":"gpt","value":...}], ...}, ...]
    Builds (input_ids, attention_mask, labels) so the Trainer can evaluate with predict_with_generate=True.
    """

    def __init__(self, path, processor, max_length=None):
        self.processor = processor
        self.max_length = max_length
        # Load list-of-dicts from .json or .jsonl
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                self.data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        convs = ex["conversations"]
        # Pick the first human + its assistant reply (adjust if your files have more turns)
        # If there are multiple pairs, you can also keep the latest pair:
        human_msg = next(m for m in convs if m["from"].lower().startswith("human"))
        gpt_msg = next(m for m in convs if m["from"].lower().startswith("gpt"))

        # Build messages for the chat template
        msgs_prompt_only = [{"role": "user", "content": human_msg["value"]}]
        msgs_full = [{"role": "user", "content": human_msg["value"]},
                     {"role": "assistant", "content": gpt_msg["value"]}]

        # Safer path: build strings via template, then tokenize once so we can get boundary cleanly
        prompt_text = self.processor.apply_chat_template(
            msgs_prompt_only, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False
        )

        tok = self.processor.tokenizer
        full_enc = tok(full_text, add_special_tokens=False, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        prompt_enc = tok(prompt_text, add_special_tokens=False, truncation=True,
                         max_length=self.max_length, return_tensors="pt")

        input_ids = full_enc.input_ids[0]
        attn_mask = full_enc.attention_mask[0]
        labels = input_ids.clone()

        # Mask the prompt part from the loss
        prompt_len = prompt_enc.input_ids.shape[1]
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            # keep id if you want to join back to CSV/debug
            "example_id": ex.get("id", str(idx)),
        }


def train():
    global local_rank
    import wandb
    os.environ["WANDB_PROJECT"] = "sft"
    # wandb.init(project=os.environ["WANDB_PROJECT"])
    print('starting============')
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print('parser============')
    assert not (
                training_args.lora_enable and training_args.freeze_llm), 'When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA.'

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = ["multi_modal_projector"]

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["vision_model", "multi_modal_projector"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["multi_modal_projector", "vision_model"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    print('start model============')
    model = MllamaForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        **bnb_model_from_pretrained_args
    )
    modules_dict = dict(model.named_modules())

    try:
        from transformers.models.llama.modeling_llama import (
            ColumnParallelLinear, RowParallelLinear
        )
    except ImportError:
        ColumnParallelLinear = RowParallelLinear = Linear

    model.config.hidden_size = model.config.text_config.hidden_size
    model.config.text_config.use_cache = False

    print('training_args============')

    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude,
                                                    num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # use unk rather than eos token to prevent endless generation
    processor.tokenizer.padding_side = 'right'

    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side

    # When using LoRA, the model is rapped once more.
    if training_args.lora_enable:
        model_to_configure = model.model
    else:
        model_to_configure = model
        configure_llm(model, training_args)

    if not training_args.vision_lora:
        configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)

            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    training_args.include_inputs_for_metrics = True
    print('trainer begin============')
    trainer = LLamaVTrainer(
        model=model,
        processor=processor,
        args=training_args,
        **data_module,
    )

    # trainer.compute_metrics = make_compute_metrics_chat_safe(processor.tokenizer, trainer)
    # trainer.compute_metrics = make_compute_metrics_chat_safe(processor.tokenizer, trainer, face_value_rule="rule10")
    # trainer.compute_metrics = trainer._make_24pt_metric_both(csv_prefix="./eval_debug_24pt")
    # training_args.predict_with_generate = True
    # training_args.generation_max_new_tokens = 128
    # training_args.generation_num_beams = 1

    # eval_dataset = data_module["eval_dataset"]
    # eval_id = ConversationsEvalDataset("/scratch/l/luli/data/SFTvsRL_Data/SFT_Data/gp-l/ind-eval_300.json", processor)
    # Run both evals whenever you want
    # id_metrics = trainer.evaluate(eval_dataset=eval_id)  # → eval_accuracy
    # ood_metrics = trainer.evaluate_ood(eval_dataset=eval_dataset)  # → ood_accuracy
    # both = trainer.evaluate_ood_both_rules(eval_dataset=eval_dataset)
    # print(id_metrics, ood_metrics, both)
    # print(id_metrics, ood_metrics)

    # During/after training step(s):
    # trainer.compute_metrics = trainer._make_24pt_metric(face_rule="rule13")  # OOD rule
    # metrics = trainer.evaluate(eval_dataset=eval_dataset)  # ← this is your OOD split
    # print(metrics)
    print('train begin============')
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        print('train begin before============')
        trainer.train()
        print('train begin after============')

    trainer.save_state()

    model.config.text_config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    # import subprocess, os, time, threading
    # def gpu_watch():
    #     while True:
    #         out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    #         print("[GPU] used MB:", out.decode().strip())
    #         time.sleep(5)
    #
    # threading.Thread(target=gpu_watch, daemon=True).start()
    train()
