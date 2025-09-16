import os
import torch
import transformers
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoConfig
from training.trainer import LLamaVTrainer, QwenTrainer
from training.data_qwen import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from torch.nn import Linear
import pathlib
from typing import Dict, List
import torch.distributed as dist
from tqdm import tqdm
local_rank = None
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.nn import Embedding, Linear
try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2ColumnParallelLinear as ColLinear,
        Qwen2RowParallelLinear as RowLinear,
    )
except ImportError:
    ColLinear = RowLinear = Linear

def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)

def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True
):
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


def configure_llm(model, training_args):
    llm_params = model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def _gather_full_param(param):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            return None
        with zero.GatheredParameters([param], modifier_rank=None):
            return param.detach().cpu().clone()
    return param.detach().cpu().clone()

keep_kw = ("q_proj","k_proj","v_proj", "down_proj" 
           "language_model.model.embed_tokens",
           "language_model.lm_head")

def train():
    global local_rank
    import wandb

    os.environ["WANDB_PROJECT"] = "sft"
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert not (
        training_args.lora_enable and training_args.freeze_llm
    ), "When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA."

    if not training_args.lora_enable:
        assert not training_args.vision_lora, (
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        )
    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = []

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                )
            )
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2"
        if not training_args.disable_flash_attn2
        else "sdpa",
        **bnb_model_from_pretrained_args,
    )

    modules_dict = dict(model.named_modules())

    try:
        from transformers.models.llama.modeling_llama import (
            ColumnParallelLinear, RowParallelLinear
        )
    except ImportError:
        ColumnParallelLinear = RowParallelLinear = Linear

    # import gc, ctypes
    torch.cuda.ipc_collect()
    from torch.cuda import memory
    # memory._free_cached_blocks()
    memory._free_mutex()
    torch.cuda.empty_cache()
    # I set a hidden size for temporary use. This is to use the deepspeed.
    # I will find a proper way later.
    # model.config.hidden_size = model.config.text_config.hidden_size
    # model.config.text_config.use_cache = False

    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if training_args.lora_enable:
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=training_args.lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, padding_side="right")

    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tokenizer_padding_side = tokenizer.padding_side

    if not training_args.lora_enable:
        configure_llm(model, training_args)

    data_module = make_supervised_data_module(processor=tokenizer, data_args=data_args)

    trainer = QwenTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()