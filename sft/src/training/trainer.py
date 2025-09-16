import os
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
import safetensors
from peft import PeftModel
from typing import Optional
import numpy as np
from transformers.processing_utils import ProcessorMixin
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from tqdm import tqdm
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import torch, ast, operator as op, numpy as np, csv, re
from torch.nn import Embedding, Linear
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def _ece(confs, correct, n_bins=15):
    confs = np.asarray(confs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    bins = np.linspace(0, 1, n_bins + 1)
    N = confs.shape[0]; e = 0.0
    for i in range(n_bins):
        m = (confs > bins[i]) & (confs <= (bins[i+1] if i < n_bins-1 else bins[i+1]))
        if m.sum() == 0:
            continue
        e += (m.sum()/N) * abs(correct[m].mean() - confs[m].mean())
    return float(e)

class LLamaVTrainer(Trainer):

    def __init__(self, *args, processor: Optional[ProcessorMixin] = None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self._is_ood_eval = False

        # pad id to allow prompt-trimming
        self._pad_id = None
        if self.processor and self.processor.tokenizer.pad_token_id is not None:
            self._pad_id = self.processor.tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = getattr(self.model.config, "pad_token_id", None)
        if self._pad_id is None:
            self._pad_id = getattr(self.model.generation_config, "pad_token_id", None)
        if self._pad_id is None:
            self._pad_id = self.model.config.eos_token_id

    def _reset_calib_buffers(self):
        self._eval_seq_conf = []   # list[np.ndarray], shape [B]
        self._eval_seq_nll  = []   # list[np.ndarray], shape [B]
        self._eval_correct  = []   # list[np.ndarray], shape [B] (filled by compute_metrics)

    @torch.no_grad()
    def prediction_step_bak(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # ------------ instrumentation (unchanged) ------------
        try:
            if not hasattr(self, "_lbl_cov"): self._lbl_cov = []
            if not hasattr(self, "_trim_pos"): self._trim_pos = []
            if not hasattr(self, "_lbl_comp_digits"): self._lbl_comp_digits = []
            if not hasattr(self, "_lbl_comp_alpha"):  self._lbl_comp_alpha = []
            if not hasattr(self, "_lbl_has_formula_key"): self._lbl_has_formula_key = []
            if not hasattr(self, "_lbl_has_equals"):      self._lbl_has_equals = []

            labels = inputs.get("labels", None)
            inp_ids = inputs.get("input_ids", None)
            if labels is not None and inp_ids is not None:
                mask = (labels != -100)
                cov = mask.float().mean().item()
                self._lbl_cov.append(cov)

                B, T = labels.shape
                for i in range(B):
                    nz = (labels[i] != -100).nonzero(as_tuple=False)
                    self._trim_pos.append(float(int(nz[0])) / float(T) if nz.numel() > 0 else 1.0)

                labeled_ids = inp_ids[mask]
                if labeled_ids.numel() > 0:
                    txt = self.processor.tokenizer.decode(labeled_ids.tolist(), skip_special_tokens=True)
                    if len(txt) > 0:
                        n = len(txt)
                        digits = sum(ch.isdigit() for ch in txt) / n
                        alpha = sum(ch.isalpha() for ch in txt) / n
                        self._lbl_comp_digits.append(digits)
                        self._lbl_comp_alpha.append(alpha)
                        self._lbl_has_formula_key.append(1.0 if "formula" in txt.lower() else 0.0)
                        self._lbl_has_equals.append(1.0 if "=" in txt else 0.0)
                        # if self.accelerator.is_main_process:
                        #     print("txt=====", txt)
        except Exception as _e:
            if self.accelerator.is_main_process:
                print("[instr] prediction_step logging failed:", _e)

        has_labels = "labels" in inputs and inputs["labels"] is not None
        do_generate = bool(self.args.predict_with_generate)

        gen_tokens = None
        loss = None
        prompt_lens = []  # start indices inside returned sequences

        if do_generate:
            # ---- Build a LEFT-PADDED prompt-only batch ----
            in_ids = inputs["input_ids"]
            attn = inputs.get("attention_mask", None)
            labels = inputs.get("labels", None)

            B, T = in_ids.shape
            cuts = []

            for i in range(B):
                if labels is not None:
                    nz = (labels[i] != -100).nonzero(as_tuple=False)
                    cut = int(nz[0]) if nz.numel() > 0 else int(attn[i].sum().item())
                else:
                    cut = int(attn[i].sum().item()) if attn is not None else T
                cuts.append(cut)

            max_len = max(cuts)  # width of the prompt-only batch
            pad_id = self._pad_id

            new_input_ids = in_ids.new_full((B, max_len), pad_id)
            new_attention = in_ids.new_zeros((B, max_len))

            for i in range(B):
                cut = cuts[i]
                # take exactly the prompt tokens [0:cut], left-pad to max_len
                src = in_ids[i, :cut]
                L = int(src.shape[0])
                new_input_ids[i, -L:] = src
                new_attention[i, -L:] = 1

            gen_inputs = {"input_ids": new_input_ids, "attention_mask": new_attention}

            gen_kwargs = dict(
                max_new_tokens=128,
                num_beams=self.args.generation_num_beams or 1,
                do_sample=False,
                pad_token_id=self._pad_id,
                eos_token_id=self.model.config.eos_token_id,
            )

            gen_tokens = model.generate(**gen_inputs, **gen_kwargs)
            if gen_tokens.ndim == 1:
                gen_tokens = gen_tokens.unsqueeze(0)
            gen_tokens = gen_tokens[:, new_input_ids.size(1):]
            # We no longer need __gen_prompt_len for metrics
            prompt_lens = [0] * gen_tokens.size(0)
            # Start of generated tail is exactly the prompt batch width (max_len)
            prompt_lens = [max_len] * B

        else:
            # fallback if generation disabled
            B = inputs["input_ids"].size(0)
            prompt_lens = [int(inputs["attention_mask"][i].sum().item()) for i in range(B)]

        # expose prompt_lens for compute_metrics slicing
        inputs["__gen_prompt_len"] = torch.tensor(
            prompt_lens, device=inputs["input_ids"].device
        )

        if has_labels:
            model_inputs = {k: v for k, v in inputs.items() if k != "__gen_prompt_len"}
            outputs = model(**model_inputs)
            loss = outputs.get("loss", None)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, gen_tokens, inputs.get("labels", None))

    # def evaluate_ood_both_rules(self, eval_dataset, prefix_main="ood"):
    #     prev_cm = self.compute_metrics
    #     try:
    #         self.compute_metrics = self._metric_ood
    #         m1 = self.evaluate(eval_dataset=eval_dataset, metric_key_prefix=prefix_main)  # <— self.evaluate
    #
    #         self.compute_metrics = self._make_24pt_metric(
    #             face_rule="rule10", csv_name="./eval_debug_24pt_ood_as_rule10.csv"
    #         )
    #         m2 = self.evaluate(eval_dataset=eval_dataset, metric_key_prefix=f"{prefix_main}_as_rule10")  # <—
    #     finally:
    #         self.compute_metrics = prev_cm
    #     return m1 | m2

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # reset buffers for this eval
        self._lbl_cov = [];
        self._trim_pos = []
        self._lbl_comp_digits = [];
        self._lbl_comp_alpha = []
        self._lbl_has_formula_key = [];
        self._lbl_has_equals = []

        metrics = super().evaluate(eval_dataset=eval_dataset,
                                   ignore_keys=ignore_keys,
                                   metric_key_prefix=metric_key_prefix)

        # aggregate
        import numpy as _np
        def _m(x): return float(_np.mean(x)) if len(x) > 0 else float("nan")

        metrics[f"{metric_key_prefix}_label_cov_mean"] = _m(self._lbl_cov)
        metrics[f"{metric_key_prefix}_trim_pos_mean"] = _m(self._trim_pos)
        metrics[f"{metric_key_prefix}_lbl_digits_share"] = _m(self._lbl_comp_digits)
        metrics[f"{metric_key_prefix}_lbl_alpha_share"] = _m(self._lbl_comp_alpha)
        metrics[f"{metric_key_prefix}_lbl_has_formula"] = _m(self._lbl_has_formula_key)
        metrics[f"{metric_key_prefix}_lbl_has_equals"] = _m(self._lbl_has_equals)

        if self.accelerator.is_main_process:
            step = int(self.state.global_step) if hasattr(self, "state") and self.state.global_step is not None else -1
            print(f"[{metric_key_prefix}] step={step} "
                  f"label_cov_mean={metrics[f'{metric_key_prefix}_label_cov_mean']:.4f} "
                  f"trim_pos_mean={metrics[f'{metric_key_prefix}_trim_pos_mean']:.4f} "
                  f"digits={metrics[f'{metric_key_prefix}_lbl_digits_share']:.3f} "
                  f"alpha={metrics[f'{metric_key_prefix}_lbl_alpha_share']:.3f} "
                  f"has_formula={metrics[f'{metric_key_prefix}_lbl_has_formula']:.2f} "
                  f"has_eq={metrics[f'{metric_key_prefix}_lbl_has_equals']:.2f}")

        # (optionally) also push via trainer.log so W&B can pick them up if enabled
        self.log(metrics)
        return metrics

    # ----------------- Public helper to run OOD eval -----------------
    def evaluate_ood(self, eval_dataset, metric_prefix="ood", face_rule="rule13"):
        prev_cm = self.compute_metrics
        try:
            self.compute_metrics = self._metric_ood if face_rule == "rule13" else self._make_24pt_metric(
                face_rule=face_rule, csv_name="./eval_debug_24pt_ood.csv"
            )
            return self.evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_prefix)  # <—
        finally:
            self.compute_metrics = prev_cm

    # ----------------- In-class 24pt metric factory -----------------
    def _make_24pt_metric_both(self, csv_prefix="./eval_debug_24pt"):
        # local helpers (same parsing/eval you already use)
        import numpy as _np, csv, re, ast, operator as op
        _ALLOWED_BINOPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
        _ALLOWED_UNOPS = {ast.UAdd: lambda x: x, ast.USub: op.neg}

        FACE_MAPS = {
            "rule10": {"j": 10, "q": 10, "k": 10, "a": 1},
            "rule13": {"j": 11, "q": 12, "k": 13, "a": 1},
        }

        def _safe_eval(expr):
            try:
                node = ast.parse(expr, mode="eval")
            except Exception:
                return False, float("nan"), []
            used = []

            def ev(n):
                if isinstance(n, ast.Expression): return ev(n.body)
                if isinstance(n, ast.Num):        v = float(n.n); used.append(int(round(v))); return v
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                    v = float(n.value);
                    used.append(int(round(v)));
                    return v
                if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNOPS:
                    return _ALLOWED_UNOPS[type(n.op)](ev(n.operand))
                if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
                    a = ev(n.left);
                    b = ev(n.right)
                    if isinstance(n.op, ast.Div) and abs(b) < 1e-12: raise ZeroDivisionError
                    return _ALLOWED_BINOPS[type(n.op)](a, b)
                raise ValueError

            try:
                val = ev(node)
            except Exception:
                return False, float("nan"), []
            return True, float(val), used

        def _parse_cards_with_rule(text, face_map):
            def _norm_rank(s):
                s = s.strip().lower().strip("'\"")
                return int(face_map.get(s, s))

            m = re.search(r"(?i)cards?\s*[:：]\s*\[([^\]]+)\]", text)
            if not m: return None
            toks = re.findall(r"[A-Za-z]+|\d+", m.group(1))
            return [_norm_rank(t) for t in toks] if toks else None

        def _mset_eq(a, b):
            from collections import Counter
            return Counter(a) == Counter(b)

        def _extract_formula_candidates(text: str):
            cands = []
            # JSON-style value
            for s in re.findall(r'"formula"\s*:\s*[\'"]([^\'"\n\r}]+)[\'"]', text, flags=re.IGNORECASE):
                s = s.strip()
                if s: cands.append(s)
            # loose lines
            for line in text.splitlines():
                line = line.strip()
                if any(ch in line for ch in "+-*/()=") and len(line) >= 3:
                    if re.search(r"[A-Za-z]", line) and not re.match(r"^[\s(]*\d", line):
                        continue
                    cands.append(line)
            # de-dup
            out, seen = [], set()
            for s in cands:
                s = s.strip()
                if s and s not in seen:
                    seen.add(s);
                    out.append(s)
            return out

        def _eval_formula_or_equality(s):
            s = s.strip()
            if "=" in s:
                left, right = s.split("=", 1)
                left, right = left.strip(), right.strip()
                okL, vL, numsL = _safe_eval(left)
                okR, vR, numsR = _safe_eval(right)
                if not (okL and okR): return False, float("nan"), []
                if abs(vL - vR) > 1e-6: return False, float("nan"), []
                # Prefer counting numbers from the side that is NOT literally 24
                if abs(vR - 24.0) < 1e-6:  # constructed expr on left
                    return True, vL, numsL
                if abs(vL - 24.0) < 1e-6:  # constructed expr on right
                    return True, vR, numsR
                # equal but not 24; caller will reject on the 24-check
                return True, vL, numsL
            else:
                return _safe_eval(s)

        tok = self.processor.tokenizer if self.processor is not None else self.tokenizer

        def _metric(eval_pred):
            preds_obj = eval_pred.predictions
            if isinstance(preds_obj, (list, tuple)) and len(preds_obj) == 1:
                preds_obj = preds_obj[0]
            if isinstance(preds_obj, torch.Tensor):
                preds_np = preds_obj.detach().cpu().numpy()
            else:
                preds_np = _np.asarray(preds_obj)
            if preds_np.ndim == 1: preds_np = preds_np[None, :]

            # decode only the generated tail using the prompt lengths we stashed
            inputs = getattr(eval_pred, "inputs", None)
            if inputs is None or "__gen_prompt_len" not in inputs:
                # fallback: decode whole seq (should rarely happen)
                tails = tok.batch_decode(preds_np, skip_special_tokens=True)
            else:
                lens = inputs["__gen_prompt_len"]
                lens = lens.detach().cpu().numpy() if hasattr(lens, "device") else _np.asarray(lens)
                tails = []
                for i in range(preds_np.shape[0]):
                    start = int(lens[i]);
                    seq_ids = preds_np[i]
                    tail_ids = seq_ids[start:] if start < seq_ids.shape[0] else seq_ids[-0:]
                    tails.append(tok.decode(tail_ids.tolist(), skip_special_tokens=True))

            # decode prompts (for cards)
            prompt_texts = None
            if inputs is not None and "input_ids" in inputs:
                inp_ids = inputs["input_ids"]
                prompt_texts = tok.batch_decode(inp_ids, skip_special_tokens=True) if hasattr(inp_ids, "device") \
                    else tok.batch_decode(inp_ids.tolist(), skip_special_tokens=True)

            ok_id = []
            ok_ood = []
            rows = []
            for i, tail in enumerate(tails):
                prompt_text = prompt_texts[i] if prompt_texts is not None else ""
                cands = _extract_formula_candidates(tail)
                if self.accelerator.is_main_process and i < 8:
                    # print(f"[dbg] tail[{i}] len={len(tail)} head={repr(tail[:256])}")
                    print(f"[dbg] tail[{i}] len={len(tail)} head={repr(tail)}")
                    print(f"[dbg] cands[{i}]: {cands[:3]}")

                def score_for(face_map):
                    cards = _parse_cards_with_rule(prompt_text, face_map)
                    if not cands:  return 0, "no-formula-in-gen", "", [], ""
                    last_reason = "no-valid-formula"
                    for cand in reversed(cands):
                        safe, v, nums = _eval_formula_or_equality(cand)
                        if not safe:            last_reason = "unsafe-eval"; continue
                        if cards is None:       last_reason = "no-cards-in-prompt"; continue
                        if not _mset_eq([int(x) for x in nums], [int(x) for x in cards]):
                            last_reason = f"wrong-multiset used={nums} cards={cards}";
                            continue
                        if abs(v - 24.0) > 1e-6:    last_reason = f"value!=24({v})"; continue
                        return 1, "ok", cand, nums, f"{v:.6f}"
                    return 0, last_reason, "", [], ""

                id_hit, id_reason, id_c, id_nums, id_val = score_for(FACE_MAPS["rule10"])
                ood_hit, ood_reason, ood_c, ood_nums, ood_val = score_for(FACE_MAPS["rule13"])

                ok_id.append(id_hit);
                ok_ood.append(ood_hit)

                if self.accelerator.is_main_process and i < 20:
                    print(f"[both] idx={i} ID={'OK' if id_hit else id_reason} | "
                          f"OOD={'OK' if ood_hit else ood_reason}")

                rows.append([i, id_hit, ood_hit, id_reason, ood_reason, id_c or ood_c,
                             id_val or ood_val, str(id_nums or ood_nums), tail[:200].replace("\n", "\\n")])

            # optional CSV per eval call (named by global step)
            if self.accelerator.is_main_process:
                step = int(self.state.global_step) if getattr(self.state, "global_step", None) is not None else 0
                csv_path = f"{csv_prefix}_both_step{step}.csv"
                try:
                    with open(csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["idx", "id_ok", "ood_ok", "id_reason", "ood_reason",
                                    "chosen_formula", "value", "used_nums", "gen_head"])
                        w.writerows(rows)
                    print(f"[both] wrote {len(rows)} rows -> {csv_path}")
                except Exception as e:
                    print("[both] csv write failed:", e)

            return {
                "eval_id_accuracy": float(_np.mean(ok_id)) if len(ok_id) > 0 else 0.0,
                "eval_ood_accuracy": float(_np.mean(ok_ood)) if len(ok_ood) > 0 else 0.0,
            }

        return _metric

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper["multi_modal_projector"] = self.args.projector_lr
            if self.args.vision_lr is not None:
                lr_mapper["vision_model"] = self.args.vision_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(LLamaVTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                state_dict = {k:v for k, v in state_dict.items() if "wte" not in k}
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            if self.processor is not None:
                self.processor.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

class QwenTrainer(Trainer):

    def __init__(self, *args, processor: Optional[ProcessorMixin] = None,
                 **kwargs):
        super(QwenTrainer, self).__init__(*args, **kwargs)
        self.processor = processor

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper["multi_modal_projector"] = self.args.projector_lr
            if self.args.vision_lr is not None:
                lr_mapper["vision_model"] = self.args.vision_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if
                                         any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (
                                    n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if
                                           (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (
                                            n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)

            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(),
                                                                    require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                        self.state.best_metric is None
                        or self.state.best_model_checkpoint is None
                        or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(QwenTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            state_dict = {k: v for k, v in state_dict.items() if "tok_embedding" not in k}
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))