import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.trainer.model_tz_llama import VLMValue, VLMPolicy
from rl.trainer.storage_tz import RolloutStorage
from rl.trainer.base_trainer import BaseTrainer
import rl.trainer.algo as algo

import wandb
from typing import Optional, Dict, List, Any
import accelerate
from accelerate.state import AcceleratorState

from utils_mllm import evaluate_model_config
from utils_general import progress_bar, re_match
import os

from PIL import Image

class QwenTrainer(BaseTrainer):
    def __init__(self, action_space, daytime, accelerator,
                 optimizer_config, ppo_config, compute_return_kwargs,
                 num_steps, num_updates,
                 env_config,
                 model, model_path,
                 prompt_config, generation_config,
                 output_dir, ood_num_steps=10, seed=42, report_to=None,
                 run_name='default', save_ckpt=False, **kw):
        self.ood_num_steps = ood_num_steps
        super().__init__(action_space, daytime, accelerator, optimizer_config,
                         ppo_config, compute_return_kwargs,
                         num_steps, num_updates,
                         env_config, model, model_path,
                         prompt_config, generation_config,
                         output_dir, seed, report_to, run_name, save_ckpt, **kw)
    
    def init_model_optimizer_algo(self, model, model_path, ppo_cfg, opt_cfg):
        self.tokenizer, self.model = evaluate_model_config(model, model_path)

        value_model = VLMValue(self.model)
        actor_critic = VLMPolicy(
            tokenizer=self.tokenizer,
            value_model=value_model,
            generation_config=self.generation_config
        )

        optimizer = optim.Adam(actor_critic.value_model.parameters(),
                               lr=opt_cfg.init_lr,
                               eps=opt_cfg.eps,
                               weight_decay=opt_cfg.weight_decay)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt_cfg.lr_max_steps, eta_min=opt_cfg.end_lr
        )

        ds = AcceleratorState().deepspeed_plugin
        if ds is not None:
            ds.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

        self.actor_critic, self.optimizer, self.lr_scheduler = \
            self.accelerator.prepare(actor_critic, optimizer, lr_sched)

        self.agent = algo.PPO(self.actor_critic, self.optimizer,
                              self.accelerator, **ppo_cfg)

        self.rollouts = RolloutStorage(self.num_steps,
                                       self.env.action_space,
                                       self.generation_config.max_new_tokens)

        self.obs, self.info = self.env.reset()
        self.rollouts.obs[0]['image'] = self.obs

        if hasattr(self, "ood_env"):
            self.rollouts_ood = RolloutStorage(
                self.ood_num_steps,  # ← short length
                self.ood_env.action_space,
                self.generation_config.max_new_tokens
            )
            self.ood_obs, self.ood_info = self.ood_env.reset()
        
    def formulate_payload(self, question):
        self.payload = [{"role": "user", "content": question}]

    def formulate_prompt(self, v_dict, l_dict, prompt=None, info=None):
        if 'gym_cards' in self.id:
            if info['Verify Info'] is not None:
                self.payload.append({"role": "user",
                                     "content": f"You failed this trial because {info['Verify Info']}"})
            else:
                q = prompt.format(**v_dict, **l_dict, **self.oracle_arguments)
                self.formulate_payload(q)
        elif 'gym_virl' in self.id:
            if info['Verify Info'] is None or \
               'Correct action' in info['Verify Info'] or \
               'step_limit_reached' in info['Verify Info']:
                q = prompt.format(**v_dict, **l_dict, **self.oracle_arguments)
                self.formulate_payload(q)
            else:
                self.payload.append({"role": "user",
                                     "content": f"You failed this trial because {info['Verify Info']}"})

    @torch.no_grad()
    def collect_trajectories_ood(self, steps=None):
        """
        Collect a short OOD rollout with the *current* policy (no grads),
        storing old log-probs & values into self.rollouts_ood.
        Supports both text-only (Qwen) and vision (processor) paths.
        """
        steps = steps or getattr(self, "ood_num_steps", 10)
        obs, info = self.ood_obs, self.ood_info

        # Select prompts/patterns
        if self.use_vision:
            prompts = (self.ood_prompts or self.prompt_vision)
            patterns = self.pattern_vision
        else:
            prompts = (self.ood_prompts or self.prompt_language)
            patterns = self.pattern_language

        pbar = progress_bar(steps, "Collecting OOD Trajectories", "magenta", self.accelerator)

        # convenience: build image if needed
        obs_img = Image.fromarray(obs) if (self.use_vision and isinstance(obs, np.ndarray)) else (
            obs if self.use_vision else None)

        next_values = None
        for step in range(steps + 1):
            vision_res_dict, language_res_dict = {}, {}
            self.formulate_vision_arguments(vision_res_dict, info)

            # For each template/pattern pair, construct a single 'user' message with OOD oracle args
            for prompt, pattern in zip(prompts, patterns):
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments_ood)

                if self.use_vision and hasattr(self, "processor"):
                    # VISION PATH (VLM): message content can include image + text parts,
                    # BUT we only pass the rendered string to the processor along with image separately.
                    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
                    # Render chat to string (important: tokenize=False -> string)
                    prompt_text = self.processor.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    # Build inputs with processor (image provided separately)
                    inputs = self.processor(
                        obs_img, prompt_text, return_tensors="pt", add_special_tokens=False
                    ).to(self.model.device)

                else:
                    # TEXT-ONLY PATH (Qwen tokenizer): content MUST be a string
                    messages = [{"role": "user", "content": question}]
                    # Render chat to string (tokenize=False ensures str, not token-ids)
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    # Tokenize the rendered string
                    inputs = self.tokenizer(
                        prompt_text, return_tensors="pt", add_special_tokens=False
                    ).to(self.model.device)

                # Generate one step with your policy
                values, io_dict, output_text, action_log_prob, _ = self.actor_critic.act_oneline(inputs, obs_img)

            # After building output_text for this step:
            if step == steps:
                next_values = values
                self.rollouts_ood.obs[-1]['io_dict'] = io_dict
                break

            # Step the OOD env with the model's output
            obs, reward, done, truncated, info = self.ood_env.step(output_text)

            # Insert into OOD rollouts (keep your shapes/types)
            self.rollouts_ood.insert(
                {"image": obs, "io_dict": io_dict},  # obs dict
                None,  # output_ids placeholder
                torch.tensor([0]),  # actions placeholder (unused)
                action_log_prob,  # old_action_log_probs
                values.squeeze(),  # value_pred
                reward,  # reward
                torch.tensor([1 - done]),  # masks
                torch.tensor([1 - truncated])  # truncated masks
            )

            pbar.update()
            # Episode end handling
            if done or truncated:
                obs, info = self.ood_env.reset()
                obs_img = Image.fromarray(obs) if (self.use_vision and isinstance(obs, np.ndarray)) else (
                    obs if self.use_vision else None)

        pbar.close()
        self.ood_obs, self.ood_info = obs, info
        return next_values

    def collect_trajectories(self):
        self.stat.reset()
        running_reward = 0
        obs, info = self.obs, self.info

        prompts, patterns = self.prompt_language, self.pattern_language
        pbar = progress_bar(self.num_steps, "Collecting Trajectories", "blue",
                            self.accelerator)
        for step in range(self.num_steps + 1):
            v_dict, l_dict = {}, {}
            self.formulate_vision_arguments(v_dict, info)  # keeps cards meta

            with torch.no_grad():
                for pr, pat in zip(prompts, patterns):
                    self.formulate_prompt(v_dict, l_dict, pr, info=info)

                    chat_text = self.tokenizer.apply_chat_template(self.payload, add_generation_prompt=True, tokenize=False)
                    inputs = self.tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

                    val, io_dict, out_text, act_logp, tok_logp = \
                        self.actor_critic.act_oneline(inputs, None)

                    self.append_intermidiate_res(out_text)

            current_formula = re_match(out_text, 'formula')
            try:
                current_formula = current_formula.split('=')[0]
            except Exception:
                pass

            if step == self.num_steps:
                next_values = val
                self.rollouts.obs[-1]['io_dict'] = io_dict
                break

            obs, reward, done, truncated, info = self.env.step(out_text)
            running_reward += reward

            self.rollouts.insert({"image": obs, "io_dict": io_dict},
                                 None, torch.tensor([0]), act_logp,
                                 val.squeeze(), reward,
                                 torch.tensor([1 - done]),
                                 torch.tensor([1 - truncated]))

            self.stat.log_step(reward, done or truncated or
                               not self.enable_verification)
            if done or truncated or not self.enable_verification:
                if 'gym_virl' in self.id:
                    self.stat.log_virl_success(info['is_success'])
                self.stat.insert_running_reward(running_reward)
                self.stat.insert_action_tokens_log_prob(tok_logp.item())
                running_reward = 0
                obs, info = self.env.reset()

            pbar.update()
        pbar.close()

        return next_values, obs, info
    
    def append_intermidiate_res(self, res):
        self.payload.append({"role": "assistant", "content": res})
    
    def extract_final_action(self, language_res_dict):
        # task specific implementation
        return language_res_dict['formula']
    
    # def save_model(self, output_dir):
    #     if self.accelerator.is_main_process:
    #         torch.cuda.synchronize()
    #         unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
    #         mllm_model = unwrapped_model.value_model.base
    #         # print(unwrapped_model)
    #         mllm_model.save_pretrained(output_dir, safe_serialization = True, max_shard_size="8GB")
    #         self.processor.save_pretrained(output_dir)

    def save_model(self, output_dir):
        if self.accelerator.is_main_process:
            torch.cuda.synchronize()
            unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
            mllm_model = unwrapped_model.value_model.base
            mllm_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="8GB")
            (getattr(self, "processor", None) or getattr(self, "tokenizer")).save_pretrained(output_dir)

    @torch.no_grad()
    def compute_sft_ce(self, obs, info, use_ood: bool = False):
        """
        Compute an SFT-style cross-entropy on the assistant target only.
        Returns float CE or None if no target/invalid.
        """
        # 0) Need a target (pick first solution)
        sols = info.get("Solution", [])
        if not sols:
            return None
        target_formula = sols[0]

        # 1) Build user message exactly like your training prompt
        vision_res_dict, language_res_dict = {}, {}
        self.formulate_vision_arguments(vision_res_dict, info)

        args = self.oracle_arguments_ood if use_ood and hasattr(self, "oracle_arguments_ood") else self.oracle_arguments
        prompt_tmpl = self.prompt_vision[0] if self.use_vision else self.prompt_language[0]
        question = prompt_tmpl.format(**vision_res_dict, **language_res_dict, **args)

        # 2) Build assistant target text (ground truth)
        cards_plain = info.get("Plain Cards", [])
        numbers = info.get("Numbers", [])
        target_text = (
            "{\n"
            f"  \"cards\": {cards_plain},\n"
            f"  \"number\": {numbers},\n"
            f"  \"formula\": \"{target_formula}=24\",\n"
            "}"
        )

        # 3) Compose chat messages
        # TEXT-ONLY schema (Qwen tokenizer): content must be a STRING
        messages = [{"role": "user", "content": question}]
        messages_full = messages + [{"role": "assistant", "content": target_text}]

        # 4) Tokenize prompt-only and full convo
        if self.use_vision and hasattr(self, "processor"):
            # VLM path: go through the processor
            img = Image.fromarray(obs) if isinstance(obs, np.ndarray) else obs

            # For processors, use the *tokenizer* inside to build the template string
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            full_text = self.processor.tokenizer.apply_chat_template(
                messages_full, add_generation_prompt=False, tokenize=False
            )

            prompt_inputs = self.processor(
                img, prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(self.model.device)
            full_inputs = self.processor(
                img, full_text, return_tensors="pt", add_special_tokens=False
            ).to(self.model.device)

        else:
            # TEXT path: use the Qwen tokenizer directly
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            full_text = self.tokenizer.apply_chat_template(
                messages_full, add_generation_prompt=False, tokenize=False
            )
            # Ensure we really got strings (the original error)
            if not isinstance(prompt_text, str) or not isinstance(full_text, str):
                raise TypeError(
                    f"apply_chat_template must return str with tokenize=False, got "
                    f"{type(prompt_text)} and {type(full_text)}"
                )

            prompt_inputs = self.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(self.model.device)
            full_inputs = self.tokenizer(
                full_text, return_tensors="pt", add_special_tokens=False
            ).to(self.model.device)

        input_ids = full_inputs["input_ids"]
        attn_mask = full_inputs.get("attention_mask", None)
        prompt_len = prompt_inputs["input_ids"].size(1)

        # Guard: if assistant has no tokens (unlikely, but safe)
        if input_ids.size(1) <= prompt_len:
            return None

        # 5) Labels: mask prompt with -100 so CE is computed only on assistant span
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        # 6) Forward through base LM with labels (HF returns mean CE over non-ignored tokens)
        lm = self.actor_critic.value_model.base
        outputs = lm(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        return float(outputs.loss.detach().cpu().item())

    def train_one_epoch(self, save_model=False, update=0):
        next_values, self.obs, self.info = self.collect_trajectories()
        # ce_id = self.compute_sft_ce(self.obs, self.info)
        self.rollouts.compute_returns(next_values, **self.compute_return_kwargs)
        value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

        # ran_ood = hasattr(self, "rollouts_ood")
        # if ran_ood:
        #     ood_next_values = self.collect_trajectories_ood(self.ood_num_steps)
        #     self.rollouts_ood.compute_returns(ood_next_values, **self.compute_return_kwargs)
        #     ood_value_loss, ood_action_loss, _ = self.agent.evaluate(self.rollouts_ood)
        #     ce_ood = self.compute_sft_ce(self.ood_obs, self.ood_info)

        self.lr_scheduler.step()
        episode_rewards = self.stat.running_reward
        episode_action_tokens_log_prob = self.stat.action_tokens_log_prob
        self.rollouts.after_update()
        self.total_num_steps += self.num_steps * self.accelerator.num_processes
        if save_model:
            torch.cuda.empty_cache()
            self.save_model(os.path.join(self.output_dir, f"checkpoint-epoch-{update}"))

        ep = list(episode_rewards)
        log_str = {
            'total_num_steps': self.total_num_steps,
            'compute_tokens': self.actor_critic.token_cnt,
            'inference_fwd': self.actor_critic.called_inference_time,
            'bp_forward': self.actor_critic.called_bp_time,
            'value_loss': value_loss,
            'action_loss': action_loss,
            # 'cross-entropy': ce_id,
            'dist_entropy': dist_entropy,
            'reward.mean': self.rollouts.rewards.mean().item(),
            'reward.std': self.rollouts.rewards.std().item(),
            'reward.max': self.rollouts.rewards.max().item(),
            'reward.min': self.rollouts.rewards.min().item(),
            'value.mean': self.rollouts.value_preds.mean().item(),
            'value.std': self.rollouts.value_preds.std().item(),
            'value.max': self.rollouts.value_preds.max().item(),
            'value.min': self.rollouts.value_preds.min().item(),
            'return.mean': self.rollouts.returns.mean().item(),
            'return.std': self.rollouts.returns.std().item(),
            'return.max': self.rollouts.returns.max().item(),
            'return.min': self.rollouts.returns.min().item(),
            'episode_rewards.mean': float(np.mean(ep)) if len(ep) else float('nan'),
            'episode_rewards.std': float(np.std(ep)) if len(ep) else float('nan'),
            'episode_rewards.max': float(np.max(ep)) if len(ep) else float('nan'),
            'episode_rewards.min': float(np.min(ep)) if len(ep) else float('nan'),
            'episode_action_tokens_log_prob.mean': np.mean(episode_action_tokens_log_prob),
            'recog_acc': self.stat.cal_vision_acc(),
            'success_rate': self.stat.cal_success_rate(),
        }

        # if ran_ood:
        #     log_str.update({
        #         'ood.value_loss': ood_value_loss,
        #         'ood.action_loss': ood_action_loss,
        #         'ood.reward.mean': self.rollouts_ood.rewards.mean().item(),
        #         'ood.return.mean': self.rollouts_ood.returns.mean().item(),
        #         'ood.cross-entropy': ce_ood,
        #     })
        #     self.accelerator.print(
        #         f"[ID] v={value_loss:.4f} a={action_loss:.4f} | "
        #         f"[OOD({self.ood_num_steps})] v={ood_value_loss:.4f} a={ood_action_loss:.4f}"
        #     )
        #     self.rollouts_ood.after_update()  # clear OOD buffer
        #
        #     self.accelerator.print(
        #         f"[ID] vloss={value_loss:.4f} aloss={action_loss:.4f} | "
        #         f"[OOD] vloss={ood_value_loss:.4f} aloss={ood_action_loss:.4f}"
        #     )

        if self.report_to == 'wandb':
            wandb.log(log_str)
        print(log_str)