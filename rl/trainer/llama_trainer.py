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
import gymnasium as gym

class LlamaTrainer(BaseTrainer):
    def __init__(self, 
            action_space: Optional[List[Any]], \
            daytime: str, \
            accelerator: accelerate.Accelerator, \
            optimizer_config, ppo_config, compute_return_kwargs, \
            num_steps, num_updates, \
            env_config, \
            model, model_path, \
            prompt_config, generation_config, \
            output_dir, ood_num_steps=10, seed=42, report_to=None, run_name = 'default', save_ckpt=False, **kwargs):

        self.ood_num_steps = ood_num_steps
        super(LlamaTrainer, self).__init__(action_space, daytime, accelerator, optimizer_config, ppo_config, compute_return_kwargs, num_steps, num_updates, env_config, model, model_path, prompt_config, generation_config, output_dir, seed, report_to, run_name, save_ckpt, **kwargs)

    
    
    def init_model_optimizer_algo(self, model, model_path, ppo_config, optimizer_config):
        self.processor, self.model = evaluate_model_config(model, model_path)
        
        # this is a naive value model containing base model + linear layer
        value_model: nn.Module = VLMValue(self.model)
        actor_critic: nn.Module = VLMPolicy(tokenizer = self.processor,
                                value_model = value_model, 
                                generation_config = self.generation_config)

        optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=optimizer_config.init_lr, eps=optimizer_config.eps, weight_decay=optimizer_config.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimizer_config.lr_max_steps, eta_min=optimizer_config.end_lr)
        
        # AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
        ds_plugin = AcceleratorState().deepspeed_plugin
        if ds_plugin is not None:
            ds_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
        else:
            print("Warning: DeepSpeed plugin is not initialized. Skipping DeepSpeed configuration update.")
        self.actor_critic, self.optimizer, self.lr_scheduler = self.accelerator.prepare(actor_critic, optimizer, lr_scheduler)
        self.agent = algo.PPO(
            self.actor_critic,
            self.optimizer,
            self.accelerator,
            **ppo_config
        )
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
        
    def formulate_payload(self, question, obs = None):
        self.payload = [
            {
            "role": "user",
            "content": [{"type": "text", "text": question}]
            }
        ]
        if obs is not None:
            # append image to payload
            self.payload[0]['content'].insert(0, {"type": "image", "image": obs})

    
    def formulate_prompt(self, vision_res_dict, language_res_dict, prompt = None, obs = None, info = None):
        # task specific implementation
        if 'gym_cards' in self.id:
            if info['Verify Info'] is not None:
                self.payload.append({"role": "user", "content": [{"type": "text", "text": f"You failed this trial because {info['Verify Info']}"}]})
            else:
                # for prompt, pattern in zip(self.prompt_language, self.pattern_language):
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
                self.formulate_payload(question, obs)
        elif 'gym_virl' in self.id:
            if info['Verify Info'] is None or 'Correct action' in info['Verify Info'] or 'step_limit_reached"' in info['Verify Info']:
                
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
                self.formulate_payload(question, obs)
            else:
                self.payload.append({"role": "user", "content": [{"type": "text", "text": f"You failed this trial because {info['Verify Info']}"}]})

    @torch.no_grad()
    def collect_trajectories_ood(self, steps=None):
        steps = steps or self.ood_num_steps
        obs, info = self.ood_obs, self.ood_info

        # pick prompts/patterns for OOD
        if self.use_vision:
            prompts, patterns = (self.ood_prompts or self.prompt_vision), self.pattern_vision
        else:
            prompts, patterns = (self.ood_prompts or self.prompt_language), self.pattern_language

        pbar = progress_bar(steps, "Collecting OOD Trajectories", "magenta", self.accelerator)
        for step in range(steps + 1):
            vision_res_dict, language_res_dict = {}, {}
            self.formulate_vision_arguments(vision_res_dict, info)

            obs_img = None if not self.use_vision else obs
            if isinstance(obs_img, np.ndarray):
                obs_img = Image.fromarray(obs_img)

            # Build payload & generate exactly like ID, but use OOD oracle args
            for prompt, pattern in zip(prompts, patterns):
                # Use OOD oracle arguments here:
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments_ood)
                self.payload = [{"role": "user", "content": []}]
                if obs_img is not None:
                    self.payload[0]['content'].append({"type": "image", "image": obs_img})
                self.payload[0]['content'].append({"type": "text", "text": question})

                input_text = self.processor.apply_chat_template(self.payload, add_generation_prompt=True)
                inputs = self.processor(obs_img, input_text, return_tensors="pt", add_special_tokens=False).to(
                    self.model.device)
                values, io_dict, output_text, action_log_prob, _ = self.actor_critic.act_oneline(inputs, obs_img)

            if step == steps:
                next_values = values
                self.rollouts_ood.obs[-1]['io_dict'] = io_dict
                break

            obs, reward, done, truncated, info = self.ood_env.step(output_text)
            self.rollouts_ood.insert({"image": obs, "io_dict": io_dict}, None, torch.tensor([0]),
                                     action_log_prob, values.squeeze(), reward,
                                     torch.Tensor([1 - done]), torch.Tensor([1 - truncated]))
            pbar.update()
            if done or truncated:
                obs, info = self.ood_env.reset()

        pbar.close()
        self.ood_obs, self.ood_info = obs, info
        return next_values

    def collect_trajectories(self):
        self.stat.reset()
        running_reward = 0
        obs = self.obs
        info = self.info
        if self.use_vision:
            prompts, patterns = self.prompt_vision, self.pattern_vision
        else:
            prompts, patterns = self.prompt_language, self.pattern_language
        pbar = progress_bar(self.num_steps, f"Collecting Trajectories", "blue", self.accelerator)
        for step in range(self.num_steps + 1):
            vision_res_dict = {}
            language_res_dict = {}
            self.formulate_vision_arguments(vision_res_dict, info)
            
            with torch.no_grad():
                obs = None if not self.use_vision else obs
                if isinstance(obs, np.ndarray):
                    obs = Image.fromarray(obs)

                for prompt, pattern in zip(prompts, patterns):
                    self.formulate_prompt(vision_res_dict, language_res_dict, prompt = prompt, obs = obs, info = info)
                    input_text = self.processor.apply_chat_template(self.payload, add_generation_prompt=True)
                    inputs = self.processor(obs, input_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                    values, io_dict, output_text, action_log_prob, action_tokens_log_prob = self.actor_critic.act_oneline(inputs, obs)
                    self.append_intermidiate_res(output_text)
            current_formula = re_match(output_text, 'formula')
            try:
                current_formula = current_formula.split('=')[0]
            except:
                pass
            if step == self.num_steps:
                next_values = values
                self.rollouts.obs[-1]['io_dict'] = io_dict
                break
            if step == self.num_steps - 1:
                print("Running example")
                print("Input: ")
                print(input_text)
                print("Output: ")
                print(output_text)
                print("Formula: ")
                print(current_formula)
                print("Action log prob: ")
                print(action_log_prob)
                print("Action tokens log prob: ")
                print(action_tokens_log_prob)
            obs, reward, done, truncated, info = self.env.step(output_text)
            # print(info["Verify Info"])
            running_reward += reward
            self.rollouts.insert({"image": obs, "io_dict": io_dict}, None, torch.tensor([0]), action_log_prob, values.squeeze(), reward, torch.Tensor([1-done]), torch.Tensor([1-truncated]))
            self.stat.log_step(reward, done or truncated or not self.enable_verification)
            if done or truncated or not self.enable_verification:
                if 'gym_virl' in self.id:
                    self.stat.log_virl_success(info['is_success'])
                self.stat.insert_running_reward(running_reward)
                self.stat.insert_action_tokens_log_prob(action_tokens_log_prob.item())
                running_reward = 0
                obs, info = self.env.reset()
            pbar.update()
        pbar.close()
        
        return next_values, obs, info

    def append_intermidiate_res(self, res):
        # task specific implementation
        self.payload.append({"role": "assistant", "content": [{"type": "text", "text": res}]})
    
    def extract_final_action(self, language_res_dict):
        # task specific implementation
        return language_res_dict['formula']
    
    def save_model(self, output_dir):
        if self.accelerator.is_main_process:
            torch.cuda.synchronize()
            unwrapped_model = self.accelerator.unwrap_model(self.actor_critic)
            mllm_model = unwrapped_model.value_model.base
            # print(unwrapped_model)
            mllm_model.save_pretrained(output_dir, safe_serialization = True, max_shard_size="8GB")
            self.processor.save_pretrained(output_dir)

    @torch.no_grad()
    def compute_sft_ce(self, obs, info):
        """
        Returns: float cross-entropy (averaged over assistant tokens), or None if no target.
        """
        # 0) Need a target formula
        sols = info.get("Solution", [])
        if not sols:
            return None
        target_formula = sols[0]  # pick the first valid solution

        # 1) Build the same user prompt you use in RL
        # Reuse your existing prompt-building path so tokens match.
        # We'll use the language-only path for clarity; adapt if you're using vision prompts.
        vision_res_dict, language_res_dict = {}, {}
        self.formulate_vision_arguments(vision_res_dict, info)

        # Choose the same prompt/pattern you used for this env (take the first as a proxy)
        if self.use_vision:
            prompt = self.prompt_vision[0]
        else:
            prompt = self.prompt_language[0]

        # Build the user message and (optionally) include the image
        self.payload = [{"role": "user", "content": []}]
        obs_img = None if not self.use_vision else obs
        if isinstance(obs_img, np.ndarray):
            obs_img = Image.fromarray(obs_img)
        if obs_img is not None:
            self.payload[0]["content"].append({"type": "image", "image": obs_img})

        # Fill the user question using the *ID* oracle arguments here; if you want OOD CE,
        # swap for self.oracle_arguments_ood.
        question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
        self.payload[0]["content"].append({"type": "text", "text": question})

        # 2) Build the assistant *target* text (canonical JSON like your examples)
        cards_plain = info.get("Plain Cards", [])
        numbers = info.get("Numbers", [])
        target_text = (
            "{\n"
            f"  \"cards\": {cards_plain},\n"
            f"  \"number\": {numbers},\n"
            f"  \"formula\": \"{target_formula}=24\",\n"
            "}"
        )

        # 3) Tokenize prompt-only (system+user) to get the split point
        prompt_text = self.processor.apply_chat_template(self.payload, add_generation_prompt=True)
        prompt_inputs = self.processor(
            obs_img, prompt_text, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        prompt_len = prompt_inputs["input_ids"].size(1)

        # 4) Tokenize full conversation (user + assistant target)
        payload_full = list(self.payload) + [{
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        }]
        full_text = self.processor.apply_chat_template(payload_full, add_generation_prompt=False)
        full_inputs = self.processor(
            obs_img, full_text, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)

        input_ids = full_inputs["input_ids"]
        attn_mask = full_inputs.get("attention_mask", None)

        # 5) Build labels: mask prompt tokens with -100 so loss is over assistant tokens only
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        # 6) Forward through the *base* LM with labels to get CE (exactly like Trainer)
        # If your model is wrapped, use the underlying base LM head:
        lm = self.actor_critic.value_model.base  # same object you save_pretrained(...)
        outputs = lm(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        # HF returns mean CE over non-ignored tokens
        return float(outputs.loss.item())

    def train_one_epoch(self, save_model = False, update = 0):
        next_values, self.obs, self.info = self.collect_trajectories()
        #For loss during RL fine-tuning
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