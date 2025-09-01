"""
Here we define the base class for all evaluators.
This class contained basic methods that all evaluators should have.
Updated: Oct-11-2024
By Tianzhe
"""

import gymnasium as gym
import gym_cards, gym_virl
import torch
from utils_general import StatLogger, process_formula, re_match, init_seed, progress_bar
from utils_rl import REWARD_FN, REWARD_FN_VIRL, set_str_action_space
from prompt_lib import PROMPT_FN
import os
import json
from tqdm import tqdm
import datetime
import accelerate
from typing import Optional, Dict, List, Any
import wandb
# from transformers import Trainer

class BaseTrainer:
    def __init__(self, 
            action_space: Optional[List[Any]], \
            daytime: str, \
            accelerator: accelerate.Accelerator, \
            optimizer_config, ppo_config, compute_return_kwargs, \
            num_steps, num_updates, \
            env_config, \
            model, model_path, \
            prompt_config, generation_config, \
            output_dir, \
            seed=42, report_to=None, run_name = 'default', save_ckpt=False, save_every=None, ood_env_config=None, ood_sample=None, ood_prompts=None, **kwargs):
        
        self.accelerator = accelerator
        self.save_ckpt = save_ckpt
        self.save_every = save_every
        init_seed(seed + self.accelerator.local_process_index)
        self.total_num_steps = 0 # used for logging
        self.env = gym.make(**env_config, language_only=prompt_config.use_language)

        self.action_space = action_space
        self.id = env_config.id
        if 'gym_cards' in env_config.id:
            self.target_number = env_config.target_points
            self.treat_face_cards_as_10 = env_config.treat_face_cards_as_10
        
        # statistics and logging
        if 'gym_cards' in env_config.id:
            self.stat = StatLogger(reward_fn=REWARD_FN)
        else:
            self.stat = StatLogger(reward_fn=REWARD_FN_VIRL)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        if not os.path.isdir(output_dir):
            self.output_dir = output_dir.replace('.', f'_{daytime}.')
        else:
            self.output_dir = os.path.join(output_dir, f'output_{daytime}/')
            os.makedirs(self.output_dir, exist_ok=True)
        # self.output_dir = output_dir.replace('.', f'process{self.accelerator.local_process_index}_{daytime}.')
        
        
        # prompt related settings
        self.enable_verification = prompt_config.enable_verification
        self.use_vision = prompt_config.use_vision
        self.use_language = prompt_config.use_language
        if prompt_config.use_vision:
            self.prompt_vision = [PROMPT_FN[i] for i in prompt_config.prompt_vision]
            self.pattern_vision = prompt_config.pattern_vision
            assert len(self.prompt_vision) == len(self.prompt_vision), "Young folk, check # of prompts & # of patterns in vision."
            
        if prompt_config.use_language:
            self.prompt_language = [PROMPT_FN[i] for i in prompt_config.prompt_language]
            self.pattern_language = prompt_config.pattern_language
            assert len(self.prompt_language) == len(self.prompt_language), "Young folk, check # of prompts & # of patterns in language."


        self.generation_config = generation_config
        self.formulate_oracle_arguments()
        
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.compute_return_kwargs = compute_return_kwargs
        self.device = self.accelerator.device
        self.report_to = report_to


        # setup for running ood eval
        if ood_env_config is not None: 
            self.ood_env = gym.make(**ood_env_config, language_only=prompt_config.use_language)
            self.ood_sample = ood_sample
            self.ood_prompts = [PROMPT_FN[i] for i in ood_prompts]
            self.action_space_ood = set_str_action_space(ood_env_config)
            self.formulate_orcale_arguments_ood()
            

        self.init_model_optimizer_algo(model, model_path, ppo_config, optimizer_config)
    
    def init_model_optimizer_algo(self, model, model_path, ppo_config, optimizer_config):
        raise NotImplementedError
    
    def formulate_orcale_arguments_ood(self):
        self.oracle_arguments_ood = {}
        if 'gym_cards' in self.id:
            self.oracle_arguments_ood['valid_ops'] = self.action_space_ood
            self.oracle_arguments_ood['face_card_msg'] = "'J', 'Q', and 'K' count as '10'." if not self.treat_face_cards_as_10 \
                                            else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
            self.oracle_arguments_ood['target_number'] = str(self.target_number)
        elif 'gym_virl' in self.id:
            self.oracle_arguments_ood['action_space'] = self.action_space_ood

    def formulate_oracle_arguments(self):
        self.oracle_arguments = {}
        if 'gym_cards' in self.id:
            self.oracle_arguments['valid_ops'] = self.action_space
            self.oracle_arguments['face_card_msg'] = "'J', 'Q', and 'K' count as '10'." if self.treat_face_cards_as_10 \
                                            else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
            self.oracle_arguments['target_number'] = str(self.target_number)
        elif 'gym_virl' in self.id:
            self.oracle_arguments['action_space'] = self.action_space
    
    def formulate_vision_arguments(self, vision_res_dict, info):
        # if 'formula' not in vision_res_dict.keys():
        #     # hard code gt formula into dict
        #     # this key is never used in oneline eqn setting
        #     vision_res_dict['formula'] = process_formula(info['Formula'])
        if 'gym_cards' in self.id:
            if 'cards' not in vision_res_dict.keys():
                # hard code gt cards into dict
                vision_res_dict['cards'] = info['Plain Cards']
        elif 'gym_virl' in self.id:
            vision_res_dict['instruction'] = info['global_instruction']
            vision_res_dict['obs_act_seq'] = info['obs_act_seq']
    
    def collect_trajectories(self):
        raise NotImplementedError
    
    def train_one_epoch(self, save_model = False):
        raise NotImplementedError
    
    def save_model(self, output_dir):
        raise NotImplementedError
    
    def train(self):
        pbar = progress_bar(self.num_updates, f"Training", "white", self.accelerator)
        for update in range(self.num_updates):
            if self.save_ckpt:
                save_model = (update + 1) % self.save_every == 0
            else:
                save_model = False
            self.train_one_epoch(save_model=save_model, update=update)
            print(self.stat)
            pbar.update()
        pbar.close()
    
    
    def wandb_log(self, log_dict):
        if self.report_to == "wandb":
            wandb.log(log_dict)