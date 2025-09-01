"""
Here we define the base class for all evaluators.
This class contained basic methods that all evaluators should have.
Updated: Oct-11-2024
By Tianzhe
"""

import gymnasium as gym
# import gym_cards
# import gym_virl
# import torch
from utils_general import StatLogger, process_formula, re_match, init_seed
from utils_rl import REWARD_FN, REWARD_FN_VIRL
from prompt_lib import PROMPT_FN
import os
import json
from tqdm import tqdm
import accelerate
class BaseEvaluator:
    def __init__(self, action_space, daytime, env_config, prompt_config, generation_config, output_dir, seed=42, num_traj=1, order=False, **kwargs):
        # first set random seed
        self.num_traj = num_traj
        self.seed = seed
        init_seed(seed)
        
        # env related settings
        self.env = gym.make(**env_config, language_only=prompt_config.use_language)
        self.order = order
        self.id = env_config.id
        self.action_space = action_space
        self.accelerator = accelerate.Accelerator()
        if 'gym_cards' in env_config.id:
            self.target_number = env_config.target_points
            self.treat_face_cards_as_10 = env_config.treat_face_cards_as_10
        
        
        # statistics and logging
        if 'gym_cards' in env_config.id:
            self.stat = StatLogger(reward_fn=REWARD_FN)
        elif 'gym_virl' in env_config.id:
            self.stat = StatLogger(reward_fn=REWARD_FN_VIRL)
        else:
            raise NotImplementedError
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

        if not os.path.isdir(output_dir):
            self.output_dir = output_dir.replace('.', f'_{daytime}.')
        else:
            self.output_dir = os.path.join(output_dir, f'output_{daytime}/')
            os.makedirs(self.output_dir, exist_ok=True)
        
        # prompt related settings
        self.enable_verification = prompt_config.enable_verification
        self.use_vision = prompt_config.use_vision
        self.use_language = prompt_config.use_language

        # this is a legacy implementation to support multiple-prompts, but works for signle-prompt cases
        if prompt_config.use_vision:
            self.prompt_vision = [PROMPT_FN[i] for i in prompt_config.prompt_vision]
            self.pattern_vision = prompt_config.pattern_vision
            assert len(self.prompt_vision) == len(self.prompt_vision), "Young man, check # of prompts & # of patterns in vision."
        
        if prompt_config.use_language:
            self.prompt_language = [PROMPT_FN[i] for i in prompt_config.prompt_language]
            self.pattern_language = prompt_config.pattern_language
            assert len(self.prompt_language) == len(self.prompt_language), "Young man, check # of prompts & # of patterns in language."
        
        self.generation_config = generation_config
        self.formulate_oracle_arguments()
        self.sample_num = 0
    
    def generate_one_response(self, vision_res_dict, language_res_dict, obs = None, info = None):
        # model specific implementation
        raise NotImplementedError

    def append_intermidiate_res(self, res):
        # model specific implementation
        raise NotImplementedError
    
    def extract_final_action(self, language_res_dict):
        # task specific implementation
        raise NotImplementedError
    
    def dump_stuff_into_json(self, stuff):
        # log results into a .jsonl file
        if self.accelerator.is_main_process:
            with open(self.output_dir, 'a') as f:
                json_str = json.dumps(stuff)
                f.write(json_str + '\n')
    
    def formulate_oracle_arguments(self):
        """
        Goal: 
        formulate static arguments into self.orcale_arguments

        Input:
            None
        Output:
            In-place modification 
        Usage:
            prompt.formate(**self.oracle_arguments)
        """
        self.oracle_arguments = {}
        if 'gym_cards' in self.id:
            self.oracle_arguments['valid_ops'] = self.action_space
            self.oracle_arguments['face_card_msg'] = "'J', 'Q', and 'K' count as '10'." if self.treat_face_cards_as_10 \
                                            else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
            self.oracle_arguments['target_number'] = str(self.target_number)
        elif 'gym_virl' in self.id:
            self.oracle_arguments['action_space'] = self.action_space
    
    def formulate_vision_arguments(self, vision_res_dict, info):
        """
        Goal:
        formulate dynamic, possibly vision-related arguments, into vision_res_dict

        Input:
            vision_res_dict: dictionary, usually empty
            info: dictionary, contains step-wise information from the environment
        Output:
            In-place modification
        Usage:
            prompt.formate(**vision_res_dict)
        """
        if 'gym_cards' in self.id:
            if 'cards' not in vision_res_dict.keys():
                # hard code gt cards into dict
                vision_res_dict['cards'] = info['Plain Cards']
        elif 'gym_virl' in self.id:
            vision_res_dict['instruction'] = info['global_instruction']
            vision_res_dict['obs_act_seq'] = info['obs_act_seq']
                
    def evaluate_one_trajectory(self, **kwargs):
        """
        Goal:
        evaluate one trajectory, with sequential revision (optional, enabled by config)

        Input:
            None
        Output:
            None
        Usage:
            write a set of evaluation results in self.output_dir
        """
        if not self.order:
            obs, info = self.env.reset(seed = self.seed)
        else:
            obs, info = self.env.reset(seed = self.seed, options={'route_idx': self.sample_num})
        running_reward = 0
        verification_step = 0
        while True:
            vision_res_dict = {}
            language_res_dict = {}
            if self.use_vision:
                self.generate_one_response(vision_res_dict, language_res_dict, obs, info)
                key_pattern = self.pattern_vision[-1]
            if self.use_language:
                self.generate_one_response(vision_res_dict, language_res_dict, None, info)
                key_pattern = self.pattern_language[-1]
            action = self.extract_final_action(language_res_dict, key_pattern)
            
            obs, reward, done, truncated, info = self.env.step(action)
            
            running_reward += reward

            self.stat.log_step(reward, done or truncated or not self.enable_verification)
            self.dump_stuff_into_json({"sample_id": self.sample_num, "veri_step": verification_step, "output": action, "reward": reward, "info": info}) 
            verification_step += 1
            if done or truncated or not self.enable_verification:
                self.stat.insert_running_reward(running_reward)
                if 'gym_virl' in self.id:
                    self.dump_stuff_into_json({"Success": info['is_success']})
                    self.stat.log_virl_success(info['is_success'])
                self.dump_stuff_into_json({"sample_id": self.sample_num, "output": action, "reward": running_reward, "info": info})
                self.dump_stuff_into_json({"Split": "===================="})
                running_reward = 0
                self.sample_num += 1
                break

    def baseline(self, n):
        # legacy function
        raise NotImplementedError
    
    def evaluate(self):
        """
        Goal:
        evaluate self.num_traj samples

        Input:
            None
        Output:
            None
        """
        for _ in tqdm(range(self.num_traj)):
            self.evaluate_one_trajectory()
        
        self.dump_stuff_into_json(self.stat.get_stat())
    
    