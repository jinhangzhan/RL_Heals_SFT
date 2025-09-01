"""
Here, we define a few functions related to:
 -configration
 -statistics
 -regular expression matching
 ...

Updated: Oct-11-2024
Last edited by Tianzhe
"""

import yaml
from box import Box
import argparse
from typing import Dict
import re
import json
import torch
import numpy as np
import random
from tqdm import tqdm
import transformers
"""
Init seed
"""
def init_seed(seed):
    """
    Set up random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
"""
Configuration handling functions
"""
def update_nested_dict(d, key_path, value):
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def is_valid_key(config, key_path):
    keys = key_path.split('.')
    d = config
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return False
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Script to launch with a specific configuration.")
    parser.add_argument('-f', type=str, required=True, help='Path to the YAML configuration file.')
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def load_config():
    known_args, unknown_args = parse_args()

    # Load the configuration file
    with open(known_args.f, 'r') as f:
        config = yaml.safe_load(f)

    # Now we can validate the unknown args against the config
    override_args = {}
    for arg in unknown_args:
        if arg.startswith('--'):
            key, value = arg[2:].split('=')
            if is_valid_key(config, key):
                override_args[key] = value
            else:
                print(f"Warning: Ignoring invalid configuration key: {key}")
                # assert False, f"Invalid configuration key: {key}"

    # Override configuration parameters if provided
    for key, value in override_args.items():
        # Convert value to appropriate type
        try:
            value = eval(value)
        except:
            pass
        update_nested_dict(config, key, value)
    
    # Print the updated configuration for debugging
    
    return config, Box(config)

"""
Statistics
"""

class StatLogger:
    def __init__(self, reward_fn: Dict[str, int]):
        self.reward_fn = reward_fn
        self.reward_fn_inv = {v: k for k, v in reward_fn.items()}
        self.reset()
    
    def reset(self):
        self.traj_num = 0
        self.step_num = 0
        self.running_reward = []
        self.action_tokens_log_prob = []
        self.to_be_returned = "None"
        self.log = {}
    def insert_action_tokens_log_prob(self, action_tokens_log_prob):
        self.action_tokens_log_prob.append(action_tokens_log_prob)

    def insert_running_reward(self, reward: int):
        self.running_reward.append(reward)
    
    def log_virl_success(self, success: bool):
        if success:
            self.log["CORRECT_SOLUTION"] = self.log.get("CORRECT_SOLUTION", 0) + 1
    
    def log_step(self, reward: int, done_or_truncated: bool):
        self.step_num += 1
        try:
            state_name = self.reward_fn_inv[reward]
            self.log[state_name] = self.log.get(state_name, 0) + 1
            if done_or_truncated:
                self.traj_num += 1
        except:
            self.log['AGGREGATED_ERROR'] = self.log.get('AGGREGATED_ERROR', 0) + 1
            if done_or_truncated:
                self.traj_num += 1
    def get_stat(self):
        print(self)
        return self.to_be_returned_dict
    
    def cal_success_rate(self):
        try:
            success_rate = self.log["CORRECT_SOLUTION"] / self.traj_num
        except:
            success_rate = 0
        return success_rate
    
    def cal_vision_acc(self):
        try:
            vision_acc = 1 - self.log["INCORRECT_VISION"] / self.step_num
        except:
            vision_acc = 1
        return vision_acc
    def __str__(self):
        rates = {k: round(v / self.step_num, 4) for k, v in self.log.items()}
        if "CORRECT_SOLUTION" in rates:
            rates["CORRECT_SOLUTION"] = round(self.log["CORRECT_SOLUTION"] / self.traj_num, 4)
        else:
            rates["CORRECT_SOLUTION"] = 0
        
        self.to_be_returned = f"Step Number: {self.step_num}\nTrajectory Number: {self.traj_num}\nRates: {rates}\n"
        self.to_be_returned_dict = {"Step Number": self.step_num, "Trajectory Number": self.traj_num, "Rates": rates}
        return self.to_be_returned

"""
Regular expression matching
"""

def process_formula(formula_list):
    str_to_ret = ""
    for i in formula_list:
        str_to_ret+=str(i)
    return str_to_ret

# some predefined patterns
RE_PATTERN_DICT = {
    "cards": r'"cards": \[([^\]]+)\]',
    "number": r'"number": \[([^\]]+)\]',
    "answer": r'"cards": \[([^\]]+)\]',
    "cards_remained": r'"cards_remained": \[([^\]]+)\]',
    "action": r'"action": "(.*?)"',
    "formula": r'"formula": "(.*?)"',
    "current observation": r'"current observation": "(.*?)"',
    "current instruction": r'"current instruction": "(.*?)"',
    
}
def re_match(text: str, pattern: str):
    
    try:
        output_dict = json.loads(text)
        pred = output_dict[pattern]
    except:
        try:
            pattern_re = re.search(RE_PATTERN_DICT[pattern], text)
            pred = pattern_re.group(1)
            # print("Pred in try 2:", pred)
            # handle cases for list
            if 'cards' in pattern:
                try:
                    pred = list(map(int, pattern_re.group(1).split(', ')))
                except:
                    pred = '[' + pred + ']'
            
        except:
            pred = "None"
    
    return pred

def robust_str_to_list(list_like_str: str):
    try:
        list_like_str = list_like_str.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(',')
        assert len(list_like_str) == 4
    except:
        list_like_str = []
    return list_like_str

def parse_navigation_string(input_string):
    # Initialize result list
    result = []
    
    # Split the input string into individual dictionary blocks
    # Remove any leading/trailing whitespace and filter out empty strings
    blocks = [block.strip() for block in input_string.split('}') if block.strip()]
    
    for block in blocks:
        # Remove leading '{' and any whitespace
        block = block.lstrip('{').strip()
        
        # Skip empty blocks
        if not block:
            continue
            
        # Initialize dictionary for current block
        current_dict = {}
        
        # Split into lines and process each line
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        for line in lines:
            # Remove any trailing comma
            line = line.rstrip(',')
            
            # Split by colon and handle potential formatting issues
            try:
                key, value = [part.strip() for part in line.split(':', 1)]
                
                # Clean up the key and value
                key = key.strip('"\'')
                value = value.strip('"\'')
                
                current_dict[key] = value
            except ValueError:
                continue  # Skip malformed lines
                
        # Add non-empty dictionaries to result
        if current_dict:
            result.append(current_dict)
    
    return result

def parse_direction_string(string):
    """
    parse "north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest" out of the string
    """
    DIRECTION_LIST = ['northeast', 'northwest', 'southeast', 'southwest', 'north', 'south', 'east', 'west']
    for direction in DIRECTION_LIST:
        if direction in string:
            return direction
    
    return random.choice(DIRECTION_LIST)
"""
Init a colorful progress bar
"""

def progress_bar(total_steps, desc, color, accelerator):
    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, desc=desc, colour=color)
        return pbar
    else:
        class DummyProgressBar:
            def update(self, *args, **kwargs):
                pass
            def close(self, *args, **kwargs):
                pass
        return DummyProgressBar()


"""
save function
"""
# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
#                                    output_dir: str):
#     """Collects the state dict and dump to disk."""

#     if trainer.deepspeed:
#         torch.cuda.synchronize()
#         trainer.save_model(output_dir)
#         return

#     state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         cpu_state_dict = {
#             key: value.cpu()
#             for key, value in state_dict.items()
#         }
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
#         trainer.model.config.save_pretrained(output_dir)


if __name__ == "__main__":
    # Test stat logger
    reward_fn = {"ILLEGAL_FORMULA": -10, "INCORRECT_NUMBER": -5, "NO_SOLUTION": -1, "PARTIAL_SOLUTION": 1, "CORRECT_SOLUTION": 5}
    stat_logger = StatLogger(reward_fn)
    stat_logger.log_step(-10, False)
    print(stat_logger)
    stat_logger.log_step(-7, False)
    print(stat_logger)
    stat_logger.log_step(5, True)
    print(stat_logger)