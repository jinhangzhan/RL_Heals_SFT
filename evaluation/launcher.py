# print("sys.path:", sys.path)
import sys
# sys.path.insert(0, "/path/RL_Heals_SFT")
# sys.path.insert(1, "/path/RL_Heals_SFT/gym")

import yaml
from box import Box
from evaluation.evaluator import evaluator_init
from tqdm import tqdm
import accelerate
import wandb
import datetime

from utils_general import load_config
from utils_rl import set_str_action_space
import gym_cards
import importlib
import gym_virl
# import virl
# import virl.utils
# import virl.utils.common_utils
importlib.reload(gym_cards)
importlib.reload(gym_virl)
# importlib.reload(virl)
# importlib.reload(virl.utils)
# importlib.reload(virl.utils.common_utils)

# from datasets import load_dataset
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json

def main(config, boxed_config):
    action_space = set_str_action_space(boxed_config.env_config)

    print(yaml.dump(config))
    daytime =  datetime.datetime.now().strftime("%Y-%m-%d||%H:%M:%S")
    player = evaluator_init[boxed_config.evaluator](action_space = action_space, daytime=daytime, **boxed_config)
    if getattr(boxed_config, 'best_of_n_baseline', 0) > 0:
        player.baseline(getattr(boxed_config, 'best_of_n_baseline', 0))
    else:
        player.evaluate()

def main_gp(config, boxed_config):
    action_space = set_str_action_space(boxed_config.env_config)

    print(yaml.dump(config))
    daytime = datetime.datetime.now().strftime("%Y-%m-%d||%H:%M:%S")
    player = evaluator_init[boxed_config.evaluator](action_space = action_space, daytime=daytime, **boxed_config)
    if getattr(boxed_config, 'best_of_n_baseline', 0) > 0:
        player.baseline(getattr(boxed_config, 'best_of_n_baseline', 0))
    else:
        # player.evaluate()
        output_json_path = '/path/eval-multi-answer.json'

        records = []
        current_index = 0
        seen_cards = set()
        for i in range(300):
            print(i)
            obs, info = player.env.reset(seed=0)
            input_cards = info.get("Plain Cards", [])
            cards_key = tuple(input_cards)
            if cards_key in seen_cards:
                continue
            seen_cards.add(cards_key)
            r, ind = generate_ood_dataset(info, current_index)
            records.append(r[0])
            current_index = ind + current_index
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)

def main_virl(config, boxed_config):
    action_space = set_str_action_space(boxed_config.env_config)

    print(yaml.dump(config))
    daytime = datetime.datetime.now().strftime("%Y-%m-%d||%H:%M:%S")
    player = evaluator_init[boxed_config.evaluator](action_space = action_space, daytime=daytime, **boxed_config)
    if getattr(boxed_config, 'best_of_n_baseline', 0) > 0:
        player.baseline(getattr(boxed_config, 'best_of_n_baseline', 0))
    else:
        # player.evaluate()
        output_json_path = '/path/ood-data-300.json'

        records = []

        current_index = 0
        seen_cards = set()
        for i in range(300):
            print(i)
            obs, info = player.env.reset(seed=42)
            input_cards = info.get('obs_act_seq') + info.get('current_instruction') + info.get("global_instruction")
            cards_key = tuple(input_cards)
            if cards_key in seen_cards:
                continue
            seen_cards.add(cards_key)
            r, ind = generate_ood_dataset_virl(player, info, current_index)
            records.append(r)
            current_index = ind + current_index
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)

def generate_ood_dataset(meta, current_index):

    input_cards = meta.get("Plain Cards", [])
    numbers = meta.get("Numbers", [])
    numbers_str = [str(n) for n in numbers]

    human_prompt = (
            "\n[Task Description]\n"
            "You are an expert 24 points card game player. You will receive a set of 4 cards.\n"
            "Note that 'J', 'Q', and 'K' count as '11', '12', '13', and each card must be used once.\n"
            "Your goal is to output a formula that evaluates to 24 using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.\n\n"
            "[Input]\n"
            "Cards: " + str(input_cards) + "\n\n"
                                           "[Output]\n"
                                           "{\n"
                                           "  \"cards\": [x, y, z, w], where 'J', 'Q', and 'K' count as '11','12','13',\n"
                                           "  \"number\": [a, b, c, d], where a, b, c, and d are the numbers on the cards,\n"
                                           "  \"formula\": 'an equation that equals 24',\n"
                                           "}\n\n"
    )

    records = []
    solutions = meta.get("Solution", [])
    for idx, sol in enumerate(solutions):
        record = {
            "id": str(idx + current_index),
            "image": None,
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "gpt",
                    "value": (
                            "\n{\n"
                            "  \"cards\": " + str(input_cards) + ",\n"
                                                                 "  \"number\": " + str(numbers_str) + ",\n"
                                                                                                       "  \"formula\": \"" + sol + "\"\n"
                                                                                                                                   "}\n"
                    )
                }
            ]
        }
        records.append(record)
    return records, len(solutions)
    # print(f"Generated {len(records)} records and saved to {output_json_path}")

Q_VIRL_L = """
[Task Description]
You are an expert in navigation. You will receive a sequence of instructions to follow. You
are also provided with your observation and action history in text. Your goal is to first analyze the instruction and identify the next sentence to be executed. 
Then, you need to provide the action to be taken based on the current observation and instruction.

[Instruction]
{instruction}

[Action space]
{action_space}

[Observations and actions sequence]
{obs_act_seq}

[Output]
{{
  "current observation": latest observation from the observation sequence,
  "current instruction": analyze the full instruction and identify the sentence to be executed,
  "action": the action to be taken chosen from the action space,
}}
"""

def generate_ood_dataset_virl(evaluator, info, index_id):

    global_instruction = info.get("global_instruction", "")
    obs_act_seq = info.get("obs_act_seq", "")
    current_obs = info.get("current_obs", "")
    action_space_str = evaluator.action_space

    prompt = Q_VIRL_L.format(
        instruction=global_instruction,
        action_space=action_space_str,
        obs_act_seq=obs_act_seq
    )

    evaluator.payload = []
    evaluator.prompt_language = [prompt]
    evaluator.pattern_language = ["action"]

    vision_res_dict = {}
    language_res_dict = {}
    # evaluator.generate_one_response(vision_res_dict, language_res_dict, obs=None, info=info)

    # model_answer = language_res_dict.get("action", "")
    gpt_value = {
        "current observation": info['current_obs'],
        "current instruction": info['current_instruction'],
        "action": info['gt_action']
    }

    gpt_value_str = "\n" + json.dumps(gpt_value, indent=2, ensure_ascii=False) + "\n"
    record = {
        "id": str(index_id),
        "image": None,
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": gpt_value_str
            }
        ]
    }

    return record, 1

if __name__ == "__main__":
    print('test========================')
    config, boxed_config = load_config()
    print(config)
    main(config, boxed_config)
