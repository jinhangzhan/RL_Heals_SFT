"""
Here, we define functions for RL, environment related util functions.
Updated: Oct-9-2024
By Tianzhe
"""

import re
from dataclasses import dataclass
from typing import List
from collections import Counter
from utils_general import re_match
"""
Naive reward function and step verifier for p24
"""
REWARD_FN = {
    "CORRECT_SOLUTION": 5,
    "PARTIAL_SOLUTION": 1,
    "NO_SOLUTION": -1,
    "INCORRECT_VISION": -1.5,
    "INCORRECT_NUMBER": -2,
    "ILLEGAL_FORMULA": -3,
}

def aggregate_fn(message_list):
    # aggregate message into a string with comma separated and ending with '.'
    
    return ', '.join(message_list) + '.'

@dataclass
class StepResult:
    reward: int
    message_list: List[str]

    def __iter__(self):
        return iter((self.reward, aggregate_fn(self.message_list)))
    
    def add(self, reward, message):
        self.reward += reward
        self.message_list.append(message)

def step_rewards(card_nums: list,
                 current_formula: str,
                 solutions: list,
                 target_points: int, 
                 recognized_cards: list = None,
                 translated_number: list = None,
                 gt_cards: list = None,
                 language_only: bool = True,
                 ) -> int:
    
    # print(recognized_cards, translated_number, gt_cards, card_nums)
    if not language_only:
        sorted_card_nums = sorted(card_nums)
        sorted_recognized_cards = sorted(recognized_cards)
        sorted_translated_number = sorted(translated_number)
        sorted_gt_cards = sorted(gt_cards)
        if sorted_gt_cards != sorted_recognized_cards:
            return REWARD_FN["INCORRECT_VISION"], "you recognized wrong cards"
    # assert False
    # Function to get token type
    def get_token_type(token):
        if token in '+-*/':
            return 'operator'
        elif token == '(':
            return 'open_paren'
        elif token == ')':
            return 'close_paren'
        elif re.match(r'\d+', token):
            return 'number'
        else:
            return 'unknown'

    # Extract tokens
    tokens = re.findall(r'\d+|[+\-*/()]', current_formula)
    tokens_str = ''.join(tokens)
    stepper = StepResult(0, [])
    if tokens_str != current_formula:
        # There are illegal characters in current_formula
        stepper.add(REWARD_FN["ILLEGAL_FORMULA"], "illegal characters")
        return stepper
    prev_token_type = None
    paren_stack = []
    for token in tokens:
        token_type = get_token_type(token)
        if token_type == 'number':
            if prev_token_type in [None, 'operator', 'open_paren']:
                pass
            else:
                # invalid sequence
                stepper.add(REWARD_FN["ILLEGAL_FORMULA"], "invalid sequence after number")
        elif token_type == 'operator':
            if prev_token_type in ['number', 'close_paren']:
                pass
            else:
                # invalid sequence
                stepper.add(REWARD_FN["ILLEGAL_FORMULA"], "invalid sequence after operator")
        elif token_type == 'open_paren':
            if prev_token_type in [None, 'operator', 'open_paren']:
                paren_stack.append(token)
            else:
                # invalid sequence
                stepper.add(REWARD_FN["ILLEGAL_FORMULA"], "invalid sequence before '('")
        elif token_type == 'close_paren':
            if prev_token_type in ['number', 'close_paren']:
                if paren_stack:
                    paren_stack.pop()
                else:
                    # invalid sequence
                    stepper.add(REWARD_FN["ILLEGAL_FORMULA"], "unmatched closing parenthesis")
            else:
                # invalid sequence
                stepper.add(REWARD_FN["ILLEGAL_FORMULA"], "invalid sequence after ')'")
        else:
            stepper.add(REWARD_FN["ILLEGAL_FORMULA"], "unknown token")
        prev_token_type = token_type

    # Extract numbers from current_formula
    numbers_in_formula = re.findall(r'\d+', current_formula)
    numbers_in_formula_int = [int(n) for n in numbers_in_formula]

    # Count numbers
    card_nums_counts = Counter(card_nums)
    numbers_in_formula_counts = Counter(numbers_in_formula_int)
    # Check for invalid numbers
    invalid_numbers = [num for num in numbers_in_formula_counts if num not in card_nums_counts]
    overused_numbers = [num for num in numbers_in_formula_counts if numbers_in_formula_counts[num] > card_nums_counts[num]]
    if invalid_numbers:
        stepper.add(REWARD_FN["INCORRECT_NUMBER"], f"you used invalid numbers: {invalid_numbers}")
    elif overused_numbers:
        stepper.add(REWARD_FN["INCORRECT_NUMBER"], f"you used numbers: {overused_numbers} too many times")
    
    underused_numbers = [num for num in card_nums_counts if card_nums_counts[num] > numbers_in_formula_counts[num]]
    if underused_numbers:
        stepper.add(REWARD_FN["INCORRECT_NUMBER"], f"you didn't use numbers: {underused_numbers}")
    
    # if currently reward is 0
    if stepper.reward == 0:
        try:
            if eval(current_formula) == target_points:
                stepper.add(REWARD_FN["CORRECT_SOLUTION"], "Correct solution")
                return stepper
        except Exception as e:
            if any(sol.startswith(current_formula) for sol in solutions):
                stepper.add(REWARD_FN["PARTIAL_SOLUTION"], "partial solution")
                return stepper
    else:
        return stepper
    
    # Now we check cases with formula that use valid numbers but still no solution
    register = ""
    for token in tokens:
        # print(token)
        prev_register = register
        register += token
        # check if register is in any solution
        if any(sol.startswith(register) for sol in solutions):
            pass
        else:
            if prev_register == "":
                stepper.add(REWARD_FN["NO_SOLUTION"], "your formula does not match any solution")
            else:
                stepper.add(REWARD_FN["NO_SOLUTION"], f"{prev_register} is partially correct but the full formula does not match any solution")
            break

    return stepper


"""
This function is used to set string action space function for stepwise-p24
Not used.
"""
def set_str_action_space(env_config):
    if "gym_cards" in env_config.id and 'oneline' not in env_config.id:
        if env_config.treat_face_cards_as_10 :
            list_numbers = list(range(1, 11))
            ACTION_SPACE = ['delete'] + [str(i) for i in list_numbers]+ ['+', '-', '*', '/', '(', ')', '=']
        else:
            list_numbers = list(range(1, 14))
            ACTION_SPACE = ['delete'] + [str(i) for i in list_numbers]+ ['+', '-', '*', '/', '(', ')', '=']
    elif "gym_cards" in env_config.id and 'oneline' in env_config.id:
        # string action space not supported for this environment
        ACTION_SPACE = []
    elif "gym_virl" in env_config.id:
        if getattr(env_config, 'absolute_action', True):
            ACTION_SPACE = """"forward()": indicates moving forward one step
"turn_direction(x)": indicates adjust the ego agent direction towards x direction. x could be any following 8 directions ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
"stop()": indicates the navigation is finished."""
        else:
            ACTION_SPACE = """"forward()": indicates moving forward one step
"turn_direction(x)": indicates adjust the ego agent direction towards x direction. x could be any following ['left', 'right', 'slightly left', 'slightly right']
"stop()": indicates the navigation is finished."""

    return ACTION_SPACE




"""
Reward FN for VIRL
"""

REWARD_FN_VIRL = {
    "CORRECT_ACTION": 1,
    "INCORRECT_ACTION": -1,
    "INCORRECT_OBS": -1.5,
    "INCORRECT_INSTRUCTION": -1.75,
}

def step_rewards_virl(raw_output: str, gt_action: str, gt_obs: str, gt_instruction: str):
    stepper = StepResult(0, [])
    output_action  = re_match(raw_output, "action")
    output_obs = re_match(raw_output, "current observation")
    output_instruction = re_match(raw_output, "current instruction")

    format_token = """
[Output]
{{
  "current observation": latest observation from the streetview grid,
  "current instruction": analyze the full instruction and identify the sentence to be executed,
  "action": the action to be taken chosen from the action space,
}}"""
    if output_action == gt_action:
        stepper.add(REWARD_FN_VIRL["CORRECT_ACTION"], "Correct action")
        return stepper
    else:
        if ('intersection' in output_obs) and ('intersection' not in gt_obs):
            stepper.add(REWARD_FN_VIRL["INCORRECT_OBS"], "you are not at any intersection. Please avoid this observation and try again in the same format:" + format_token)
            return stepper
        elif ('intersection' not in output_obs) and ('intersection' in gt_obs):
            stepper.add(REWARD_FN_VIRL["INCORRECT_OBS"], "you are at an intersection. Please avoid this observation and try again in the same format:" + format_token)
            return stepper
        stepper.add(REWARD_FN_VIRL["INCORRECT_ACTION"], "you make incorrect action. Please avoid this action and try again in the same format:" + format_token)
        
    return stepper

if __name__ == "__main__":
    # test step_rewards
    card_nums = [1, 2, 3, 4]
    current_formula = "4+2+3-5"
    solutions = ["1+2+3+4", "1+3+2+4"]
    target_points = 10
    # print(step_rewards(card_nums, current_formula, solutions, target_points))
    reward, message = step_rewards(card_nums, current_formula, solutions, target_points)
    print(reward, message)
