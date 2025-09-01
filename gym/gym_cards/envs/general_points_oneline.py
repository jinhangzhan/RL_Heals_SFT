import os
from typing import Optional

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from itertools import permutations, product, chain, zip_longest
from fractions import Fraction as F
# from data_collection.verifier import step_rewards
from utils_rl import step_rewards
from utils_general import re_match, robust_str_to_list
from PIL import Image, ImageDraw, ImageFont
from utils_rl import REWARD_FN

def get_image(card_name):
    path = f"img/{card_name}.png"
    cwd = os.path.dirname(__file__)
    image = Image.open(os.path.join(cwd, path))
    return image

# Constants for actions
NUMBER_ACTIONS_FULL = list(range(1, 14))
NUMBER_ACTIONS_TEN = list(range(1, 11))
OPERATOR_ACTIONS = ['+', '-', '*', '/', '(', ')', '=']

class GeneralPointEnv_oneline(gym.Env):
    """
    A custom Gym environment for solving the "generalized 24 Game".

    Actions:
        - When treat_face_cards_as_10=True:

            0: 'delete'
            1: 1
            2: 2
            3: 3
            4: 4
            5: 5
            6: 6
            7: 7
            8: 8
            9: 9
            10: 10
            11: '+'
            12: '-'
            13: '*'
            14: '/'
            15: '('
            16: ')'
            17: '='

        - When treat_face_cards_as_10=False:
            0: 'delete'
            1: 1
            2: 2
            3: 3
            4: 4
            5: 5
            6: 6
            7: 7
            8: 8
            9: 9
            10: 10
            11: 11
            12: 12
            13: 13
            14: '+'
            15: '-'
            16: '*'
            17: '/'
            18: '('
            19: ')'
            20: '='

    Termination:
        - If the formula length exceeds 20.
        - If '=' action is taken, the formula is evaluated.

    Reward:
        - 10 if the formula evaluates to the target_points.
        - -1 if an invalid action is taken.
        - 0 otherwise

    Initialization Options:
        - treat_face_cards_as_10: Treats face cards J, Q, K as 10 (default is True).
        - target_points: The target sum to reach (default is 24).

    """
    def __init__(self, treat_face_cards_as_10=True, target_points=24, \
            resolution=1200, show_eqn=True, verify_iter=5, ood=False, language_only=True, face_cards_color='mixed'):
        self.target_points = target_points
        self.treat_face_cards_as_10 = treat_face_cards_as_10
        self.set_action_space()
        self.canvas_width, self.canvas_height = resolution, resolution
        self.show_eqn = show_eqn
        self.verify_iter = verify_iter
        self.observation_space = spaces.Box(low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8)
        self.ood = ood
        self.language_only = language_only
        self.face_cards_color = face_cards_color
        self.reset()

    def solve(self, digits: list):
        """
            Code obtained from here: https://rosettacode.org/wiki/24_game/Solve#Python
            This function takes a list of 4 digits and returns
            True if a solution exists, False otherwise.
            If true, we also save the solution.
        """
        digilen = len(digits)
        # length of an exp without brackets
        exprlen = 2 * digilen - 1
        # permute all the digits
        # added shuffle to avoid always the same solution
        digiperm = sorted(set(permutations(digits)))
        random.shuffle(digiperm)
        # All the possible operator combinations
        opcomb = list(product('+-*/', repeat=digilen-1))
        # All the bracket insertion points:
        brackets = ([()] + [(x, y)
                            for x in range(0, exprlen, 2)
                            for y in range(x+4, exprlen+2, 2)
                            if (x, y) != (0, exprlen+1)]
                    + [(0, 3+1, 4+2, 7+3)])  # double brackets case
        self.solution = []
        for d in digiperm:
            for ops in opcomb:
                if '/' in ops:
                    d2 = [('F(%s)' % i) for i in d]  # Use Fractions for accuracy
                else:
                    d2 = d
                ex = list(chain.from_iterable(zip_longest(d2, ops, fillvalue='')))
                for b in brackets:
                    exp = ex[::]
                    for insertpoint, bracket in zip(b, '()'*(len(b)//2)):
                        exp.insert(insertpoint, bracket)
                    txt = ''.join(str(i) for i in exp)
                    try:
                        num = eval(txt)
                    except ZeroDivisionError:
                        continue
                    if num == self.target_points:
                        if '/' in ops:
                            exp = [(term if not term.startswith('F(') else term[2:-1])
                                for term in exp]
                        ans = ''.join(str(i) for i in exp).rstrip()
                        self.solution.append(ans)
        if len(self.solution) > 0:
            return True
        else:
            return False

    def set_action_space(self):
        numbers = NUMBER_ACTIONS_TEN if self.treat_face_cards_as_10 else NUMBER_ACTIONS_FULL
        self.allowed_numbers = numbers
        self.action_space = spaces.Discrete(len(numbers) + len(OPERATOR_ACTIONS) + 1) # added 1 for delete action

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.verify_info = None
        self.cards_num, self.cards, self.cards_without_suit = self._generate_cards(seed=seed)
        # Generate a new set of cards if the current set is unsolvable
        while not self.solve(self.cards_num):
            self.cards_num, self.cards, self.cards_without_suit = self._generate_cards(seed=seed)
        self.remaining_step = self.verify_iter
        self.card_imgs = []
        self.card_width = int(self.canvas_width / len(self.cards) * 0.9)  # Adjust as needed
        self.card_height = int(self.card_width * 7/5)  # Assuming a 5:7 card ratio; adjust if different
        for _, card in enumerate(self.cards):
            pil_img = get_image(card).resize((self.card_width, self.card_height))  # Resize the card
            self.card_imgs.append(pil_img)
        self.formula = []
        self.used_cards = []
        # additional feature for allowing the agent to know the remaining numbers
        self.remaining_nums = self.cards_num.copy()
        info = {"Cards": self.cards, "Plain Cards": self.cards_without_suit,
                "Numbers": self.cards_num, "Formula": self.formula,
                "Solution": self.solution, "Remaining Numbers": self.remaining_nums,
                "Remaining Step": self.remaining_step, "Verify Info": self.verify_info}
        return self._get_observation(), info

    def step(self, action):
        terminated, reward, info = False, 0, {}
        self.remaining_step -= 1
        if self.remaining_step == -1:
            return self._terminate_step(-1, 'step_limit_reached', is_truncated=True)

        current_formula = re_match(action, 'formula')
        recognized_cards = re_match(action, 'cards')
        translated_number = re_match(action, 'number')
        # transform string to list
        
        try:
            current_formula = current_formula.split('=')[0]
        except:
            pass
        
        recognized_cards = robust_str_to_list(recognized_cards)
        translated_number = robust_str_to_list(translated_number)

        reward, verify_info = step_rewards(card_nums=self.cards_num, current_formula=current_formula, solutions=self.solution, target_points=self.target_points, \
            recognized_cards=recognized_cards, translated_number=translated_number, gt_cards=self.cards_without_suit, language_only=self.language_only)
        self.verify_info = verify_info

        if reward == max(REWARD_FN.values()):
            terminated = True
        info = {"Cards": self.cards, "Plain Cards": self.cards_without_suit,
                "Numbers": self.cards_num, "Formula": self.formula,
                "Solution": self.solution, "Remaining Numbers": self.remaining_nums,
                "Remaining Step": self.remaining_step, "Verify Info": self.verify_info}
        return self._get_observation(), reward, terminated, False, info
        
        

    def _generate_cards(self, seed: Optional[int] = 0):
        # if seed is not None:
        #     random.seed(seed)
        if not self.ood:
            cards_num = [random.randint(1, 13) for _ in range(4)]
        else:
            cards_num = [random.randint(1, 13) for _ in range(3)] + [random.randint(11, 13)]
            # shuffle the cards
            random.shuffle(cards_num)
        if self.face_cards_color == 'mixed':
            suits = ["H", "S", "D", "C"]
        elif self.face_cards_color == 'red':
            suits = ["H", "D"]
        elif self.face_cards_color == 'black':
            suits = ["S", "C"]
        else:
            raise ValueError("Invalid face cards color")
        cards_suit = [random.choice(suits) for _ in range(4)]
        cards = [y + self._card_num_to_str(x) for x, y in zip(cards_num, cards_suit)]
        cards_without_suit = [self._card_num_to_str(x) for x in cards_num]
        cards_without_suit = [card.replace('T', '10') for card in cards_without_suit]
        if self.treat_face_cards_as_10:
            cards_num = [min(x, 10) for x in cards_num]
        return cards_num, cards, cards_without_suit

    def _card_num_to_str(self, num):
        face_cards = {1: 'A', 10:'T', 11: 'J', 12: 'Q', 13: 'K'}
        return face_cards.get(num, str(num))

    def _is_valid_action(self,action):
        if action not in self.allowed_numbers:
            # We don't check for operators
            return True
        else:
            new_used_cards = self.used_cards + [action]
            is_valid = not any(new_used_cards.count(x) > self.cards_num.count(x) for x in new_used_cards)
            return is_valid


    def _terminate_step(self, reward, info_key, is_truncated=False):
        info = {"Cards": self.cards, "Plain Cards": self.cards_without_suit,
                "Numbers": self.cards_num, "Formula": self.formula,
                "Solution": self.solution, "Remaining Numbers": self.remaining_nums,
                "Remaining Step": self.remaining_step, "Verify Info": "step_limit_reached"}
        return self._get_observation(), reward, not is_truncated, is_truncated, info

    def _get_observation(self):
        # Create a blank white canvas
        # Misc: the color code is dark green taken from casino tables
        canvas = Image.new('RGB', (self.canvas_width, self.canvas_height), '#35654d')

        # Paste each card onto the canvas
        for i, pil_img in enumerate(self.card_imgs):
            # Calculate position for pasting
            x_offset = 5+ int(i * pil_img.width * 1.1)  # adjust this multiplier (1.1) for spacing
            y_offset = int((self.canvas_height - pil_img.height) / 2)  # center vertically
            canvas.paste(pil_img, (x_offset, y_offset))

        # Draw formula onto the canvas
        draw = ImageDraw.Draw(canvas)
        if self.show_eqn:
            text_formula = 'Formula:'
            text = f'{" ".join(map(str, self.formula))}'
            font = ImageFont.truetype('dejavu/DejaVuSans.ttf', (self.canvas_width / 300) * 16)
            draw.text((10, self.canvas_height*0.70), text_formula, fill="white", font=font)  # adjust position and other properties as needed
            draw.text((10, self.canvas_height*0.80), text, fill="white", font=font)  # adjust position and other properties as needed
        # Convert PIL image to numpy array if required
        image_array = np.array(canvas)

        return image_array
