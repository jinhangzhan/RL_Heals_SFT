"""
Finalized prompts
Updated: Jan-28-2025
By Tianzhe
"""


Q_GeneralPoint_EQN_VL = """
[Task Description]
You are an expert {target_number} points card game player. You are observing these four cards in the image.
Note that {face_card_msg}, and each card must be used once.
Your goal is to output a formula that evaluates to {target_number} using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

[Output]
Your response should be a valid json file in the following format:
{{
  "cards": [x, y, z, w], where {face_card_msg},
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "formula": 'an equation that equals {target_number}',
}}

"""

Q_GeneralPoint_EQN_L = """
[Task Description]
You are an expert {target_number} points card game player. You will receive a set of 4 cards.
Note that {face_card_msg}, and each card must be used once.
Your goal is to output a formula that evaluates to {target_number} using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

[Input]
Cards: {cards}

[Output]
{{
  "cards": [x, y, z, w], where {face_card_msg},
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "formula": 'an equation that equals {target_number}',
}}

"""

"""
    *** Responses templates ***
"""

ResponseEqn = """
{{
  "cards": {cards},
  "number": {numbers},
  "formula": "{formula}",
}}"""



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

ResponseVIRL = """
{{
  "current observation": "{current_observation}",
  "current instruction": "{current_instruction}",
  "action": "{action}",
}}
"""

Q_VIRL_VL = """
[Task Description]
You are an expert in navigation. You will receive a sequence of instructions to follow while observing your surrounding stree tviews. You
are also provided with your observation and action history in text. Your goal is to first analyze the instruction and identify the next sentence to be executed. 
Then, you need to provide the action to be taken based on the current observation and instruction.

[Instruction]
{instruction}

[Observation format]
You observe a 2x2 grid of streetview images with the following headings:
[front, right
 back, left]
You need to identify if any of the landmarks in the instruction are visible in the street view grid.

[Action space]
{action_space}

[Observations and actions sequence]
{obs_act_seq}

[Output]
{{
  "current observation": latest observation from the street view grid,
  "current instruction": analyze the full instruction and identify the sentence to be executed,
  "action": the action to be taken chosen from the action space,
}}
"""