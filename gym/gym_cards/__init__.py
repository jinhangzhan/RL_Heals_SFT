from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from gymnasium.envs.registration import register

register(
    id='gym_cards/GeneralPoint-oneline-v0',
    entry_point='gym_cards.envs:GeneralPointEnv_oneline',
    max_episode_steps=300,
)