from gym_det.envs.determinant_env import DeterminantEnv
from gymnasium.envs.registration import register

register(
    id='gym_det/Determinant-v0',
    entry_point='gym_det.envs:DeterminantEnv',
    max_episode_steps=5,
)

