from gym_virl.envs.navigation_environment import NavigationEnvironment
from gymnasium.envs.registration import register

register(
    id='gym_virl/Navigation-v0',
    entry_point='gym_virl.envs:NavigationEnvironment',
    max_episode_steps=300,
)