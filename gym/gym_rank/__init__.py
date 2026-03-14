from gymnasium.envs.registration import register

from gym_rank.envs.rank_env import RankEnv

register(
    id='gym_rank/Rank-v0',
    entry_point='gym_rank.envs:RankEnv',
)

