from gym.envs.registration import register

register(
    id='gym_examples/GridWorldTm-v0',
    entry_point='gym_examples.envs:GridWorldEnvTm',
    max_episode_steps=500,
)