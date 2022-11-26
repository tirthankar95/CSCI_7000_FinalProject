import gym
import gym_examples

env=gym.make('gym_examples/GridWorldTm-v0',render_mode='human',size=4,g=50,t=-50,w=-1,p=0.7)
env.action_space.seed(2000)
env.reset(seed=2022)
windy=0
for _ in range(100):
    # Random action is sampled.
    action_i=env.action_space.sample()
    obs,reward,terminated,truncated,info=env.step(action_i)
    print('NextState {}; Reward {}; WindDivert {}'.format(obs['agent'],reward,obs['diverted']))
    if obs['diverted']:
        windy+=1
    if terminated:
        print('Terminal State Reached.')
        env.reset(seed=2022)
env.close()
print("{}".format(windy/100))
