
import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid
def main():
    env = gym.make('MiniGrid-DoorKey-5x5-v0')
    env.reset()
    done = False
    i = 0
    #env.render('human')
    while(not done):
        action = env.action_space.sample()
        env.render('human')
        obs, reward, done, info = env.step_q(action)
        print(done, info, reward, env.step_count)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
