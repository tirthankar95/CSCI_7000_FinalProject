
import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid
from gym_minigrid.envs.mixed import *
from gym_minigrid.envs.wind import *
def main():
    #env = SimpleMixedEnv()
    env = MixedEnvS21N5Env()
    #env = WindyEnv()
    #env = gym.make('MiniGrid-DoorKey-6x6-v0')
    #env = gym.make('MiniGrid-LavaCrossingS9N1-v0')
    """
    env.reset()
    done = False
    i = 0
    #env.render('human')
    while(not done):
        action = env.action_space.sample()
        env.render('human')
        obs, reward, done, info = env.step_q(action)
        print(done, info, reward, env.step_count)
        time.sleep(1)
    """
    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step_q(action)

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break


if __name__ == "__main__":
    main()
