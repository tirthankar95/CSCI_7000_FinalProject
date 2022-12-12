import gym
import keras
from keras.models import load_model
from gym_minigrid.envs.doorkey import *
from gym_minigrid.envs.crossing import *
from gym_minigrid.envs.mixed import *
import time


# env = DoorKeyEnv(size=8)
# env = CrossingEnv(size=8)
# env = MixedEnv(size=8)
seed = 42
env.seed(seed)
actions=env.action_space.n


noOfFrames=8
np.random.seed(seed=seed)

ACTIONS_TO_TXT=["\nRotated Left", "\nRotated Right", "\nMoved Forward", "\nPicked the object", "\nToggled the object"]

def best_action(action_probs):
  action = np.argmax(action_probs)
  action_probs = np.abs(action_probs - action_probs[action])
  possible_actions = []
  for i in range(5):
    if (action_probs[i] < 0.2):
      possible_actions.append(i)
  # print(action_probs, possible_actions)
  length = len(possible_actions)
  action_probs = [1 / length for i in possible_actions]
  return np.random.choice(possible_actions, p=action_probs)

# model=keras.models.load_model('SavedModels/DoorKey')
# model=keras.models.load_model('SavedModels/Lava')
# model=keras.models.load_model('SavedModels/MixedTransferLearning')

mission = env.mission
state = np.array(env.reset_m())
total_rewards = 0
i = 0
epsilon = 0.2

state = np.array(state)
env.render()
#time.sleep(10)
temp_state=dict(state.item(0))
state_history=[temp_state['image'] for i in range(noOfFrames)]
done = False
random_actions = 0
while not done:
  if epsilon > np.random.rand(1)[0]:
    action = np.random.choice(5)
    random_actions += 1
  else:
    state_numpy = np.array(state_history[-noOfFrames:]).reshape(noOfFrames,147)
    state_numpy = np.array([state_numpy])
    action_probs = model(state_numpy, training=False)
    action = best_action(action_probs[0])
  state, reward, done, _ = env.step_m(action)
  state=np.array(state)
  temp_state=dict(state.item(0))
  del state_history[:1]
  state_history.append(temp_state['image'])
  i = i+1
  if(i>5000 or reward <-100):
    failures += 1
    break
  total_rewards += reward
  performance_str = "\nAction reward : " + str(reward) + ", Cumulative Reward: " + str(total_rewards) + ", Steps completed: "+ str(i)
  env.mission = mission + ACTIONS_TO_TXT[action] + performance_str

  env.render()
  print(ACTIONS_TO_TXT[action] + performance_str)

  time.sleep(0.5)

