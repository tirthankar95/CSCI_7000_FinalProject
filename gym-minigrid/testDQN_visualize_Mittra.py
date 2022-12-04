import gym
import keras
from keras.models import load_model
from gym_minigrid.envs.doorkey import *
import time


#env = gym.make('MiniGrid-DoorKey-6x6-v0')
env = DoorKeyEnv(size=6)
seed = 42
env.seed(seed)
actions=env.action_space.n
done = False
i=0
noOfFrames=3

model=keras.models.load_model('Expert2')
print(model.summary())
#model.summary()
state = np.array(env.reset_m())
env.render()
state, reward, done, _ = env.step_m(1)
env.render()
state, reward, done, _ = env.step_m(2)
env.render()
state, reward, done, _ = env.step_m(2)
env.render()
# state, reward, done, _ = env.step_m(3)
# env.render()
# state, reward, done, _ = env.step_m(0)
# env.render()
# state, reward, done, _ = env.step_m(0)
# env.render()
# state, reward, done, _ = env.step_m(2)
# env.render()
# state, reward, done, _ = env.step_m(1)
# env.render()
# state, reward, done, _ = env.step_m(4)
# env.render()
# state, reward, done, _ = env.step_m(2)
# env.render()
# state, reward, done, _ = env.step_m(2)
# env.render()
# state, reward, done, _ = env.step_m(2)
# env.render()
# state, reward, done, _ = env.step_m(1)
# env.render()
# state, reward, done, _ = env.step_m(2)
# env.render()
state = np.array(state)
env.render()
time.sleep(1)
temp_state=dict(state.item(0))
state_history=[temp_state['image'] for i in range(noOfFrames)]

while not done:
  state_numpy = np.array(state_history[-3:]).reshape(3,147)
  state_numpy = np.array([state_numpy])
  action_probs = model(state_numpy, training=False)
  action = np.argmax(action_probs[0])
  state, reward, done, _ = env.step_m(action)
  state=np.array(state)
  temp_state=dict(state.item(0))
  del state_history[:1]
  state_history.append(temp_state['image'])
  i = i+1
  print(action)
  env.render()
  time.sleep(0.1)
