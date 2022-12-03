from gym_minigrid.envs.doorkey import *
env = DoorKeyEnv(size=6)
env.reset_q()
env.action_space.n

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 4098  # Size of batch taken from replay buffer
max_steps_per_episode = 5000


num_actions = env.action_space.n
print(num_actions)
def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(7, 7, 3,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(16, 3,  activation="relu", padding='same')(inputs)
    layer2 = layers.Conv2D(32, 3,  activation="relu", padding='same')(layer1)
    layer3 = layers.Conv2D(32, 3,  activation="relu", padding='same')(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(32, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()
model.load_weights("DoorKey_ckpt")


import matplotlib.pyplot as plt
import collections

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
max_memory_length = 100000

# Experience replay buffers
action_history = collections.deque(maxlen=max_memory_length)
state_history = collections.deque(maxlen=max_memory_length)
state_next_history = collections.deque(maxlen=max_memory_length)
rewards_history = collections.deque(maxlen=max_memory_length)
done_history = collections.deque(maxlen=max_memory_length)
episode_reward_history = collections.deque(maxlen=max_memory_length)
reward_plot = collections.deque(maxlen=max_memory_length)
timesteps = collections.deque(maxlen=max_memory_length)
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
# Train the model after 4 actions
update_after_actions = 100
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

#iter=100
once=True

from tqdm import tqdm
iter=500

model.summary()

#model = tf.saved_model.load("DoorKey")
state = np.array(env.reset_q())

state = np.moveaxis(state,0, -1)
#state = np.moveaxis(state,0, -1)
print(state.shape)

state_tensor = tf.convert_to_tensor(state)
state_tensor = tf.expand_dims(state_tensor, 0)
print(state_tensor.shape)
action_probs = model(state_tensor, training=False)
# Take best action
action = tf.argmax(action_probs[0]).numpy()
print(action)

import time
done = False
i=0
while not done:
  state_tensor = tf.convert_to_tensor(state)
  state_tensor = tf.expand_dims(state_tensor, 0)
  action_probs = model(state_tensor, training=False)
  action = tf.argmax(action_probs[0]).numpy()
  i = i+1
  state, reward, done, _ = env.step_q(action)
  print(action,i)
  env.render()
  time.sleep(0.1)
