import gym
import numpy as np
import random
import gym_minigrid
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import LSTM,Bidirectional,Dense,Input,Embedding,TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gym_minigrid.envs.doorkey import *
from gym_minigrid.envs.crossing import *
from tqdm import tqdm



num_actions=5

def plotProgress(reward_plot):
    plt.plot(reward_plot)
    plt.xlabel('Episodes')
    plt.ylabel('Avg. reward')
    plt.title('Avg Reward Per Step V/S Episodes.')
    plt.show()

def create():
    global num_actions
    input=Input(shape=(3,147)) # (3,3,7,7) ~ the mini-grid by default returns (3,7,7) image.
    model=LSTM(units=128,return_sequences=False)(input)
    x1 = Dense(units=128, activation='relu')(model)
    x1 = Dense(units=64, activation='relu')(x1)
    x1 = Dense(units=32, activation='relu')(x1)
    x1 = Dense(units=16, activation='relu')(x1)
    output=Dense(units=num_actions,activation='linear')(x1)
    model=Model(input,output)
    return model


def main():
    global num_actions
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

    env = DoorKeyEnv(size=6)
    env.seed(seed)

    model=create()
    model_target=create()
    loss_function = keras.losses.MeanSquaredError()
    optimizer=keras.optimizers.RMSprop()

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    episode_reward_history=[]
    done_history = []
    reward_plot = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # Number of frames for exploration
    epsilon_greedy_frames = 1000000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 10
    # How often to update the target network
    update_target_network = 10000
    # Using huber loss for stability
    # We are taking 3 frames in our LSTM
    frame_offset=2

    noOfEpisodes=1000 #beast 100000
    for _ in tqdm(range(noOfEpisodes)):  # Run until solved
        noOfEpisodes-=1
        state = np.array(env.reset_m())
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_numpy = np.array(state_history[-3:]).reshape(3,147)
                state_numpy = np.array([state_numpy])
                action_probs = model(state_numpy, training=False)
                # Take best action
                action = np.argmax(action_probs[0])

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step_m(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            temp_state=dict(state.item(0)) # state is a 0-d numpy array.
            state_history.append(temp_state['image'])
            temp_state=dict(state_next.item(0)) # state is a 0-d numpy array.
            state_next_history.append(temp_state['image'])
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)-frame_offset), size=batch_size)+frame_offset
                i=indices[0]
                # Using list comprehension to sample from replay buffer
                state_sample = np.array([ np.array(state_history[i-frame_offset:i+1]).reshape(3,147) for i in indices])
                state_next_sample = np.array([ np.array(state_next_history[i-frame_offset:i+1]).reshape(3,147) for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample,verbose=False)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * np.max(\
                    future_rewards, axis=1)
                updated_q_values = updated_q_values.astype('float32')
                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)
                with tf.GradientTape() as tape:    
                    q_values = model(state_sample)
                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = loss_function(updated_q_values,q_action)
                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                

            if frame_count % update_target_network == 0:
                model.save('Expert.ml')
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]
            if done:
                break

    # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        reward_plot.append(episode_reward/timestep)
        episode_count += 1
    plotProgress(reward_plot)
    return model

if __name__ == '__main__':
    model=main()
    model.save('Expert.ml')