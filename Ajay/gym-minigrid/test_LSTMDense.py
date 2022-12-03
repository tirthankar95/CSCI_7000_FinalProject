import gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR
import gym_minigrid
import matplotlib.pyplot as plt
"""
Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1


"""
If the observations are images we use CNNs.
"""
class QNetworkCNN(nn.Module):
    def __init__(self, action_dim,batch_size):
        super(QNetworkCNN, self).__init__()

        #self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        #self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        #self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv_3 = nn.Conv2d(64, 16, kernel_size=3,padding='same')
        self.fc_1 = nn.Linear(784, 32)
        self.fc_2 = nn.Linear(32, action_dim)
        self.batch_size = batch_size

    def forward(self, inp):
        if inp.shape == (3,7,7):
            inp = inp.view((1,3,7,7))
        else:
            inp = inp.view((self.batch_size, 3, 7, 7))
        x1 = F.relu(self.conv_1(inp))
        #print(x1.shape)
        x1 = F.relu(self.conv_2(x1))
        #print(x1.shape)
        x1 = F.relu(self.conv_3(x1))
        #print(x1.shape)
        x1 = torch.flatten(x1, 1)
        #print(x1.shape)
        x1 = F.leaky_relu(self.fc_1(x1))
        x1 = self.fc_2(x1)

        return x1



class QNetworkLSTM(nn.Module):
    def __init__(self, action_dim,state_dim, hidden_dim, batch_size):
        super(QNetworkLSTM, self).__init__()

        #self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        #self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        #self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.conv_1 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        #self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        #self.conv_3 = nn.Conv2d(64, 16, kernel_size=3,padding='same')
        self.input_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.fc_1 = nn.Linear(self.hidden_dim, 32)
        self.fc_2 = nn.Linear(32, action_dim)
        self.batch_size = batch_size

    def forward(self, inp):
        #print(inp.shape)
        if inp.shape == (4,3,7,7):
            inp = np.reshape(inp,(4,147))
            #print(inp.shape)
            inp = torch.Tensor(inp).to(device)
            inp = inp.view((1,4,147))
            #print(inp.shape)
        else:
            inp = torch.reshape(inp, (batch_size,4,147))
            inp = inp.view((self.batch_size, 4,147))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(inp, [self.batch_size], batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        #output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)

        #x1, _ = self.lstm(inp)
        x1 = hidden[-1]
        #print("LSTM",hidden[-1].shape)

        #x1 = F.relu(self.conv_2(x1))
        #print(x1.shape)
        #x1 = F.relu(self.conv_3(x1))
        #print(x1.shape)
        #x1 = torch.flatten(x1, 1)
        x1 = self.fc_1(x1[-1])
        #print(x1.shape)
        #x1 = F.leaky_relu(self.fc_1(x1))
        #print(x1.shape)
        x1 = self.fc_2(x1)
        #x1 = torch.nn.functional.log_softmax(x1, dim=1)
        return x1



"""
memory to save the state, action, reward sequence from the current episode. 
"""
class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(3, n-1), batch_size)
        seq_states = np.zeros((batch_size,4,3,7,7))
        seq_next_states = np.zeros((batch_size,4,3,7,7))
        for i in idx:
            seq_states[i,0,:,:,:] =self.state[i-3]
            seq_states[i, 1, :, :, :] = self.state[i - 2]
            seq_states[i, 2, :, :, :] = self.state[i - 1]
            seq_states[i, 3, :, :, :] = self.state[i]

            seq_next_states[i,0,:,:,:] =self.state[i-2]
            seq_next_states[i, 1, :, :, :] = self.state[i - 1]
            seq_next_states[i, 2, :, :, :] = self.state[i]
            seq_next_states[i, 3, :, :, :] = self.state[i+1]
        #seq_states = [self.state[i] for i in range(idx-3,idx+1)]
        #seq_next_states = [self.state[i] for i in range(idx-2,idx+2)]
        return torch.Tensor(seq_states).to(device), torch.LongTensor(self.action)[idx].to(device), \
               torch.Tensor(seq_next_states).to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)


    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


def select_action(model, env, state, eps):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(values.cpu().numpy())

    return action


def train(batch_size, current, target, optim, memory, gamma):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)
    #print("Train shape",states.shape)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()


    optim.zero_grad()
    loss.backward()
    optim.step()

def train_lstm(batch_size, current, target, optim, memory, gamma):
    seq_len = 4
    states, actions, next_states, rewards, is_done = memory.sample(batch_size)
    #print("Train shape",states.shape)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()


    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()

    perform = 0
    for _ in range(repeats):
        state = env.reset_q()
        seq_states = np.zeros((4, 3, 7, 7))
        seq_states[0, :, :, :] = state
        seq_states[1, :, :, :] = state
        seq_states[2, :, :, :] = state
        seq_states[3, :, :, :] = state
        #state = state['image']
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            #state = state.permute(2,0,1)
            with torch.no_grad():
                values = Qmodel(seq_states)
            #print("Values",values)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step_q(action)
            perform += reward
            seq_states[0, :, :, :] = seq_states[1, :, :, :]
            seq_states[1, :, :, :] = seq_states[2, :, :, :]
            seq_states[2, :, :, :] = seq_states[3, :, :, :]
            seq_states[3, :, :, :] = state

    #Qmodel.train()
    return perform/repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def main(gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.995, eps_min=0.01, update_step=10, batch_size=64, update_repeats=50,
         num_episodes=3000, seed=42, max_memory_size=50000, lr_gamma=0.9, lr_step=100, measure_step=100,
         measure_repeats=100, hidden_dim=64, env_name='CartPole-v1', cnn=False, horizon=np.inf, render=True, render_step=50):
    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """
    env_name = 'MiniGrid-DoorKey-16x16-v0'
    #env_name = 'MiniGrid-LavaCrossingS11N5-v0'
    env = gym.make(env_name)
    torch.manual_seed(seed)
    env.seed(seed)
    render = False
    cnn = True
    min_episodes = 4
    batch_size = 4
    num_episodes = 100
    measure_step = 2
    update_step = 2
    measure_repeats = 5
    eps_decay = 0.99
    #max_memory_size
    """
    if cnn:
        Q_1 = QNetworkCNN(action_dim=env.action_space.n, batch_size = batch_size).to(device)
        Q_2 = QNetworkCNN(action_dim=env.action_space.n, batch_size = batch_size).to(device)
    else:
        Q_1 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                                        hidden_dim=hidden_dim).to(device)
        Q_2 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                                        hidden_dim=hidden_dim).to(device)
    """
    Q_1 = QNetworkLSTM(action_dim=env.action_space.n, state_dim=147, hidden_dim=16, batch_size=batch_size)
    # transfer parameters from Q_1 to Q_2
    #update_parameters(Q_1, Q_2)

    # we only train Q_1
    #for param in Q_2.parameters():
    #    param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []
    eps_logs=[]
    steps = []
    for episode in range(num_episodes):
        eps_logs.append(eps)
        print("Episode: ", episode)
        # display the performance
        if episode % measure_step == 0:
            performance.append([episode, evaluate(Q_1, env, measure_repeats)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("lr: ", scheduler.get_lr()[0])
            print("eps: ", eps)

        state = env.reset_q()
        #state = state['image']
        for i in range(4):
            memory.state.append(state)

        seq_states = np.zeros((4,3,7,7))
        seq_states[0,:,:,:]=state
        seq_states[1, :, :, :] = state
        seq_states[2, :, :, :] = state
        seq_states[3, :, :, :] = state
        done = False
        i = 0
        while not done:
            i += 1
            action = select_action(Q_1, env, seq_states, eps)
            state, reward, done, loc = env.step_q(action)
            #print("Check state:",i, episode, state.shape, done, loc)

            if i > horizon:
                done = True

            # render the environment if render == True
            if render and episode % render_step == 0:
                env.render()

            # save state, action, reward sequence
            memory.update(state, action, reward, done)
            seq_states[0, :, :, :] = seq_states[1, :, :, :]
            seq_states[1, :, :, :] = seq_states[2, :, :, :]
            seq_states[2, :, :, :] = seq_states[3, :, :, :]
            seq_states[3, :, :, :] = state

        steps.append(i)
        print("steps: ",i)
        if episode >= min_episodes and episode % update_step == 0:
            print("Training: ", episode)
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_1, optimizer, memory, gamma)

            eps = max(eps * eps_decay, eps_min)
            # transfer new parameter from Q_1 to Q_2
            #update_parameters(Q_1, Q_2)

        # update learning rate and eps
        scheduler.step()
        #eps = max(eps*eps_decay, eps_min)
    performance.append([episode, evaluate(Q_1, env, measure_repeats)])
    print(performance)
    plt.plot([rewards[1] for rewards in performance])
    plt.xlabel('epochs')
    plt.ylabel('rewards')
    #plt.title('Double DQN Per')
    plt.savefig(env_name+"_rewards_LSTMbasic.png")
    plt.close()
    plt.plot(eps_logs)
    plt.xlabel('epochs')
    plt.ylabel('epsilon')
    plt.savefig(env_name+"_epsilon_LSTMbasic.png")
    plt.close()
    plt.plot(steps)
    plt.xlabel('episodes')
    plt.ylabel('steps')
    plt.title('Steps required for termination')
    plt.savefig(env_name+"_steps_LSTMbasic.png")
    plt.close()
    print("Performance:",performance)
    print("Epsilon:", eps_logs)
    print("Steps: ", steps)
    MODEL_PATH = 'LSTMbasic_Q1_'+str(env_name)+'_'+str(num_episodes)+'.pt'
    torch.save(Q_1, MODEL_PATH)
    #MODEL_PATH = 'DoubleDQN_Q2_' + str(env_name) + '_' + str(num_episodes) + '.pt'
    #torch.save(Q_2, MODEL_PATH)
    return Q_1, performance


if __name__ == '__main__':
    main()