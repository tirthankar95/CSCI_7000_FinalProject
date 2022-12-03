import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from A2C_models import ActorCriticContinuous, ActorCriticDiscrete
from A2C_memory import Memory
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import StepLR


"""
Implementation of Advantage-Actor-Critic for gym environments
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Actor Critic Model
"""

class ActorCriticContinuous(nn.Module):
    """
    Actor-Critic for continuous action spaces. The network returns a state_value (critic) and
    action mean and action standarddeviation (actor). The action is the sampled from a normal
    distribution with mean and std given by the actor.
    """
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCriticContinuous, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, int(hidden_dim/2))

        # critic head
        self.critic_head = nn.Linear(int(hidden_dim/2), 1)

        # actor head
        self.actor_head_mean = nn.Linear(int(hidden_dim/2), action_dim)
        self.actor_head_sigma = nn.Linear(int(hidden_dim / 2), action_dim)

    def forward(self, inp):
        x = F.leaky_relu(self.fc_1(inp))
        x = F.leaky_relu(self.fc_2(x))

        # how good is the current state?
        state_value = self.critic_head(x)

        action_mean = (self.actor_head_mean(x))
        action_sigma = F.softplus(self.actor_head_sigma(x) + 0.0001)

        return action_mean, action_sigma, state_value


class ActorCriticDiscrete(nn.Module):
    """
    Actor-Critic for discrete action spaces. The network returns a state_value (critic)and action probabilities (actor).
    """
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCriticDiscrete, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, int(hidden_dim/2))

        # critic head
        self.critic_head = nn.Linear(int(hidden_dim/2), 1)

        # actor head
        self.actor_head = nn.Linear(int(hidden_dim/2), action_dim)

    def forward(self, inp):
        x = F.leaky_relu(self.fc_1(inp))
        x = F.leaky_relu(self.fc_2(x))

        # how good is the current state?
        state_value = self.critic_head(x)

        # actor's probability to take each action
        action_prob = F.softmax(self.actor_head(x), dim=-1)

        return action_prob, state_value


"""
Memory
"""
class Memory:
    def __init__(self):
        self.rewards = []
        self.action_prob = []
        self.state_values = []
        self.entropy = []

    def calculate_data(self, gamma):
        # compute the discounted rewards
        disc_rewards = []
        R = 0
        for reward in self.rewards[::-1]:
            R = reward + gamma*R
            disc_rewards.insert(0, R)

        # transform to tensor and normalize
        disc_rewards = torch.Tensor(disc_rewards)
        disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + 0.001)

        return torch.stack(self.action_prob), torch.stack(self.state_values), \
               disc_rewards.to(device), torch.stack(self.entropy)

    def update(self, reward, entropy, log_prob, state_value):
        self.entropy.append(entropy)
        self.action_prob.append(log_prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)

    def reset(self):
        del self.rewards[:]
        del self.action_prob[:]
        del self.state_values[:]
        del self.entropy[:]



def select_action(model, state, mode):
    state = torch.Tensor(state).to(device)
    if mode == "continuous":
        mean, sigma, state_value = model(state)
        s = torch.distributions.MultivariateNormal(mean, torch.diag(sigma))
    else:
        probs, state_value = model(state)
        s = Categorical(probs)

    action = s.sample()
    entropy = s.entropy()

    return action.numpy(), entropy, s.log_prob(action), state_value


def evaluate(actor_critic, env, repeats, mode):
    actor_critic.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                if mode == "continuous":
                    mean, sigma, _ = actor_critic(state)
                    m = torch.distributions.Normal(mean, sigma)
                else:
                    probs, _ = actor_critic(state)
                    m = Categorical(probs)

            action = m.sample()
            state, reward, done, _ = env.step(action.numpy())
            perform += reward
    actor_critic.train()
    return perform/repeats


def train(memory, optimizer, gamma, eps):
    action_prob, values, disc_rewards, entropy = memory.calculate_data(gamma)

    advantage = disc_rewards.detach() - values

    policy_loss = (-action_prob*advantage.detach()).mean()
    value_loss = 0.5 * advantage.pow(2).mean()
    loss = policy_loss + value_loss - eps*entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main(gamma=0.99, lr=5e-3, num_episodes=400, eps=0.001, seed=42, lr_step=100, lr_gamma=0.9, measure_step=100,
         measure_repeats=100, horizon=np.inf, hidden_dim=64, env_name='CartPole-v1', render=True):
    """
    :param gamma: reward discount factor
    :param lr: initial learning rate
    :param num_episodes: total number of episodes performed in the environment
    :param eps: entropy regularization parameter (increases exploration)
    :param seed: random seed
    :param lr_step: every "lr_step" many episodes the lr is updated by the factor "lr_gamma"
    :param lr_gamma: see above
    :param measure_step: every "measure_step" many episodes the the performance is measured using "measure_repeats" many
    episodes
    :param measure_repeats: see above
    :param horizon: if not set to infinity limits the length of the episodes when training
    :param hidden_dim: hidden dimension used for the DNN
    :param env_name: name of the gym environment
    :param render: if True the environment is rendered twice every "measure_step" many episodes
    """
    batch_size=4
    env = gym.make(env_name)
    torch.manual_seed(seed)
    env.seed(seed)

    # check whether the environment has a continuous or discrete action space.
    if type(env.action_space) == gym.spaces.Discrete:
        action_mode = "discrete"
    elif type(env.action_space) == gym.spaces.Box:
        action_mode = "continuous"
    else:
        raise Exception("action space is not known")

    # Get number of actions for the discrete case and action dimension for the continuous case.
    if action_mode == "continuous":
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n


    if action_mode == "continuous":
        state_dim = env.observation_space.shape[0]
        actor_critic = ActorCriticContinuous(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    else:
        actor_critic = ActorCriticDiscrete(action_dim=action_dim, batch_size=4).to(device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    performance = []
    for episode in range(num_episodes):
        # reset memory
        memory = Memory()
        # display the episode_performance
        if episode % measure_step == 0:
            performance.append([episode, evaluate(actor_critic, env, measure_repeats, action_mode)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("lr: ", scheduler.get_lr())

        state = env.reset()

        done = False
        count = 0
        while not done and count < horizon:
            count += 1
            action, entropy, log_prob, state_value = select_action(actor_critic, state, action_mode)
            state, reward, done, _ = env.step(action)
            if render and episode % int((measure_step/2)) == 0:
                env.render()

            # save the information
            memory.update(reward, entropy, log_prob, state_value)

        # train on the observed data
        train(memory, optimizer, gamma, eps)
        # update the learning rate
        scheduler.step()

    return actor_critic, performance


if __name__ == '__main__':
    main()