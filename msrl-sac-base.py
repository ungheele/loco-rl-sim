import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.utils.dataset import compute_J
from loco_mujoco import LocoEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import datetime


class CustomGymWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._max_episode_steps = env.spec.max_episode_steps
        self.gamma = 0.99
        self.horizon = self._max_episode_steps
    
  
        self._convert_action = lambda a: a

    def reset(self, state=None):
        if state is None:
            state, info = self.env.reset()
            return np.atleast_1d(state)
        else:
            _, info = self.env.reset()
            self.env.state = state

            return np.atleast_1d(state)

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, absorbing, _, info = self.env.step(action) #truncated flag is ignored 

        return np.atleast_1d(obs), reward, absorbing, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    @property
    def info(self):
        return MDPInfo(self.observation_space, self.action_space, self.gamma, self.horizon)

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, n_epochs, n_steps, n_steps_test, save, load):
    np.random.seed()

    today = datetime.date.today().strftime("%Y-%m-%d")
    logs_dir = f'./logs/{today}/'
    logger = Logger(alg.__name__, results_dir=logs_dir if save else None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)
    sw = SummaryWriter(log_dir=logs_dir)

    # MDP
    horizon = 200
    gamma = 0.99
    # mdp = InvertedPendulum()
    mdp = LocoEnv.make("HumanoidTorque.walk")
 

    # Settings
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4

    if load:
        agent = SAC.load('logs/SAC/agent-best.msh')
    else:
        # Approximator
        actor_input_shape = mdp.info.observation_space.shape
        actor_mu_params = dict(network=ActorNetwork,
                               n_features=n_features,
                               input_shape=actor_input_shape,
                               output_shape=mdp.info.action_space.shape)
        actor_sigma_params = dict(network=ActorNetwork,
                                  n_features=n_features,
                                  input_shape=actor_input_shape,
                                  output_shape=mdp.info.action_space.shape)

        actor_optimizer = {'class': optim.Adam,
                           'params': {'lr': 3e-4}}

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
        critic_params = dict(network=CriticNetwork,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': 3e-4}},
                             loss=F.mse_loss,
                             n_features=n_features,
                             input_shape=critic_input_shape,
                             output_shape=(1,))

        # Agent
        agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                    actor_optimizer, critic_params, batch_size, initial_replay_size,
                    max_replay_size, warmup_transitions, tau, lr_alpha,
                    critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = np.mean(compute_J(dataset, gamma=gamma))
    R = np.mean(compute_J(dataset))
    # E = agent.policy.entropy(dataset.state)

    logger.epoch_info(0, J=J, R=R)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)

        if n % 10 == 0: 
            dataset = core.evaluate(n_steps=n_steps_test, render=False)

            J = np.mean(compute_J(dataset, gamma=gamma))
            R = np.mean(compute_J(dataset))
            sw.add_scalar("Eval_R", R, n)
            sw.add_scalar("Eval_J", J, n)
            # E = agent.policy.entropy(dataset.state)

            logger.epoch_info(n+1, J=J, R=R)

            if save:
                logger.log_best_agent(agent, J)

    logger.info('Press a button to visualize pendulum')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    save = True
    load = False
    experiment(alg=SAC, n_epochs=5000, n_steps=1000, n_steps_test=2000, save=save, load=load)