import gymnasium as gym

import numpy as np
from imitation.data.types import Trajectory
from imitation.algorithms import bc
from imitation.util.util import save_policy
import loco_mujoco
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies import base as policy_base
import torch as th
import os
from imitation.util import util
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure

# Check if CUDA (GPU) is available
# if th.cuda.is_available():
#     device = th.device("cuda")
#     print("CUDA is available")
# else:
device = th.device("cpu")
    # print("CUDA is not available, using CPU")


# Create environment and generate dataset
env = gym.make("LocoMujoco", env_name="HumanoidTorque.walk.perfect", render_mode="human")
dataset = env.create_dataset()

# Extract states and actions from the dataset
# observations = th.tensor(dataset['states'], dtype=th.float32).to(device)
# actions = th.tensor(dataset['actions'], dtype=th.float32).to(device)

observations = dataset['states']
actions = dataset['actions']
last = dataset['last']

# observations = th.tensor(observations, dtype=th.float32).to(device)
# actions = th.tensor(actions, dtype=th.float32).to(device)
# # Create terminal flags (all False except the last one)
# terminals = np.zeros(len(observations), dtype=bool)
# terminals[-1] = True

# Assuming each trajectory ends when the dataset ends
# trajectories = [Trajectory(obs=observations, acts=actions, terminal=terminals, infos=None)]
# Create trajectories based on 'last' indicating the end of a sequence
trajectories = []
start_idx = 0

# for end_idx in range(len(last)):
#     if last[end_idx]:
#         traj_obs = observations[start_idx:end_idx + 1]
#         traj_acts = actions[start_idx:end_idx]
#         trajectories.append(Trajectory(obs=traj_obs, acts=traj_acts, terminal=True, infos = None))
#         start_idx = end_idx + 1
trajectories = [Trajectory(obs=observations, acts=actions[:-1], terminal=False, infos=None)]
log_dir = "logs_sb3/sac_bc_20240726/"
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.set_logger(new_logger)

# Create a BC trainer
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=trajectories,
    policy=model.policy,
    # policy_base.FeedForward32Policy(
    #             observation_space=env.observation_space,
    #             action_space=env.action_space,
    #             # Set lr_schedule to max value to force error if policy.optimizer
    #             # is used by mistake (should use self.optimizer instead).
    #             lr_schedule=lambda _: th.finfo(th.float32).max,
    #         ),  # Replace with the appropriate policy class
    rng=np.random.default_rng(),
    device = device,
    custom_logger=new_logger,
)

# Train the BC model
bc_trainer.train(n_epochs=1000)

reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)


# # Save the agent
model.save(log_dir + "sac_humanoid_bc")


vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")



# save_dir = "trained_models"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "bc_trained_model")
# # bc_trainer.save(save_path)

# util.save_policy(bc_trainer.policy, save_path)


# episodes = 10
# for episode in range(episodes):
#     obs = env.reset()
    
#     done = False
#     while not done:
#         action = bc_trainer.policy.predict(obs[0])
#         obs, reward, done, info, _ = env.step(action[0])
#         env.render(mode="human")