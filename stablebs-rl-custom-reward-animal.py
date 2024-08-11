
import datetime
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from loco_mujoco import LocoEnv

from scipy.spatial.transform import Rotation as R

# implementation of https://xbpeng.github.io/projects/Robotic_Imitation/Robotic_Imitation_2020.pdf

KINEMATIC_INDEX = 17


def experiment(alg, total_timesteps):

    class CustomRewardWrapper(gym.Wrapper):
        def __init__(self, env, expert_model):
            super().__init__(env)
            self.env = env
            self.expert_model = expert_model
        def reset(self, **kwargs):
            self.current_step = 0
            return self.env.reset(**kwargs)

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Calculate custom reward
            custom_reward = self.custom_reward_function(observation, action, reward, info)

            return observation, custom_reward, terminated, truncated, info

        def custom_reward_function(self, observation, action, original_reward, info):


            def pose_reward(observation, ref_observation):
                # Compare all joint positions
                pose_diff = np.linalg.norm(observation[:17] - ref_observation[:17])
                return np.exp(-5 * pose_diff)  # Adjust the scaling factor as needed

            def joint_velocity_reward(observation, ref_observation):
                # Compare all joint velocities
                vel_diff = np.linalg.norm(observation[17:] - ref_observation[17:])
                return np.exp(-0.1 * vel_diff)  # Adjust the scaling factor as needed

            x_vel_idx = KINEMATIC_INDEX # dq_pelvis_tx
            x_vel = observation[x_vel_idx]
            target_vel = 1.25
            velocity_reward = np.exp(- 10* np.square(x_vel - target_vel))


            # Get the reference pose and velocity for the current timestep
            import pdb; pdb.set_trace()
            action, reference_state = self.expert_model.predict(observation, deterministic=True)

            # Calculate pose reward
            p_reward = pose_reward(observation, reference_state)

            # Calculate joint velocity reward
            jv_reward = joint_velocity_reward(observation, reference_state)

            # Combine rewards
            w_p, w_jv, w_v = 0.5, 0.4, 0.09  # Adjust these weights as needed
            combined_reward = w_p * p_reward + w_jv * jv_reward + w_v * velocity_reward


            return combined_reward
    

    # Create the original environment
    env = gym.make("LocoMujoco", env_name="HumanoidTorque.walk.perfect", render_mode="human")

    # Create the dataset
    # dataset = env.create_dataset()
    # expert_observations = dataset['states']
    # expert_actions = dataset['actions']
    expert_model = SAC.load("logs_sb3/sac_bc_20240809/sac_humanoid_bc", env=env)

    env = CustomRewardWrapper(env, expert_model)


    model = SAC('MlpPolicy', env, verbose=1)

    today = datetime.date.today().strftime("%Y-%m-%d")
    log_dir = f'./logs/sb3/{today}/{alg.__name__}/expert_model_rew_0.5_0.4_0.09/'

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model.set_logger(new_logger)

    eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=max(500, 1),
                                n_eval_episodes=5, deterministic=True,
                                render=False)

    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=eval_callback)



if __name__ == '__main__':
    save = True
    load = False
    experiment(alg=SAC, total_timesteps=int(1e5))