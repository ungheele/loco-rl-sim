
import datetime
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from loco_mujoco import LocoEnv

def experiment(alg, total_timesteps):

    class CustomRewardWrapper(gym.Wrapper):
        def __init__(self, env, expert_observations, expert_actions):
            super().__init__(env)
            self.env = env
            self.expert_observations = expert_observations
            self.expert_actions = expert_actions
            self.current_step = 0
            self.episode_length = len(expert_observations)
        def reset(self, **kwargs):
            self.current_step = 0
            return self.env.reset(**kwargs)

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Calculate custom reward
            custom_reward = self.custom_reward_function(observation, action, reward, info)

            self.current_step = (self.current_step + 1) % self.episode_length
            
            return observation, custom_reward, terminated, truncated, info

        def custom_reward_function(self, observation, action, original_reward, info):
            # def find_nearest_demo(observation, expert_observations):
            #     distances = np.sum((expert_observations - observation)**2, axis=1)
            #     nearest_index = np.argmin(distances)
            #     return nearest_index

            # def calculate_bc_loss(observation, action, expert_observations, expert_actions):
            #     nearest_index = find_nearest_demo(observation, expert_observations)
            #     bc_loss = np.mean((action - expert_actions[nearest_index])**2)
            #     return bc_loss, nearest_index
            # def calculate_pose_similarity(observation, expert_observation):
            #     # Assuming the first 17 elements are joint positions
            #     pose_diff = np.linalg.norm(observation - expert_observation)
            #     pose_similarity = np.exp(-0.1 * pose_diff)  # Adjust the scaling factor as needed
            #     return pose_similarity

            def pose_reward(current_pose, reference_pose):
                # Compare all joint positions
                pose_diff = np.linalg.norm(current_pose[:17] - reference_pose[:17])
                return np.exp(-2 * pose_diff)  # Adjust the scaling factor as needed

            def joint_velocity_reward(current_vel, reference_vel):
                # Compare all joint velocities
                vel_diff = np.linalg.norm(current_vel[17:] - reference_vel[17:])
                return np.exp(-0.1 * vel_diff)  # Adjust the scaling factor as needed



            x_vel_idx = 17 # dq_pelvis_tx
            x_vel = observation[x_vel_idx]
            target_vel = 1.25
            velocity_reward = np.exp(- 10* np.square(x_vel - target_vel))

            # # Calculate BC loss
            # bc_loss, nearest_index = calculate_bc_loss(observation, action, expert_observations, expert_actions)
            # pose_similarity = calculate_pose_similarity(observation, self.expert_observations[nearest_index])
            # w_vel, w_bc, w_pose = 0.4, 0.3, 0.3  # Adjust these weights as needed
            # combined_reward = w_vel * velocity_reward - w_bc * bc_loss + w_pose * pose_similarity

        # Get the reference pose and velocity for the current timestep
            reference_state = self.expert_observations[self.current_step]

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
    dataset = env.create_dataset()
    expert_observations = dataset['states']
    expert_actions = dataset['actions']

    env = CustomRewardWrapper(env, expert_observations, expert_actions)


    model = SAC('MlpPolicy', env, verbose=1)

    today = datetime.date.today().strftime("%Y-%m-%d")
    log_dir = f'./logs/sb3/{today}/{alg.__name__}/joint_vel_reward_0.5_0.4_0.09/'

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model.set_logger(new_logger)

    eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=max(500, 1),
                                n_eval_episodes=5, deterministic=True,
                                render=False)

    # Train the model
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)



if __name__ == '__main__':
    save = True
    load = False
    experiment(alg=SAC, total_timesteps=int(5e5))