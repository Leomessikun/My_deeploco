import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class G1DeeplocoGymWrapper(gym.Env):
    def __init__(self, genesis_env):
        """
        wrap the genesis-based G1DeeplocoEnv to make it compatible with Gymnasium.

        Args:
            genesis_env(G1DeeplocoEnv): your existing Genesis-based environment.
        """
        super(G1DeeplocoGymWrapper, self).__init__()

        self.genesis_env = genesis_env

        # define observation and action spaces 
        self. observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(100,), # observation dimension
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.genesis_env.num_actions,), # action dimension
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        reset the environment and return the initial observation.

        Args:
            seed(int): Seed for random number generation.
            options(dict): Addotional options for reset.

        Returns:
            obs(np.ndarray): Initial observation.
            info(dict): Additional information.
        """

        # reset the genesis environment
        obs_buf= self.genesis_env.reset()

        # convert observation to numpy array
        obs = obs_buf.cpu().numpy()

        # return observation and info
        return obs, {}
    
    def step(self, action):
        """
        take a step in the environment.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            obs (np.ndarray): Next observation.
            reward (float): Reward for the step.
            done (bool): Whether the episode is done.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """

        # convert action totensor if necessary
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.genesis_env.device, dtype=torch.float32)

        # perform the step in the genesis environment
        obs_buf, rew_buf, reset_buf, extras = self.genesis_env.step(action)
        obs = obs_buf.cpu().numpy()
        rewards = rew_buf.cpu().numpy()
        terminated = reset_buf.cpu().numpy()
        truncated = np.zeros_like(terminated)
        infos = extras

        # return observation, reward, done, truncated, and info
        return obs, rewards, terminated, truncated, infos
    
    def close(self):
        """Clean up the environment."""
        self.genesis_env.close()