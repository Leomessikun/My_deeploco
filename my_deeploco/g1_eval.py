import torch
import os 
import pickle
import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from g1_env import G1DeeplocoEnv
from g1_gym_wrapper import G1DeeplocoGymWrapper
import genesis as gs


def main():
    # parse arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-deeploco", help="Name of the experiment.")
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint number to load.")
    args = parser.parse_args()

    # initialize genesis
    gs.init(logging_level="warning")

    # define log directory
    log_dir = f"log/{args.exp_name}"

    # load configuration 
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # disable reward scales for evaluation 
    reward_cfg["reward_scales"] = {}

    # create the environment
     # Create the environment
    def make_env(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg):
        def _init():
            env = G1DeeplocoEnv(
                num_envs=num_envs,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=True,  # Enable visualization
                device="cuda",  # or "cpu" if you're not using GPU
            )
            return G1DeeplocoGymWrapper(env)
        return _init

    env = VecNormalize(
        DummyVecEnv([
            make_env(
                num_envs=1,  # Each environment instance handles one environment
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
            )
        ]),
        norm_obs=True,
        norm_reward=True,
    )

    # load the trained model
    model_path = os.path.join(log_dir, f"ppo_g1_{args.ckpt}.zip")
    model = PPO.load(model_path, env=env, device="cuda") 

    # reset the environment
    obs = env.reset()

    # run the simulation 
    with torch.no_grad():
        while True:
            # get actions from the policy
            actions, _ = model.predict(obs, deterministic=True)

            # step the enviroment
            obs, rewards, done, infos = env.step(actions)

            # render the environment (if visualization is enabled)
            env.render()

            # check for termination 
            if done.any():
                obs=env.reset()

if __name__ == "__main__":
    main()