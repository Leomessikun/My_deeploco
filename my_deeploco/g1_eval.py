import torch
import os 
import pickle
import argparse
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

   # Load configuration
    try:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        print("Configurations loaded successfully.")
    except Exception as e:
        print(f"Failed to load configurations: {e}")
        return
    
    # disable reward scales for evaluation 
    reward_cfg["reward_scales"] = {}

    # create the environment
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

    env = DummyVecEnv([
        make_env(env_cfg,obs_cfg,reward_cfg,command_cfg)
        ])
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # load normalized statistics
    env_save_path = os.path.join(log_dir, "vec_normalize_final.pkl")
    try: 
        env = VecNormalize.load(env_save_path, env)
        print(f"Normalization statistics loaded from {env_save_path}.")
    except Exception as e:
        print(f"Failed to load normalization statistics: {e}")
        return

    # load the trained model
    model_path = os.path.join(log_dir, "ppo_g1_final")
    try:
        model = PPO.load(model_path, env=env, device="cuda")
        print(f"Model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # reset the environment
    obs = env.reset()

   # Run the simulation
    try:
        with torch.no_grad():
            while True:
                # Get actions from the policy
                actions, _ = model.predict(obs, deterministic=True)

                # Step the environment
                obs, rewards, done, infos = env.step(actions)

                # Render the environment (if visualization is enabled)
                env.render()

                # Check for termination
                if done.any():
                    obs = env.reset()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()