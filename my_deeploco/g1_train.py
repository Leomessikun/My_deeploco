import argparse
import os 
import pickle
import shutil
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv 
from g1_env import G1DeeplocoEnv
from g1_gym_wrapper import G1DeeplocoGymWrapper 
import genesis as gs

def get_train_cfg(exp_name, max_iterations):
    """
    define training configuration.
    """ 
    train_cfg = {
        "algorithm":{
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
            "n_epochs": 10,
            "n_steps": 2048,
            "batch_size": 64,
        },
        "policy": {
            "net_arch": [dict(pi=[512, 256, 128], vf=[512, 256, 128])],
            "activation_fn": torch.nn.ELU,
        },
        "runner": {
            "experiment_name": exp_name,
            "max_iterations": max_iterations,
            "log_interval": 1,
            "save_interval": 100,
        },
    }
    return train_cfg

def get_cfgs():
    """
    define environment, observation, reward, and command configurations.
    """
    env_cfg={
        "num_actions": 29,  # Updated to match the URDF
        "default_joint_angles": {
             "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        "dof_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "base_init_pos": [0.0, 0.0, 0.5],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "clip_actions": 100.0
    }

    obs_cfg={
        "num_obs": 64,  
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    reward_cfg={
        "tracking_sigma": 0.25,
        "height_sigma": 0.1,
        "clearance_sigma": 0.05,
        "contact_sigma": 0.1,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "gait_mode": 0.1,
            "orientation": 0.5,
            "energy":-0.01,
            "action_smoothness": -0.005,
            "base_height": -50.0,
            "foot_clearance": 0.2,
            "action_rate": -0.005,
            "joint_limits": -0.1,  # Penalty for exceeding joint limits
            "torque": -0.01,
            "contact_consistency": 0.2,
            #"energy_consumption": -0.01,  # Penalty for high energy usage
        },
    }

    command_cfg={
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 1.5],
        "lin_vel_y_range": [-0.2, 0.2],
        "ang_vel_yaw_range": [-0.5, 0.5],
        "gait_mode_range": [0, 2],
        "locomotion_param_range": [0.1, 0.3],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1-deeploco")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")    
    args = parser.parse_args()

    # intialize genesis
    gs.init(logging_level="warning")

    # define log directory
    log_dir = f"log/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # load configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Save configurations
    try:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )
        print(f"Configurations saved to {log_dir}/cfgs.pkl")
    except Exception as e:
        print(f"Failed to save configurations: {e}")
        return

    # create vectorized environment
    def make_env(env_cfg, obs_cfg, reward_cfg, command_cfg):
        def __init__():
            env = G1DeeplocoEnv(
                num_envs=1,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                device="cuda" 
            )
            return G1DeeplocoGymWrapper(env)

        return __init__
    
    try:
        env = SubprocVecEnv([
            make_env(env_cfg, obs_cfg, reward_cfg, command_cfg) for _ in range(args.num_envs)
        ])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        print("Environment created successfully.")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return

    # Initialize PPO model
    try:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=train_cfg["policy"],
            learning_rate=train_cfg["algorithm"]["learning_rate"],
            n_steps=train_cfg["algorithm"]["n_steps"],
            batch_size=train_cfg["algorithm"]["batch_size"],
            n_epochs=train_cfg["algorithm"]["n_epochs"],
            gamma=train_cfg["algorithm"]["gamma"],
            gae_lambda=train_cfg["algorithm"]["gae_lambda"],
            clip_range=train_cfg["algorithm"]["clip_param"],
            ent_coef=train_cfg["algorithm"]["entropy_coef"],
            max_grad_norm=train_cfg["algorithm"]["max_grad_norm"],
            verbose=1,
            tensorboard_log=log_dir,
            device=args.device,
        )
        print("PPO model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize PPO model: {e}")
        return

   # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["runner"]["save_interval"] * args.num_envs,
        save_path=log_dir,
        name_prefix="ppo_g1",
    )
    eval_callback = EvalCallback(
        env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=train_cfg["runner"]["save_interval"] * args.num_envs,
        deterministic=True,
        render=False,
    )

    # Train the model
    try:
        model.learn(
            total_timesteps=train_cfg["runner"]["max_iterations"] * train_cfg["algorithm"]["n_steps"] * args.num_envs,
            callback=[checkpoint_callback, eval_callback],
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
   # Save the final model and normalization statistics
    try:
        model_save_path = f"{log_dir}/ppo_g1_final"
        model.save(model_save_path)
        print(f"Model saved to: {model_save_path}.zip")

        env_save_path = f"{log_dir}/vec_normalize_final.pkl"
        env.save(env_save_path)
        print(f"Normalization statistics saved to: {env_save_path}")
    except Exception as e:
        print(f"Failed to save final model or normalization: {e}")
    
if __name__ == "__main__":
    main()