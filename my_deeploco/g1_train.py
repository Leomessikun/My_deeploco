import argparse
import os
import pickle
import shutil
from g1_env import G1DeeplocoEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 0.5,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 10,
            "max_iterations": max_iterations,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "OnPolicyRunner",
            "num_steps_per_env": 32,
            "save_interval": 50,
            "empirical_normalization": False,
        },
        "seed": 1,
    }

    return train_cfg_dict

def get_cfgs():
    env_cfg = {
        "dof_names": [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint"
        ],
        "default_joint_angles": {
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0
        },
        "kp": 80.0,
        "kd": 2.0,
        # Termination
        "terminate_after_contacts_on": ["pelvis"],
        "termination_if_pelvis_z_less_than": 0.35,
        # Base pose
        "base_init_pos": [0.0, 0.0, 0.8],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "action_scale": 0.3,
        "episode_length_s": 10.0,
        "resampling_time_s": 5.0,
        "simulation_action_latency": True,
        "clip_actions": 5.0,
        "clip_observations": 5.0,
    }
    obs_cfg = {
        "num_obs": 48,
        "obs_scales": {
            "lin_vel": 2.5,
            "ang_vel": 0.3,
            "dof_pos": 1.0,
            "dof_vel": 0.07,
            "heading": 0.15,
        }
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.45,
        "feet_height_target": 0.08,
        "reward_scales": {
            "tracking_lin_vel": 4.0,
            "tracking_ang_vel": 1.0,
            "alive": 0.5,
            "gait_contact": 2.0,
            "gait_swing": 2.0,
            "knee_angle": 0.3,
            "lin_vel_z": -0.2,
            "ang_vel_xy": -0.1,
            "base_height": -0.5,
            "orientation": -0.3,
            "action_rate": -0.01,
            "dof_vel": -0.005,
            "contact_no_vel": -0.05,
            "feet_swing_height": -0.5,
            "feet_angle": -0.1,
        },
    }
    command_cfg = {
        "curriculum": True,
        "max_curriculum": 1.0,
        "num_commands": 4,
        "heading_command": True,
        "curriculum_steps": 5000,
        "ranges": {
        "lin_vel_x": [0.0, 1.0],  # Increase max forward speed from 0.5 to 1.0
        "lin_vel_y": [-0.3, 0.3],  # Slightly widen lateral movement 
        "ang_vel_yaw": [-0.5, 0.5],  # Increase rotation range
        "heading": [-1.0, 1.0]  # Wider heading range
    }
    }
    domain_rand_cfg = {
        "randomize_friction": True,
        "friction_range": [0.7, 1.3],  # More conservative range
        "randomize_mass": True,
        "added_mass_range": [-1.0, 1.0],  # Smaller mass variation
        "push_robots": True,
        "push_interval_s": 7.0,
        "max_push_vel_xy": 1.5,  # Moderate pushes
        "max_push_vel_rp": 45.0,  # Degrees/second, more conservative
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg

def main():
    parser = argparse.ArgumentParser(description="G1 Deeploco Training Script")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-deeploco-walk")
    parser.add_argument("-B", "--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=10_000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"log/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env = G1DeeplocoEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_rand_cfg=domain_rand_cfg,
        show_viewer=False,
        device="cuda"
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_rand_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()