import argparse
import os
import pickle
import shutil
import types  # Add this import
from datetime import datetime
from g1_env import G1DeeplocoEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 10,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.25,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 0.5,
            "noise_std_type": "log",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 10,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": True,
            "resume_path": "home/dodolab/tkworkspace/My_deeploco/log/g1-deeploco-walk",
            "run_name": "",
            "runner_class_name": "OnPolicyRunner",
        },
        "save_interval": 50,
        "empirical_normalization": False,
        "num_steps_per_env": 24,
        "seed": 1,
    }

    return train_cfg_dict

def get_cfgs():
    env_cfg = {
        "num_actions": 12, 
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
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.02,      # Slight outward roll
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.1,
            "left_ankle_roll_joint": -0.05,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": -0.02,    # Mirror hip roll
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.1,
            "right_ankle_roll_joint": 0.05   # Mirror of left
        },
        # PD gains matching Unitree config
        "kp": 100.0,
        "kd": 2.5,
        "stiffness": {
            "left_hip_pitch_joint": 100.0,   # Increased from 88.0
            "left_hip_roll_joint": 150.0,    # Increased from 139.0
            "left_hip_yaw_joint": 100.0,     # Increased from 88.0
            "left_knee_joint": 150.0,        # Increased from 139.0
            "left_ankle_pitch_joint": 60.0,  # Increased from 50.0
            "left_ankle_roll_joint": 60.0,   # Increased from 50.0
            "right_hip_pitch_joint": 100.0,  # Increased from 88.0
            "right_hip_roll_joint": 150.0,   # Increased from 139.0
            "right_hip_yaw_joint": 100.0,    # Increased from 88.0
            "right_knee_joint": 150.0,       # Increased from 139.0
            "right_ankle_pitch_joint": 60.0, # Increased from 50.0
            "right_ankle_roll_joint": 60.0   # Increased from 50.0
        },
        "damping": {
            "left_hip_pitch_joint": 3.0,     # Increased from 2.0
            "left_hip_roll_joint": 4.0,      # Increased from 2.5
            "left_hip_yaw_joint": 3.0,       # Increased from 2.0
            "left_knee_joint": 6.0,          # Increased from 5.0
            "left_ankle_pitch_joint": 2.5,   # Increased from 1.5
            "left_ankle_roll_joint": 2.5,    # Increased from 1.5
            "right_hip_pitch_joint": 3.0,    # Increased from 2.0
            "right_hip_roll_joint": 4.0,     # Increased from 2.5
            "right_hip_yaw_joint": 3.0,      # Increased from 2.0
            "right_knee_joint": 6.0,         # Increased from 5.0
            "right_ankle_pitch_joint": 2.5,  # Increased from 1.5
            "right_ankle_roll_joint": 2.5    # Increased from 1.5
        },
        # Termination
        "terminate_after_contacts_on": ["pelvis"],  # Remove knee links to allow for recovery
        "termination_if_pelvis_z_less_than": 0.35,  # Reduced from 0.5 to allow crouching
        # Base pose
        "base_init_pos": [0.0, 0.0, 0.8],  # Increased height for better starting position
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "action_scale": 1.0,
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,
        "simulation_action_latency": False,
        "clip_actions": 100.0,
        "clip_observations": 100.0,
        "feet_height_target": 0.085,  
    }
    obs_cfg = {
        "num_obs": 47,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.75,  # Increased from 0.6 for more upright posture
        "feet_height_target": 0.085,   # Increased from 0.08 for better foot clearance
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -1.0,
            "action_rate": -0.2,
            "base_height": -30.0,
            "alive": 0.2,
            "gait_contact": 0.5,
            "gait_swing": -0.5,
            "contact_no_vel": -0.5,
            "feet_swing_height": -20.0,
            "orientation": -1.0,
            "ang_vel_xy": -1.0,
            "dof_vel": -0.01,
            "knee_angle": 0.1,
            "feet_angle": -0.01,
        }
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 0.3],
        "lin_vel_y_range": [-0.2, 0.2],
        "ang_vel_range": [-0.5, 0.5],
    }
    domain_rand_cfg = {
        "randomize_friction": True,
        "friction_range": [0.8, 1.2],
        "randomize_mass": True,
        "added_mass_range": [-0.1, 0.2],
        "push_robots": True,
        "push_interval_s": 20.0,
        "max_push_vel_xy": 0.2,
        "max_push_vel_rp": 0.5,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg

def main():
    parser = argparse.ArgumentParser(description="G1 Deeploco Training Script")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-deeploco-walk")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10_000)
    parser.add_argument("--show_viewer", action="store_true", help="Show the viewer during training")
    parser.add_argument("--path", type=str, default="home/dodolab/tkworkspace/My_deeploco/my_deeploco/log/g1-deeploco-walk")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    # Generate a unique log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"my_deeploco/log/{args.exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Create a function to modify the environment's joint stiffness and damping
    def env_modifier(env):
        # Apply stiffness and damping values from config
        for i, name in enumerate(env_cfg["dof_names"]):
            env.robot.set_dofs_kp([env_cfg["stiffness"][name]], [env.motor_dofs[i]])
            env.robot.set_dofs_kv([env_cfg["damping"][name]], [env.motor_dofs[i]])
        
        # Adjust specific joint gains for better walking stability
        ankle_indices = [4, 5, 10, 11]
        hip_roll_indices = [1, 7]
        knee_indices = [3, 9]

        env.robot.set_dofs_kp([100.0] * len(ankle_indices), ankle_indices)
        env.robot.set_dofs_kv([15.0] * len(ankle_indices), ankle_indices)
        env.robot.set_dofs_kp([150.0] * len(hip_roll_indices), hip_roll_indices)
        env.robot.set_dofs_kv([20.0] * len(hip_roll_indices), hip_roll_indices)
        env.robot.set_dofs_kp([120.0] * len(knee_indices), knee_indices)
        env.robot.set_dofs_kv([12.0] * len(knee_indices), knee_indices)

        # Remove domain randomization delay logic, always use original step
        return env

    env = G1DeeplocoEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_rand_cfg=domain_rand_cfg,
        show_viewer=args.show_viewer,
        device="cuda"
    )
    
    # Apply the environment modifications
    env = env_modifier(env)
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    if args.path and os.path.exists(args.path):
        runner.load(args.path)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_rand_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()