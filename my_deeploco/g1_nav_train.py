import argparse
import os
import pickle
import shutil
import types
from datetime import datetime
from g1_nav_env import G1DeeplocoEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,  # Increased to encourage more exploration of goal_directed actions
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,  # Further reduced for smoother learning
            "max_grad_norm": 1.0,
            "num_learning_epochs": 10,  # Reduced to prevent overfitting to heuristics footsteps
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 0.25, # Increased to prioritize value learning for goal tasks
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],  # Larger network
            "critic_hidden_dims": [512, 256, 128],  # Larger network
            "init_noise_std": 0.5,  # Increased for more exploration
            "noise_std_type": "log",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 10,  # More frequent logging to monitor goal progress
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": True,  # Start fresh for new task
            "resume_path": "/home/dodolab/tkworkspace/My_deeploco/my_deeploco/log/g1-deeploco-goal_20250426_140711",  
            "run_name": "",
            "runner_class_name": "OnPolicyRunner",
        },
        "save_interval": 50,
        "empirical_normalization": False,
        "num_steps_per_env": 24, # Increased for longer rollouts to capture goal-reaching dynamics
        "seed": 1,
        "logger": "wandb",
        "wandb_project": exp_name,
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
            "left_hip_roll_joint": 0.02,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.1,
            "left_ankle_pitch_joint": -0.1,
            "left_ankle_roll_joint": -0.05,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": -0.02,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.1,
            "right_ankle_pitch_joint": -0.1,
            "right_ankle_roll_joint": 0.05
        },
        "kp": 100.0,
        "kd": 2.5,
        "stiffness": {
            "left_hip_pitch_joint": 120.0,
            "left_hip_roll_joint": 160.0,
            "left_hip_yaw_joint": 100.0,
            "left_knee_joint": 140.0,
            "left_ankle_pitch_joint": 70.0,
            "left_ankle_roll_joint": 70.0,
            "right_hip_pitch_joint": 120.0,
            "right_hip_roll_joint": 160.0,
            "right_hip_yaw_joint": 100.0,
            "right_knee_joint": 140.0,
            "right_ankle_pitch_joint": 70.0,
            "right_ankle_roll_joint": 70.0
        },
        "damping": {
            "left_hip_pitch_joint": 3.5,
            "left_hip_roll_joint": 4.5,
            "left_hip_yaw_joint": 3.0,
            "left_knee_joint": 6.5,
            "left_ankle_pitch_joint": 3.0,
            "left_ankle_roll_joint": 3.0,
            "right_hip_pitch_joint": 3.5,
            "right_hip_roll_joint": 4.5,
            "right_hip_yaw_joint": 3.0,
            "right_knee_joint": 6.5,
            "right_ankle_pitch_joint": 3.0,
            "right_ankle_roll_joint": 3.0
        },
        "terminate_after_contacts_on": ["pelvis"],
        "termination_if_pelvis_z_less_than": 0.35,
        "base_init_pos": [0.0, 0.0, 0.8],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "action_scale": 1.0,
        "episode_length_s": 20.0, # Shorter for faster goal-reaching
        "resampling_time_s": 10.0, # More frequent resampling to capture goal-reaching dynamics
        "simulation_action_latency": False,
        "clip_actions": 1.0,
        "clip_observations": 10.0,
        "feet_height_target": 0.075,
        # New parameters for goal-directed task, updating for new heuristic goal/footstep methods
        "goal_distance_range": [3.0, 10.0],
        "goal_angle_range": [-0.785, 0.785],
        "goal_reached_threshold": 0.2,
        "step_size": 0.10,
        "step_gap": 0.20,
        "period": 1.10,
        "swing_duration": 0.45,
        "stance_duration": 0.65,
    }
    obs_cfg = {
        "num_obs": 3 + 3 + 3 + 12 + 12 + 12 + 2 + 6 + 2,  # adjust as needed
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "goal_pos": 0.1,
            "footstep_targets": 2.0,
        }
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.75,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 0.0,
            "tracking_ang_vel": 0.0,
            "lin_vel_z": -1.0,
            "action_rate": -0.2,
            "base_height": -2.0,
            "alive": 0.2,
            "gait_contact": 0.5,
            "gait_swing": -0.5,
            "contact_no_vel": -0.5,
            "feet_swing_height": 2.0,  # Stronger penalty for low foot clearance
            "orientation": -8.0,  # Stronger penalty for torso roll
            "ang_vel_xy": -1.0,
            "dof_vel": -0.01,
            "knee_angle": 0.5,  # Increased to encourage slightly bent knees
            "feet_angle": -0.01,
            "goal_progress": 3.0,
            "footstep_tracking": 2.0,
            "forward_vel": 0.5,  # Lowered to avoid overwhelming other rewards
            "heading_alignment": 2.5,  # Stronger reward for heading alignment
        }
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 0.3],  # or [0.3, 0.4]
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }
    domain_rand_cfg = {
        "randomize_friction": True,
        "friction_range": [0.8, 1.2],
        "randomize_mass": True,
        "added_mass_range": [-0.1, 0.2],
        "push_robots": True,
        "push_interval_s": 20.0,
        "max_push_vel_xy": 0.1,  # Reduced for goal-reaching
        "max_push_vel_rp": 0.5,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg

def main():
    parser = argparse.ArgumentParser(description="G1 Deeploco Goal-Directed Training Script")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-deeploco-goal")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)  # Reduced for stability
    parser.add_argument("--max_iterations", type=int, default=10_000)
    parser.add_argument("--show_viewer", action="store_true", help="Show the viewer during training")
    parser.add_argument("--path", type=str, default=None)  # No default checkpoint
    args = parser.parse_args()

    gs.init(logging_level="warning")

    # Generate a unique log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"my_deeploco/log/{args.exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    def env_modifier(env):
        for i, name in enumerate(env_cfg["dof_names"]):
            env.robot.set_dofs_kp([env_cfg["stiffness"][name]], [env.motor_dofs[i]])
            env.robot.set_dofs_kv([env_cfg["damping"][name]], [env.motor_dofs[i]])
        ankle_indices = [4, 5, 10, 11]
        hip_roll_indices = [1, 7]
        knee_indices = [3, 9]
        env.robot.set_dofs_kp([100.0] * len(ankle_indices), ankle_indices)
        env.robot.set_dofs_kv([15.0] * len(ankle_indices), ankle_indices)
        env.robot.set_dofs_kp([150.0] * len(hip_roll_indices), hip_roll_indices)
        env.robot.set_dofs_kv([20.0] * len(hip_roll_indices), hip_roll_indices)
        env.robot.set_dofs_kp([120.0] * len(knee_indices), knee_indices)
        env.robot.set_dofs_kv([15.0] * len(knee_indices), knee_indices)  # Increased damping
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