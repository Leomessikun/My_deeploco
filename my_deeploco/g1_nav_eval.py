import argparse
import os
import pickle
import torch
from g1_nav_env import G1DeeplocoEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
import time
import numpy as np
from datetime import datetime

def get_eval_train_cfg():
    """Define a minimal train_cfg for evaluation."""
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",  # Explicitly set to match training
        },
        "policy": {
            "class_name": "ActorCritic",  # Match training policy
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 0.1,  # Lower noise for smoother evaluation
            "noise_std_type": "log"  # Use log parameterization like in training
        },
        "num_steps_per_env": 128,  # Required by OnPolicyRunner
        "save_interval": 50,       # Required by OnPolicyRunner
        "empirical_normalization": False,  # Required by OnPolicyRunner
    }
    return train_cfg_dict

def add_metrics_tracking(env):
    """Add metrics tracking to the environment for better visualization."""
    import types
    
    # Setup metrics buffers
    env.metrics = {
        "foot_clearance": [],
        "step_length": [],
        "base_height": [],
        "forward_velocity": [],
        "lateral_velocity": [],
        "angular_velocity": [],
        "alternating_contacts": [],
        #"feet_air_time": [],
        "torso_roll": [],
        "torso_pitch": []
    }
    
    # Override step function to collect metrics
    original_step = env.step
    
    def step_with_metrics(self, actions):
        obs, rew, reset, extras = original_step(actions)
        
        # Collect metrics
        if hasattr(self, 'feet_pos'):
            # Get foot heights when not in contact
            contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
            feet_height = self.feet_pos[:, :, 2]
            non_contact_heights = torch.where(
                ~contact,
                feet_height,
                torch.zeros_like(feet_height)
            )
            mean_clearance = torch.mean(non_contact_heights[non_contact_heights > 0]).item() if torch.any(non_contact_heights > 0) else 0
            
            # Step length
            feet_pos_rel = self.feet_pos - self.base_pos.unsqueeze(1)
            mean_step_length = torch.mean(torch.abs(feet_pos_rel[:, :, 0])).item()
            
            # Base height
            mean_base_height = torch.mean(self.base_pos[:, 2]).item()
            
            # Velocities
            mean_forward_vel = torch.mean(self.base_lin_vel[:, 0]).item()
            mean_lateral_vel = torch.mean(self.base_lin_vel[:, 1]).item()
            mean_angular_vel = torch.mean(self.base_ang_vel[:, 2]).item()
            
            # Alternating gait - measure if left/right feet are alternating
            alternating = torch.logical_xor(contact[:, 0], contact[:, 1]).float()
            mean_alternating = torch.mean(alternating).item()
            
            # Feet air time - average time feet spend in air
            # mean_air_time = torch.mean(self.feet_air_time).item()
            
            # Torso orientation
            mean_roll = torch.mean(torch.abs(self.base_euler[:, 0])).item()  # Roll (lateral tilt)
            mean_pitch = torch.mean(torch.abs(self.base_euler[:, 1])).item()  # Pitch (forward tilt)
            
            # Store metrics
            self.metrics["foot_clearance"].append(mean_clearance)
            self.metrics["step_length"].append(mean_step_length)
            self.metrics["base_height"].append(mean_base_height)
            self.metrics["forward_velocity"].append(mean_forward_vel)
            self.metrics["lateral_velocity"].append(mean_lateral_vel)
            self.metrics["angular_velocity"].append(mean_angular_vel)
            self.metrics["alternating_contacts"].append(mean_alternating)
            #self.metrics["feet_air_time"].append(mean_air_time)
            self.metrics["torso_roll"].append(mean_roll)
            self.metrics["torso_pitch"].append(mean_pitch)
            
            # Print metrics occasionally
            if len(self.metrics["foot_clearance"]) % 50 == 0:
                print(f"\nMetrics (avg last 50 steps):")
                print(f"  Velocity: forward={sum(self.metrics['forward_velocity'][-50:]) / 50:.3f} m/s, "
                      f"lateral={sum(self.metrics['lateral_velocity'][-50:]) / 50:.3f} m/s, "
                      f"angular={sum(self.metrics['angular_velocity'][-50:]) / 50:.3f} rad/s")
                print(f"  Gait: clearance={sum(self.metrics['foot_clearance'][-50:]) / 50:.3f} m, "
                      f"step length={sum(self.metrics['step_length'][-50:]) / 50:.3f} m, "
                      #f"air time={sum(self.metrics['feet_air_time'][-50:]) / 50:.3f} s, "
                      f"alternating={sum(self.metrics['alternating_contacts'][-50:]) / 50:.3f}")
                print(f"  Stability: height={sum(self.metrics['base_height'][-50:]) / 50:.3f} m, "
                      f"roll={sum(self.metrics['torso_roll'][-50:]) / 50:.3f} rad, "
                      f"pitch={sum(self.metrics['torso_pitch'][-50:]) / 50:.3f} rad")
                
        return obs, rew, reset, extras
    
    env.step = types.MethodType(step_with_metrics, env)
    return env

def main():
    parser = argparse.ArgumentParser(description="G1 Deeploco Evaluation Script")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-deeploco-walk")
    parser.add_argument("--ckpt", type=int, default=1000, help="Checkpoint iteration to load")
    parser.add_argument("--cmd_x", type=float, default=0.3, help="Forward velocity command")
    parser.add_argument("--cmd_y", type=float, default=0.0, help="Lateral velocity command")
    parser.add_argument("--cmd_yaw", type=float, default=0.0, help="Yaw velocity command")
    parser.add_argument("--record", action="store_true", help="Record video of evaluation")
    parser.add_argument("--duration", type=int, default=60, help="Evaluation duration in seconds")
    args = parser.parse_args()

    gs.init()

    # Load environment configs from cfgs.pkl, but not train_cfg
    log_dir = f"/home/dodolab/tkworkspace/My_deeploco/my_deeploco/log/{args.exp_name}"
    if not os.path.exists(f"{log_dir}/cfgs.pkl"):
        raise FileNotFoundError(f"Configuration file {log_dir}/cfgs.pkl not found. Run training first.")
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_saved_cfg, domain_rand_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    
    # Set reward scales to zero for evaluation (instead of clearing them)
    original_reward_scales = reward_cfg["reward_scales"].copy()
    reward_cfg["reward_scales"] = {key: 0.0 for key in original_reward_scales.keys()}

    # Set fixed commands for evaluation
    command_cfg["curriculum"] = False
    command_cfg["ranges"] = {
        "lin_vel_x": [args.cmd_x, args.cmd_x],
        "lin_vel_y": [args.cmd_y, args.cmd_y],
        "ang_vel_yaw": [args.cmd_yaw, args.cmd_yaw],
        "heading": [0.0, 0.0]
    }

    # Disable domain randomization for evaluation
    domain_rand_cfg["randomize_friction"] = False
    domain_rand_cfg["randomize_mass"] = False
    domain_rand_cfg["push_robots"] = False

    # Create environment with 1 env for evaluation
    env = G1DeeplocoEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_rand_cfg=domain_rand_cfg,
        show_viewer=True,
        device="cuda:0"  # Match training device
    )
    
    # Add metrics tracking
    env = add_metrics_tracking(env)

    # Try to detect whether the model uses log_std or std parameter
    # First, check the saved training config if it exists
    if train_saved_cfg and "policy" in train_saved_cfg and "noise_std_type" in train_saved_cfg["policy"]:
        noise_std_type = train_saved_cfg["policy"]["noise_std_type"]
    else:
        # Default to log for newer models
        noise_std_type = "log"
    
    # Use a fresh train_cfg for evaluation
    train_cfg = get_eval_train_cfg()
    train_cfg["policy"]["noise_std_type"] = noise_std_type
    
    print(f"Using noise_std_type: {noise_std_type}")
    print(f"Evaluating with commands: forward={args.cmd_x}, lateral={args.cmd_y}, yaw={args.cmd_yaw}")

    # Initialize runner and load checkpoint
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint {resume_path} not found.")
    
    # Try loading the model, if it fails due to std/log_std mismatch, try the other type
    try:
        runner.load(resume_path)
    except RuntimeError as e:
        if "Missing key(s) in state_dict: \"std\"" in str(e):
            print("Model uses log_std instead of std, updating config...")
            train_cfg["policy"]["noise_std_type"] = "log"
            runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
            runner.load(resume_path)
        elif "Missing key(s) in state_dict: \"log_std\"" in str(e):
            print("Model uses std instead of log_std, updating config...")
            train_cfg["policy"]["noise_std_type"] = "scalar"
            runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
            runner.load(resume_path)
        else:
            raise e
            
    policy = runner.get_inference_policy(device="cuda:0")

    # Setup video recording if requested
    if args.record:
        record_dir = os.path.join(log_dir, "videos")
        os.makedirs(record_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(record_dir, f"eval_x{args.cmd_x}_y{args.cmd_y}_yaw{args.cmd_yaw}_{timestamp}.mp4")
        print(f"Recording video to {video_path}")
        env.scene.toggle_video_recording(video_path)
    
    # Reset environment and run evaluation loop
    obs, _ = env.reset()
    start_time = time.time()
    step_count = 0
    
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            
            if hasattr(env, '_visualize_footstep_targets'):
                env._visualize_footstep_targets()
            
            step_count += 1
            elapsed_time = time.time() - start_time
            
            # Print status every 5 seconds
            if step_count % 250 == 0:
                print(f"Evaluation progress: {elapsed_time:.1f}s / {args.duration}s")
            
            # Reset if done or if we've reached the evaluation duration
            if dones.any() or elapsed_time >= args.duration:
                if elapsed_time >= args.duration:
                    print(f"\nEvaluation completed after {elapsed_time:.1f} seconds ({step_count} steps)")
                    # Print summary statistics
                    if hasattr(env, 'metrics') and len(env.metrics['forward_velocity']) > 0:
                        print("\nFinal metrics summary:")
                        metrics_len = min(step_count, len(env.metrics['forward_velocity']))
                        print(f"  Avg forward velocity: {np.mean(env.metrics['forward_velocity'][-metrics_len:]):.3f} m/s")
                        print(f"  Avg step length: {np.mean(env.metrics['step_length'][-metrics_len:]):.3f} m")
                        print(f"  Avg foot clearance: {np.mean(env.metrics['foot_clearance'][-metrics_len:]):.3f} m")
                        print(f"  Avg alternating gait: {np.mean(env.metrics['alternating_contacts'][-metrics_len:]):.3f}")
                        print(f"  Avg torso roll: {np.mean(env.metrics['torso_roll'][-metrics_len:]):.3f} rad")
                    
                    if args.record:
                        print("Finalizing video recording...")
                        env.scene.toggle_video_recording(None)
                    break
                
                obs, _ = env.reset()  # Reset if done

if __name__ == "__main__":
    main() 