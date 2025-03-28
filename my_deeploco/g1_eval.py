import argparse
import os
import pickle
import torch
from g1_env import G1DeeplocoEnv
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

def main():
    parser = argparse.ArgumentParser(description="G1 Deeploco Evaluation Script")
    parser.add_argument("-e", "--exp_name", type=str, default="g1-deeploco-walk")
    parser.add_argument("--ckpt", type=int, default=1000)
    args = parser.parse_args()

    gs.init()

    log_dir = f"log/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, domain_rand_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    # Clear reward scales for evaluation (no training)
    reward_cfg["reward_scales"] = {}

    env = G1DeeplocoEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        domain_rand_cfg=domain_rand_cfg,
        show_viewer=True,
        device="cuda"
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda")

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    main()