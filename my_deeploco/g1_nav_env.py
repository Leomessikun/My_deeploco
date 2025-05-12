import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, xyz_to_quat, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class G1DeeplocoEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, domain_rand_cfg, show_viewer=True, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        # Update num_obs to include goal position (2) and footstep targets (4)
        self.num_obs = obs_cfg["num_obs"]  # Existing obs + 2 (goal) + 4 (footsteps)
        self.num_privileged_obs = None
        self.num_actions = env_cfg.get("num_actions", len(env_cfg["dof_names"]))
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg.get("simulation_action_latency", False)
        self.dt = 0.02  # 50Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.domain_rand_cfg = domain_rand_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device, dtype=gs.tc_float,
        )
        reward_multiplier = 2.0
        for key in self.reward_scales.keys():
            self.reward_scales[key] *= self.dt * reward_multiplier

        # Scene setup (unchanged)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="/home/dodolab/tkworkspace/My_deeploco/my_deeploco/urdf/g1_12dof.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )
        self.scene.build(n_envs=num_envs)

        # Joint setup 
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in env_cfg["dof_names"]]
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float
        )

        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)
        
        self.hip_indices = [0, 1, 2, 6, 7, 8]
        self.hip_roll_indices = [1, 7]
        
        # hip
        self.robot.set_dofs_kv([8.0] * len(self.hip_indices), self.hip_indices)
        self.robot.set_dofs_kp([150.0] * len(self.hip_roll_indices), self.hip_roll_indices)
        self.robot.set_dofs_kv([20.0] * len(self.hip_roll_indices), self.hip_roll_indices)
        
        self.knee_indices = [3, 9]
        self.ankle_indices = [4, 5, 10, 11]
        
        # knee
        self.robot.set_dofs_kp([75.0] * len(self.knee_indices), self.knee_indices)
        self.robot.set_dofs_kv([6.0] * len(self.knee_indices), self.knee_indices)
        
        # ankle
        self.robot.set_dofs_kp([65.0] * len(self.ankle_indices), self.ankle_indices)
        self.robot.set_dofs_kv([10.0] * len(self.ankle_indices), self.ankle_indices)
        
        self.ankle_roll_indices = [5, 11]
        
        # ankle roll
        self.robot.set_dofs_kp([85.0] * len(self.ankle_roll_indices), self.ankle_roll_indices)
        self.robot.set_dofs_kv([15.0] * len(self.ankle_roll_indices), self.ankle_roll_indices)

        # Reward functions
        self.reward_functions = {}
        self._register_reward_functions()

        # Episode sums
        self.episode_sums = {key: torch.zeros(num_envs, device=self.device, dtype=gs.tc_float) 
                            for key in self.reward_scales}

        # Initial Buffers 
        self.base_lin_vel = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(num_envs, 1)
        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.goal_pos = torch.zeros((num_envs, 2), device=self.device, dtype=gs.tc_float)  # x, y
        self.footstep_targets = torch.zeros((num_envs, 2, 3), device=self.device, dtype=gs.tc_float)  # [left, right][x, y]
        self.extras = dict()

        # Contact and foot data 
        self.contact_forces = self.robot.get_links_net_contact_force()
        self.left_foot_link = self.robot.get_link(name='left_ankle_roll_link')
        self.right_foot_link = self.robot.get_link(name='right_ankle_roll_link')
        self.left_foot_id_local = self.left_foot_link.idx_local
        self.right_foot_id_local = self.right_foot_link.idx_local
        self.feet_indices = [self.left_foot_id_local, self.right_foot_id_local]
        self.feet_num = len(self.feet_indices)
        self.links_vel = self.robot.get_links_vel()
        self.feet_vel = self.links_vel[:, self.feet_indices, :]
        self.links_pos = self.robot.get_links_pos()
        self.feet_pos = self.links_pos[:, self.feet_indices, :]
        self.links_quat = self.robot.get_links_quat()
        self.feet_quat = self.links_quat[:, self.feet_indices, :]
        self.feet_quat_euler = quat_to_xyz(self.feet_quat)
        # reference for feet quat : [90, 0, 0], [-90, 0, 0]
        self.feet_quat_euler_ref = torch.tensor([[90, 0, 0], [-90, 0, 0]], device=self.device, dtype=gs.tc_float)
        self.feet_quat_euler_ref = self.feet_quat_euler_ref.repeat(self.num_envs, 1, 1)
        
        self.pelvis_link = self.robot.get_link(name='pelvis')
        self.pelvis_mass = self.pelvis_link.get_mass()
        self.pelvis_id_local = self.pelvis_link.idx_local
        self.pelvis_pos = self.links_pos[:, self.pelvis_id_local, :]

        # Initial Phase variables 
        self.phase = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)
        self.phase_left = self.phase.clone()
        self.phase_right = self.phase.clone()
        self.leg_phase = torch.zeros((num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.sin_phase = torch.zeros((num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.cos_phase = torch.zeros((num_envs, 1), device=self.device, dtype=gs.tc_float)
        
        self.original_links_mass = []

        # Termination contacts 
        termination_contact_names = env_cfg["terminate_after_contacts_on"]
        self.termination_contact_indices = []
        for name in termination_contact_names:
            link = self.robot.get_link(name)
            self.termination_contact_indices.append(link.idx_local)
        
        # Add consecutive contact tracking 
        self.consecutive_contact_count = torch.zeros((num_envs, 2), device=self.device, dtype=gs.tc_float)
        
        # Add feet air time tracking 
        self.feet_air_time = torch.zeros((num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.last_contacts = torch.zeros((num_envs, 2), device=self.device, dtype=torch.bool)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        # Initialize goal and footstep targets
        self.initial_dir_to_goal = torch.zeros((num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.initial_dir_to_goal[:, 0] = 1.0  # Default: all forward
        self._resample_goals(torch.arange(num_envs, device=self.device))
        self.base_euler = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        # Initialize step sequence and footstep targets
        # self.step_sequence = [self.plan_step_sequence(idx) for idx in range(self.num_envs)]
        self.current_step_idx = [0 for _ in range(self.num_envs)]
        for idx in range(self.num_envs):
            # Set initial footstep_targets to the first two steps in the sequence
            seq = self.plan_step_sequence(idx)
            if len(seq) >= 2:
                self.footstep_targets[idx, 0, :] = torch.tensor(seq[0][:3], device=self.device)
                self.footstep_targets[idx, 1, :] = torch.tensor(seq[1][:3], device=self.device)
            elif len(seq) == 1:
                self.footstep_targets[idx, 0, :] = torch.tensor(seq[0][:3], device=self.device)
                self.footstep_targets[idx, 1, :] = torch.tensor(seq[0][:3], device=self.device)
            else:
                self.footstep_targets[idx, :, :] = 0.0

        # Add footstep planner parameters with defaults
        self.env_cfg["step_size"] = env_cfg.get("step_size", 0.15)
        self.env_cfg["step_gap"] = env_cfg.get("step_gap", 0.25)
        self.env_cfg["feet_height_target"] = env_cfg.get("feet_height_target", 0.10)
        self.env_cfg["period"] = env_cfg.get("period", 1.1)
        self.env_cfg["swing_duration"] = env_cfg.get("swing_duration", 0.45)
        self.env_cfg["stance_duration"] = env_cfg.get("stance_duration", 0.65)

        self.cfg = {
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "domain_rand_cfg": domain_rand_cfg,
        }

    def _resample_goals(self, envs_idx):
        self.goal_pos[envs_idx, 0] = self.base_pos[envs_idx, 0] + 50.0  # Far ahead
        self.goal_pos[envs_idx, 1] = self.base_pos[envs_idx, 1]  # No lateral offset

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = 0.3  # Forward speed
        self.commands[envs_idx, 1] = 0.0  # No lateral velocity
        self.commands[envs_idx, 2] = 0.0  # No angular velocity

    def plan_step_sequence(self, idx, num_steps=2):
        base_pos = self.base_pos[idx, :2].cpu().numpy()
        goal_pos = self.goal_pos[idx, :2].cpu().numpy()
        # Compute heading toward the goal
        delta = goal_pos - base_pos
        heading = np.arctan2(delta[1], delta[0])
        step_length = float(self.env_cfg["step_size"])
        step_width = float(self.env_cfg["step_gap"])
        feet_height_target = float(self.env_cfg["feet_height_target"])
        sequence = []
        stance = 1  # 1 for left, -1 for right
        # Start at the current base position
        curr_pos = base_pos.copy()
        for n in range(num_steps):
            # Step in the heading direction
            forward = np.array([np.cos(heading), np.sin(heading)])
            lateral = np.array([-np.sin(heading), np.cos(heading)])
            step_offset = step_length * forward + stance * (step_width / 2) * lateral
            curr_pos = curr_pos + step_offset
            sequence.append([curr_pos[0], curr_pos[1], feet_height_target, 0.0])
            stance *= -1  # Alternate stance
        return sequence

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos        
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.contact_forces = self.robot.get_links_net_contact_force()
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.5
        self.consecutive_contact_count = torch.where(
            contact,
            self.consecutive_contact_count + 1,
            torch.zeros_like(self.consecutive_contact_count)
        )

        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.links_vel = self.robot.get_links_vel()
        self.feet_vel = self.links_vel[:, self.feet_indices, :]
        if torch.isnan(self.base_pos).any():
            nan_envs = torch.isnan(self.base_pos).any(dim=1).nonzero(as_tuple=False).flatten()
            self.reset_idx(nan_envs)
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        period = 1.0
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.left_stance = self.phase < 0.5
        self.right_stance = self.phase >= 0.5
        self.left_swing = ~self.left_stance
        self.right_swing = ~self.right_stance
        self.sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        self.cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        # At the end of each gait cycle, update footstep_targets dynamically
        update_footsteps = (self.episode_length_buf % int(period / self.dt) == 0).nonzero(as_tuple=False).flatten()
        for idx in update_footsteps:
            base_pos = self.base_pos[idx, :2].cpu().numpy()
            goal_pos = self.goal_pos[idx, :2].cpu().numpy()
            delta = goal_pos - base_pos
            heading = np.arctan2(delta[1], delta[0])  # Calculate heading towards the goal
            step_length = float(self.env_cfg["step_size"])
            step_width = float(self.env_cfg["step_gap"])
            feet_height_target = float(self.env_cfg["feet_height_target"])
            
            # Compute next two footsteps based on current heading
            for foot in range(2):  # 0: left, 1: right
                stance = 1 if foot == 0 else -1
                forward = np.array([np.cos(heading), np.sin(heading)])  # Forward direction
                step_offset = step_length * forward  # Only forward, no lateral
                step_pos = base_pos + step_offset
                
                # Assign footstep targets
                self.footstep_targets[idx, foot, 0] = torch.tensor(step_pos[0], device=self.device)
                self.footstep_targets[idx, foot, 1] = torch.tensor(step_pos[1], device=self.device)
                self.footstep_targets[idx, foot, 2] = torch.tensor(feet_height_target, device=self.device)

        # Resample commands (optional, less frequent)
        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt)) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)

        # Compute distance to goal for termination and reward
        goal_dist = torch.norm(self.goal_pos - self.base_pos[:, :2], dim=1)
        goal_reached = goal_dist < self.env_cfg.get("goal_reached_threshold", 0.3)

        # Termination
        self.links_pos = self.robot.get_links_pos()
        self.feet_pos = self.links_pos[:, self.feet_indices, :]
        self.pelvis_pos = self.links_pos[:, self.pelvis_id_local, :]
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) & (self.episode_length_buf > 100)
        self.reset_buf |= (self.pelvis_pos[:, 2] < self.env_cfg["termination_if_pelvis_z_less_than"]) & (self.episode_length_buf > 100)
        self.reset_buf |= goal_reached  # Reset on goal reached
        termination_contacts = torch.any(self.contact_forces[:, self.termination_contact_indices, 2] > 60.0, dim=1)
        self.reset_buf |= termination_contacts 

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.extras["goal_reached"] = goal_reached.float()

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Domain randomization (unchanged)
        if torch.any((self.episode_length_buf % int(self.domain_rand_cfg.get('push_interval_s', 10.0)/self.dt)) == 0):
            progress = 1.0
            if self.domain_rand_cfg.get('randomize_friction', False):
                self.randomize_friction(progress)
            if self.domain_rand_cfg.get('randomize_mass', False):
                self.randomize_mass(progress)
            if self.domain_rand_cfg.get('push_robots', False):
                self.push_robots(progress)

        # Compute rewards
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            if name in self.reward_scales:
                raw_rew = reward_func()
                if isinstance(raw_rew, float):
                    raw_rew = torch.ones_like(self.rew_buf) * raw_rew
                clipped_rew = torch.clamp(raw_rew, -25.0, 25.0)
                rew = clipped_rew * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew

        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        feet_pos_rel = self.feet_pos - self.base_pos.unsqueeze(1) # if need 

        # Compute observations
        rel_goal = self.goal_pos - self.base_pos[:, :2]
        rel_goal_3d = torch.cat([rel_goal, torch.zeros_like(rel_goal[:, :1])], dim=1)
        rel_goal_base = transform_by_quat(rel_goal_3d, inv_quat(self.base_quat))[:, :2]
        # Transform footstep targets to root frame for observation
        footstep_targets_root = self.footstep_targets
        # Observation: [base_ang_vel, projected_gravity, commands, dof_pos, dof_vel, actions, sin_phase, cos_phase, rel_goal_base, T1, T2, clock]
        clock = torch.cat([self.sin_phase, self.cos_phase], dim=1)
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
            rel_goal_base * self.obs_scales.get("goal_pos", 0.1),
            footstep_targets_root.view(self.num_envs, -1) * self.obs_scales.get("footstep_targets", 1.0),
            clock
        ], axis=-1)
        self.obs_buf = torch.clip(self.obs_buf, -self.env_cfg["clip_observations"], self.env_cfg["clip_observations"])
        self.last_actions[:] = self.actions[:]

        self.extras["observations"] = {
            "critic": self.obs_buf,
            "privileged": None
        }
        self.last_dof_vel = self.dof_vel.clone()
        self.feet_quat = self.links_quat[:, self.feet_indices, :]
        self.feet_quat_euler = quat_to_xyz(self.feet_quat)
        self.leg_phase[:, 0] = self.phase
        self.leg_phase[:, 1] = (self.phase + 0.5) % 1.0

        if torch.rand(1).item() < 0.01:  # Print occasionally
            print("T1 left x:", footstep_targets_root[:, 0, 0].cpu().numpy())
            print("T1 right x:", footstep_targets_root[:, 1, 0].cpu().numpy())

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def randomize_friction(self, progress=1.0):
        # Calculate progress-based friction range
        base_range = self.domain_rand_cfg['friction_range']
        
        # Start with narrow range, expand to full range as training progresses
        # Use a more conservative range initially
        center = 0.75  # Shifted toward higher friction (was 0.5 * (base_range[0] + base_range[1]))
        half_width = 0.3 * (base_range[1] - base_range[0])  # Reduced variation (was 0.5)
        
        # Apply progressively wider range based on training progress but with slower ramp-up
        progress_scaled = progress * 0.8  # Slow down the progression
        actual_range = [
            max(0.6, center - half_width * progress_scaled),  # Set minimum friction
            min(1.1, center + half_width * progress_scaled)   # Set maximum friction
        ]
        
        # Apply randomization with the calculated range
        self.robot.set_friction_ratio(
            friction_ratio = actual_range[0] + torch.rand(self.num_envs, self.robot.n_links, device=self.device) * 
                            (actual_range[1] - actual_range[0]),
            link_indices = np.arange(0, self.robot.n_links)
        )

    def randomize_mass(self, progress=1.0):
        # Only randomize pelvis mass for simplicity, ignore progress
        added_mass_range = self.domain_rand_cfg['added_mass_range']
        added_mass = float(torch.rand(1).item() * (added_mass_range[1] - added_mass_range[0]) + added_mass_range[0])
        new_mass = max(self.pelvis_mass + added_mass, 0.1)
        self.pelvis_link.set_mass(new_mass)

    def push_robots(self, progress=1.0):
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.domain_rand_cfg['push_interval_s']/self.dt) == 0]
        if len(push_env_ids) == 0:
            return
        
        # Scale push intensity by progress - much more gradually
        max_vel_xy = self.domain_rand_cfg['max_push_vel_xy'] * progress * 0.6  # Reduced by 40%
        max_vel_rp = self.domain_rand_cfg['max_push_vel_rp'] * progress * 0.5   # Reduced by 50%
        
        # Reduce number of pushes early in training - more drastically
        push_probability = min(1.0, 0.2 + 0.8 * progress * progress)  # Square term for slower ramp-up
        
        # Randomly select which environments to push based on progress
        push_mask = torch.rand(len(push_env_ids), device=self.device) < push_probability
        if not torch.any(push_mask):
            return
        
        # Only push selected environments
        push_env_ids = push_env_ids[push_mask]
        
        # Make pushes smaller in magnitude and biased toward forward direction
        # This helps the robot learn to recover from pushes in the intended direction
        new_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        new_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        
        # Generate push velocities with forward bias
        x_vel = gs_rand_float(-max_vel_xy * 0.5, max_vel_xy, (len(push_env_ids),), self.device)  # Forward bias
        y_vel = gs_rand_float(-max_vel_xy * 0.7, max_vel_xy * 0.7, (len(push_env_ids),), self.device)  # Reduced lateral
        z_vel = gs_rand_float(0, max_vel_xy * 0.3, (len(push_env_ids),), self.device)  # Small upward only
        
        new_base_lin_vel[push_env_ids] = torch.stack([x_vel, y_vel, z_vel], dim=1)
        
        # Reduced angular disturbances, especially for roll (x) which affects lateral stability
        roll_vel = gs_rand_float(-max_vel_rp * 0.4, max_vel_rp * 0.4, (len(push_env_ids),), self.device)  # Reduced roll
        pitch_vel = gs_rand_float(-max_vel_rp * 0.7, max_vel_rp * 0.7, (len(push_env_ids),), self.device)  # Less reduction for pitch
        yaw_vel = gs_rand_float(-max_vel_rp * 0.6, max_vel_rp * 0.6, (len(push_env_ids),), self.device)  # Moderate reduction for yaw
        
        new_base_ang_vel[push_env_ids] = torch.stack([roll_vel, pitch_vel, yaw_vel], dim=1)
        
        d_vel_xy = new_base_lin_vel - self.base_lin_vel[:, :3]
        d_vel_rp = new_base_ang_vel - self.base_ang_vel[:, :3]
        d_pos = d_vel_xy * self.dt
        d_pos[:, [2]] = 0
        current_pos = self.robot.get_pos()
        new_pos = current_pos[push_env_ids] + d_pos[push_env_ids]
        self.robot.set_pos(new_pos, zero_velocity=False, envs_idx=push_env_ids)
        current_euler = self.base_euler
        d_euler = d_vel_rp * self.dt
        new_euler = current_euler[push_env_ids] + d_euler[push_env_ids]
        new_quat = xyz_to_quat(new_euler)
        self.robot.set_quat(new_quat, zero_velocity=False, envs_idx=push_env_ids)

    def get_observations(self):
        extras = {
            "observations": {
                "critic": self.obs_buf,
                "privileged": None
            }
        }
        return self.obs_buf, extras
    
    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        self.phase[envs_idx] = gs_rand_float(0.0, 0.2, (len(envs_idx),), self.device)
        self._resample_goals(envs_idx)
        self.consecutive_contact_count[envs_idx] = 0
        self.feet_air_time[envs_idx] = 0.0
        self.last_contacts[envs_idx] = False
        self.last_dof_vel[envs_idx] = 0.0
        self.base_pos[envs_idx, 2] = self.base_init_pos[2] + 0.03

        base_pos = self.base_pos[envs_idx, :2]
        goal_pos = self.goal_pos[envs_idx, :2]
        to_goal = goal_pos - base_pos
        dist_to_goal = torch.norm(to_goal)
        if dist_to_goal > 1e-6:
            dir_to_goal = to_goal / dist_to_goal
        else:
            dir_to_goal = torch.tensor([1.0, 0.0], device=self.device)
        self.initial_dir_to_goal[envs_idx] = dir_to_goal

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        extras = {
            "observations": {
                "critic": self.obs_buf,
                "privileged": None
            }
        }
        return self.obs_buf, extras

    # --------------------- Reward Functions ---------------------
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_alive(self):
        return 1.0

    def _reward_gait_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_gait_swing(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_swing = self.leg_phase[:, i] >= 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_swing)
        return res

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))


    def _get_swing_height_target(self, leg_phase):
        # Sinusoidal height profile: peaks at leg_phase = 0.775 (mid-swing)
        swing_phase = (leg_phase - 0.55) / 0.45  # Normalize to [0, 1] (0.55 to 1.0 is swing)
        swing_phase = torch.clamp(swing_phase, 0.0, 1.0)
        max_height = 0.10  # Peak height (adjustable)
        height = max_height * torch.sin(np.pi * swing_phase)  # Peaks at swing_phase = 0.5
        return height
    
    def _reward_feet_swing_height(self):
        swing_height_reward = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 0.5
        for i in range(self.feet_num):
            is_swing = (self.leg_phase[:, i] >= 0.55) & (~contact[:, i])
            height_target = self._get_swing_height_target(self.leg_phase[:, i])
            height_error = torch.abs(self.feet_pos[:, i, 2] - height_target)
            swing_height_reward += torch.exp(-height_error / 0.05) * is_swing.float()
        return swing_height_reward
    
    """def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 0.5  # Lowered threshold
        pos_error = torch.square(self.feet_pos[:, :, 2] - self.reward_cfg["feet_height_target"]) * ~contact
        return torch.sum(pos_error, dim=(1))"""

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_knee_angle(self):
        target_knee_angle = 0.2  # Slightly bent knee
        knee_error = torch.square(self.dof_pos[:, self.knee_indices] - target_knee_angle)
        return -torch.sum(knee_error, dim=1)
    
    def _reward_feet_angle(self):
        return torch.sum(torch.square(self.feet_quat_euler[:,:,2] - self.feet_quat_euler_ref[:,:,2]), dim=1)

    # New reward functions
    def _reward_footstep_tracking(self):
        footstep_reward = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 0.5
        for i in range(self.feet_num):
            is_swing = (self.leg_phase[:, i] >= 0.55) & (~contact[:, i])
            # Compute error in world frame
            target_world = self.base_pos + self.footstep_targets[:, i, :]
            # Vertical error (height)
            height_error = torch.abs(self.feet_pos[:, i, 2] - target_world[:, 2])
            # Horizontal error (x, y)
            horizontal_error = torch.norm(self.feet_pos[:, i, :2] - target_world[:, :2], dim=1)
            # Combined reward: stronger weight on height
            footstep_reward += (0.7 * torch.exp(-height_error / 0.05) + 
                            0.3 * torch.exp(-horizontal_error / 0.3)) * is_swing.float()
        return footstep_reward

    def _reward_goal_progress(self):
        # Reward progress towards the goal in the global frame
        prev_dist = torch.norm(self.goal_pos - (self.base_pos[:, :2] - self.base_lin_vel[:, :2] * self.dt), dim=1)
        curr_dist = torch.norm(self.goal_pos - self.base_pos[:, :2], dim=1)
        return (prev_dist - curr_dist) * 10.0  # Positive if moving closer to goal

    def _reward_forward_vel(self):
        forward_vel = self.base_lin_vel[:, 0]  # x-direction
        lateral_vel = self.base_lin_vel[:, 1]  # y-direction
        return torch.exp(-torch.square(forward_vel - 0.3)) - torch.square(lateral_vel)

    def _reward_heading_alignment(self):
        # Get current yaw
        base_yaw = quat_to_xyz(self.base_quat)[..., 2]
        # Direction to goal
        to_goal = self.goal_pos - self.base_pos[:, :2]
        goal_yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        heading_err = torch.abs(base_yaw - goal_yaw)
        heading_err = torch.min(2 * np.pi - heading_err, heading_err)
        return torch.pow(0.5 * (torch.cos(heading_err) + 1), 4)


    def _register_reward_functions(self):
        reward_function_names = [
            #"tracking_lin_vel",
            #"tracking_ang_vel",
            "lin_vel_z",
            "action_rate",
            "base_height",
            "alive",
            "gait_contact",
            "gait_swing",
            "contact_no_vel",
            "feet_swing_height",
            "orientation",
            "ang_vel_xy",
            "dof_vel",
            "knee_angle",
            "feet_angle",
            "footstep_tracking",
            "goal_progress",
            "forward_vel",
            "heading_alignment",
        ]
        for name in reward_function_names:
            if hasattr(self, f"_reward_{name}"):
                self.reward_functions[name] = getattr(self, f"_reward_{name}")
            else:
                print(f"Warning: No reward function found for {name}")
        for key in list(self.reward_functions.keys()):
            if key not in self.reward_scales:
                print(f"Adding missing reward scale for {key}")
                self.reward_scales[key] = 0.0
        for key in list(self.reward_functions.keys()):
            if self.reward_scales.get(key, 0.0) == 0.0:
                print(f"Disabling unused reward function: {key}")
                del self.reward_functions[key]

    def _visualize_footstep_targets(self):
        self.scene.clear_debug_objects()
        for env_idx in range(self.num_envs):
            base_pos = self.base_pos[env_idx].cpu().numpy()
            base_quat = self.base_quat[env_idx]
            seq = self.plan_step_sequence(env_idx)
            for i, step in enumerate(seq):
                pos = np.array(step[:3], dtype=np.float32)
                t_world = transform_by_quat(
                    torch.tensor(pos, dtype=gs.tc_float, device=self.device).unsqueeze(0), base_quat.unsqueeze(0)
                )[0].cpu().numpy() + base_pos
                t_world[2] = 0.01
                if i == self.current_step_idx[env_idx]:
                    color = (0.0, 1.0, 0.0, 0.8)
                elif i == self.current_step_idx[env_idx] + 1:
                    color = (0.0, 0.0, 1.0, 0.8)
                else:
                    color = (0.5, 0.5, 0.5, 0.5)
                self.scene.draw_debug_sphere(
                    pos=t_world,
                    radius=0.025,
                    color=color
                )
            # Draw arrows for current footstep targets
            for foot in range(2):
                t_base = self.footstep_targets[env_idx, foot].clone()
                t_base[2] = 0.0
                t_world = transform_by_quat(
                    t_base.unsqueeze(0), base_quat.unsqueeze(0)
                )[0].cpu().numpy() + base_pos
                t_world[2] = 0.0
                forward_vec_base = torch.tensor([0.15, 0.0, 0.0], device=self.device).unsqueeze(0)
                forward_vec_world = transform_by_quat(forward_vec_base, base_quat.unsqueeze(0))[0].cpu().numpy()
                self.scene.draw_debug_arrow(
                    pos=t_world,
                    vec=forward_vec_world,
                    radius=0.01,
                    color=(0.0, 1.0, 0.0, 0.8) if foot == 0 else (0.0, 0.0, 1.0, 0.8)
                )
            # Draw heading arrow at the midpoint between the two current footstep targets
            midpoint = 0.5 * (self.footstep_targets[env_idx, 0] + self.footstep_targets[env_idx, 1])
            midpoint[2] = 0.0
            midpoint_world = transform_by_quat(
                midpoint.unsqueeze(0), base_quat.unsqueeze(0)
            )[0].cpu().numpy() + base_pos
            midpoint_world[2] = 0.02
            goal_pos = self.goal_pos[env_idx].cpu().numpy()
            base_xy = self.base_pos[env_idx, :2].cpu().numpy()
            heading = np.arctan2(goal_pos[1] - base_xy[1], goal_pos[0] - base_xy[0])
            arrow_vec = np.array([np.cos(heading), np.sin(heading), 0.0]) * 0.5
            self.scene.draw_debug_arrow(
                pos=midpoint_world,
                vec=arrow_vec,
                radius=0.015,
                color=(1.0, 0.0, 0.0, 0.8)
            )

    
    
