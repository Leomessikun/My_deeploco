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
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        # Use env_cfg["num_actions"] if defined, otherwise infer from number of dof_names.
        self.num_actions = env_cfg.get("num_actions", len(env_cfg["dof_names"]))
        self.num_commands = command_cfg["num_commands"]

        # Set simulation parameters 
        self.simulate_action_latency = env_cfg.get("simulation_action_latency", False) #(Sim2Sim mode)
        self.dt = 0.02  # 50Hz control frequency
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.domain_rand_cfg = domain_rand_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        
        # Multiply reward scales by dt 
        for key in self.reward_scales.keys():
            self.reward_scales[key] *= self.dt

        # Create scene 
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
        # Add plane
        self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        
        # Add robot – note the URDF file has been changed for g1
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

        # Map joint names to motor indices and set default positions
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in env_cfg["dof_names"]]
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float
        )

        # Set PD gains (position control) 
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # Prepare reward functions and episode statistics
        self.reward_functions = {name: getattr(self, "_reward_" + name) for name in self.reward_scales.keys()}
        self.episode_sums = {key: torch.zeros(num_envs, device=self.device, dtype=gs.tc_float) for key in self.reward_scales}

        # Initialize buffers
        self.base_lin_vel = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(num_envs, 1)
        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["heading"]],
            device=self.device, dtype=gs.tc_float,
        )
        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.extras = dict()

        # Modified physics and foot contact parameters 
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
        
        # Reference feet euler 
        self.feet_quat_euler_ref = torch.tensor([[90, 0, 0], [-90, 0, 0]], device=self.device, dtype=gs.tc_float)
        self.feet_quat_euler_ref = self.feet_quat_euler_ref.repeat(num_envs, 1, 1)
        
        # Introduce randomization in the phase offset for each environment to encourage diversity
        period = 1.0  # Increased period for more natural gait timing
        self.phase_offset = torch.rand(self.num_envs, device=self.device) * period  # Random offset per env
        self.phase = ((self.episode_length_buf * self.dt + self.phase_offset) % period) / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.phase_offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        self.sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        self.cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        
        self.pelvis_link = self.robot.get_link(name='pelvis')
        self.pelvis_mass = self.pelvis_link.get_mass()
        self.pelvis_id_local = self.pelvis_link.idx_local
        self.links_pos = self.robot.get_links_pos()  # Critical!
        self.pelvis_pos = self.links_pos[:, self.pelvis_id_local, :]
        self.original_links_mass = []
        self.command_ranges = command_cfg["ranges"]  # Store ranges
        self.heading_command = command_cfg["heading_command"]  # Store flag
        self.curriculum = command_cfg.get("curriculum", False)
        self.curriculum_steps = command_cfg.get("curriculum_steps", 10000)
        self.counter = 0

        self.knee_indices = [3, 8]

        # Design of curriculum
        self.curriculum_steps = self.command_cfg.get("curriculum_steps", 10000)
        self.counter = 0  # Tracks the number of steps

        termination_contact_names = env_cfg["terminate_after_contacts_on"]
        self.termination_contact_indices = []
        for name in termination_contact_names:
            link = self.robot.get_link(name)
            self.termination_contact_indices.append(link.idx_local)

    def _resample_commands(self, envs_idx):
        # Check if we're in the initial phase
        in_initial_phase = self.counter < self.command_cfg.get("initial_steps", 0)
        
        if in_initial_phase:
            # Use simpler initial ranges
            ranges = self.command_cfg["initial_ranges"]
            c = 1.0  # No scaling in initial phase
        else:
            # Use regular curriculum
            ranges = self.command_cfg["ranges"]
            if self.command_cfg.get("curriculum", False):
                # Calculate curriculum progress after initial phase
                progress = max(0, self.counter - self.command_cfg.get("initial_steps", 0))
                c = min(self.command_cfg["max_curriculum"], progress / self.command_cfg["curriculum_steps"])
            else:
                c = self.command_cfg["max_curriculum"]

        # Sample commands with appropriate ranges
        self.commands[envs_idx, 0] = gs_rand_float(
            ranges["lin_vel_x"][0],
            c * ranges["lin_vel_x"][1],
            (len(envs_idx),),
            self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            c * ranges["lin_vel_y"][0],
            c * ranges["lin_vel_y"][1],
            (len(envs_idx),),
            self.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            c * ranges["ang_vel_yaw"][0],
            c * ranges["ang_vel_yaw"][1],
            (len(envs_idx),),
            self.device
        )
        self.commands[envs_idx, 3] = gs_rand_float(
            c * ranges["heading"][0],
            c * ranges["heading"][1],
            (len(envs_idx),),
            self.device
        )
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos        
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # Update buffer
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()

        # if it gets NaN values in self.robot.get_*()
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
        
        # Resample commands
        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt)) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)
        
        # Check termination and reset
        self.links_pos = self.robot.get_links_pos()  # Critical!
        self.pelvis_pos = self.links_pos[:, self.pelvis_id_local, :]
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.pelvis_pos[:, 2] < self.env_cfg["termination_if_pelvis_z_less_than"]
        
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        # Domain randomization (triggered only at specified intervals)
        if self.domain_rand_cfg.get('randomize_friction', False) and self.counter % int(self.domain_rand_cfg['push_interval_s']/self.dt) == 0:
            self.randomize_friction()
        if self.domain_rand_cfg.get('randomize_mass', False) and self.counter % int(self.domain_rand_cfg['push_interval_s']/self.dt) == 0:
            self.randomize_mass()
        if self.domain_rand_cfg.get('push_robots', False):
            self.push_robots()
        
        # Compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        contact = (self.contact_forces[:, self.feet_indices, 2] > 5.0).float()  # Match threshold with rewards
        # Compute observations
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
            self.sin_phase,
            self.cos_phase,
            contact,  # Adds 2 dimensions (left, right foot contact)
        ], axis=-1)
        
        self.obs_buf = torch.clip(self.obs_buf, -self.env_cfg["clip_observations"], self.env_cfg["clip_observations"])
        self.last_actions[:] = self.actions[:]
        self.counter += 1

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def randomize_friction(self):
        friction_range = self.domain_rand_cfg['friction_range']
        self.robot.set_friction_ratio(
            friction_ratio = friction_range[0] + torch.rand(self.num_envs, self.robot.n_links, device=self.device) * (friction_range[1] - friction_range[0]),
            link_indices = np.arange(0, self.robot.n_links)
        )
        self.plane.set_friction_ratio(
            friction_ratio = friction_range[0] + torch.rand(self.num_envs, self.plane.n_links, device=self.device) * (friction_range[1] - friction_range[0]),
            link_indices = np.arange(0, self.plane.n_links)
        ) 

    def randomize_mass(self):
        added_mass_range = self.domain_rand_cfg.get('added_mass_range', [0.0, 0.0])
        added_mass = float(torch.rand(1).item() * (added_mass_range[1] - added_mass_range[0]) + added_mass_range[0])
        new_mass = max(self.pelvis_mass + added_mass, 0.1)
        self.pelvis_link.set_mass(new_mass)

    def push_robots(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.domain_rand_cfg['push_interval_s']/self.dt) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel_xy = self.domain_rand_cfg['max_push_vel_xy']
        max_vel_rp = self.domain_rand_cfg['max_push_vel_rp']
        new_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        new_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        new_base_lin_vel[push_env_ids] = gs_rand_float(-max_vel_xy, max_vel_xy, (len(push_env_ids), 3), self.device)
        new_base_ang_vel[push_env_ids] = gs_rand_float(-max_vel_rp, max_vel_rp, (len(push_env_ids), 3), self.device)
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
        return self.obs_buf
    
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
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
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
        self._resample_commands(envs_idx)

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
        return -torch.square(self.base_lin_vel[:, 2])
    
    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    def _reward_alive(self):
        return 1.0
    
    def _reward_gait_contact(self):
        # More gradual transition between stance and swing
        stance_factor = 0.5 * (1 + torch.cos(2 * np.pi * self.leg_phase))  # Smooth sinusoidal transition
        is_stance = stance_factor > 0.5
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        
        # Reward for correct contact timing
        contact_reward = torch.sum(~(contact ^ is_stance), dim=1).float()
        
        # Allow brief double support but penalize extended double support
        both_contact = torch.all(contact, dim=1).float()
        double_support_penalty = torch.where(both_contact > 0, 
                                           0.5 * torch.ones_like(both_contact),  # Reduced penalty
                                           torch.zeros_like(both_contact))
        
        return contact_reward - double_support_penalty

    def _reward_gait_swing(self):
        # More gradual transition between stance and swing
        swing_factor = 0.5 * (1 - torch.cos(2 * np.pi * self.leg_phase))  # Smooth sinusoidal transition
        is_swing = swing_factor > 0.5
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        
        # Additional reward for proper swing height during swing phase
        swing_height = self.feet_pos[:, :, 2]
        target_swing_height = torch.where(is_swing, 
                                        torch.ones_like(swing_height) * self.reward_cfg["feet_height_target"],
                                        torch.zeros_like(swing_height))
        height_reward = torch.exp(-torch.square(swing_height - target_swing_height) / 0.1)
        
        # Combine swing timing and height rewards
        swing_reward = torch.sum(~(contact ^ is_swing), dim=1).float()
        return swing_reward + 0.5 * torch.sum(height_reward * is_swing.float(), dim=1)

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = torch.norm(self.feet_vel, dim=2) * contact.float()
        return -torch.sum(contact_feet_vel, dim=1)
    
    def _reward_feet_swing_height(self):
        is_swing = self.leg_phase >= 0.5  # Changed threshold to 0.5 for smoother transition
        target_height = torch.where(
            is_swing,
            torch.ones_like(self.feet_pos[:, :, 2]) * self.reward_cfg["feet_height_target"],
            torch.zeros_like(self.feet_pos[:, :, 2])
        )
        height_error = torch.square(self.feet_pos[:, :, 2] - target_height)
        height_reward = torch.exp(-height_error / 0.1)
        
        # Add penalty for feet being too close together during swing
        feet_distance = torch.abs(self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1])  # Y-distance between feet
        distance_penalty = torch.exp(-torch.square(feet_distance - 0.3) / 0.1)  # Encourage ~30cm between feet
        
        return torch.sum(height_reward, dim=1) + distance_penalty
    
    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_knee_angle(self):
        knee_indices = [self.env_cfg["dof_names"].index(name) for name in ["left_knee_joint", "right_knee_joint"]]
        target_angle = 0.3  # A slight bend, adjust as needed
        return -torch.sum(torch.square(self.dof_pos[:, knee_indices] - target_angle), dim=1)
    
    def _reward_feet_angle(self):
        return torch.sum(torch.square(self.feet_quat_euler[:, :, 2] - self.feet_quat_euler_ref[:, :, 2]), dim=1)
    
    def close(self):
        print("Environment closed.")
