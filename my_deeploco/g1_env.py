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
        
        # Match the command_scale tensor to the actual commands used (3 instead of 4)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device, dtype=gs.tc_float,
        )
        
        # Increase all reward scales for more significant per-timestep rewards
        reward_multiplier = 2.0  # Reduced from 5.0 to create more balanced rewards
        
        # Multiply reward scales by dt and the multiplier
        for key in self.reward_scales.keys():
            self.reward_scales[key] *= self.dt * reward_multiplier

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

        # Set generic PD gains for all joints
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)  # kp = 60.0 for all
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)  # kd = 1.5 for all

        # Override kd for hips to increase damping - especially for hip roll which affects lateral stability
        self.hip_indices = [0, 1, 2, 6, 7, 8]  # Hip pitch, roll, yaw for both legs
        self.hip_roll_indices = [1, 7]  # Left and right hip roll indices
        
        # Increased damping for better stability
        self.robot.set_dofs_kv([8.0] * len(self.hip_indices), self.hip_indices)  # Increased from 5.0
        
        # Higher stiffness and damping for hip roll joints to resist lateral movements
        self.robot.set_dofs_kp([150.0] * len(self.hip_roll_indices), self.hip_roll_indices)  # Increased from 100.0
        self.robot.set_dofs_kv([20.0] * len(self.hip_roll_indices), self.hip_roll_indices)   # Increased from 10.0
        
        # Increase kd for knees and ankles for better stability
        self.knee_indices = [3, 9]  # Just knees
        self.ankle_indices = [4, 5, 10, 11]  # Just ankles
        
        # Different gains for knees vs ankles - increased for stability
        self.robot.set_dofs_kp([75.0] * len(self.knee_indices), self.knee_indices)  # Added explicit kp for knees
        self.robot.set_dofs_kv([6.0] * len(self.knee_indices), self.knee_indices)  # Increased from 4.0
        self.robot.set_dofs_kp([65.0] * len(self.ankle_indices), self.ankle_indices)  # Added explicit kp for ankles
        self.robot.set_dofs_kv([10.0] * len(self.ankle_indices), self.ankle_indices)  # Increased from 6.0
        
        # Higher stiffness for ankle roll to resist unwanted inversion/eversion
        self.ankle_roll_indices = [5, 11]  # Left and right ankle roll
        self.robot.set_dofs_kp([85.0] * len(self.ankle_roll_indices), self.ankle_roll_indices)  # Increased from 75.0
        self.robot.set_dofs_kv([15.0] * len(self.ankle_roll_indices), self.ankle_roll_indices)  # Increased from 12.0
        
        # Register reward functions
        self.reward_functions = {}
        self._register_reward_functions()
        
        # Initialize episode sums after reward functions and scales are set up
        self.episode_sums = {key: torch.zeros(num_envs, device=self.device, dtype=gs.tc_float) 
                            for key in self.reward_scales}

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
        # Reference feet eulaers: [90, 0, 0], [-90, 0, 0]
        self.feet_quat_euler_ref = torch.tensor([[90, 0, 0], [-90, 0, 0]], device=self.device, dtype=gs.tc_float)
        self.feet_quat_euler_ref = self.feet_quat_euler_ref.repeat(self.num_envs, 1, 1)
        
        self.pelvis_link = self.robot.get_link(name='pelvis')
        self.pelvis_mass = self.pelvis_link.get_mass()
        self.pelvis_id_local = self.pelvis_link.idx_local
        self.links_pos = self.robot.get_links_pos()  # Critical!
        self.pelvis_pos = self.links_pos[:, self.pelvis_id_local, :]
        
        # Initialize phase variables
        self.phase = torch.zeros(num_envs, device=self.device, dtype=gs.tc_float)
        self.phase_left = self.phase.clone()
        self.phase_right = self.phase.clone()
        self.leg_phase = torch.zeros((num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.sin_phase = torch.zeros((num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.cos_phase = torch.zeros((num_envs, 1), device=self.device, dtype=gs.tc_float)
        
        self.original_links_mass = []
        
        termination_contact_names = env_cfg["terminate_after_contacts_on"]
        self.termination_contact_indices = []
        for name in termination_contact_names:
            link = self.robot.get_link(name)
            self.termination_contact_indices.append(link.idx_local)

        # Add consecutive contact tracking
        self.consecutive_contact_count = torch.zeros((num_envs, 2), device=self.device, dtype=gs.tc_float)

        # Track feet air time for the feet_air_time reward
        self.feet_air_time = torch.zeros((num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.last_contacts = torch.zeros((num_envs, 2), device=self.device, dtype=torch.bool)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        
    def _resample_commands(self, envs_idx):
        # Use the new flat range keys from command_cfg
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos        
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # Update contact forces after the simulation step
        self.contact_forces = self.robot.get_links_net_contact_force()
        
        # Update consecutive contact count
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.5
        self.consecutive_contact_count = torch.where(
            contact,
            self.consecutive_contact_count + 1,
            torch.zeros_like(self.consecutive_contact_count)
        )

        # Update buffer
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.links_vel = self.robot.get_links_vel()
        self.feet_vel = self.links_vel[:, self.feet_indices, :]

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

        # Change period to match realistic canine gait
        period = 1.0
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.left_stance = self.phase < 0.5
        self.right_stance = self.phase >= 0.5
        self.left_swing = ~self.left_stance
        self.right_swing = ~self.right_stance
        self.sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        self.cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        
        # Resample commands
        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt)) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)
        
        # Check termination and reset
        self.links_pos = self.robot.get_links_pos()  # Critical!
        self.feet_pos = self.links_pos[:, self.feet_indices, :]
        self.pelvis_pos = self.links_pos[:, self.pelvis_id_local, :]
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) & (self.episode_length_buf > 100)
        self.reset_buf |= (self.pelvis_pos[:, 2] < self.env_cfg["termination_if_pelvis_z_less_than"]) & (self.episode_length_buf > 100)
        termination_contacts = torch.any(self.contact_forces[:, self.termination_contact_indices, 2] > 60.0, dim=1)
        self.reset_buf |= termination_contacts 

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        # Domain randomization with progressive curriculum
        if torch.any((self.episode_length_buf % int(self.domain_rand_cfg.get('push_interval_s', 10.0)/self.dt)) == 0):
            # Calculate training progress for progressive randomization
            progress = 1.0  # Always use full randomization now
            if self.domain_rand_cfg.get('randomize_friction', False):
                self.randomize_friction(progress)
            if self.domain_rand_cfg.get('randomize_mass', False):
                self.randomize_mass(progress)
            if self.domain_rand_cfg.get('push_robots', False):
                self.push_robots(progress)
        
        # Compute reward with clipping
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            if name in self.reward_scales:  # Only compute rewards with non-zero scales
                raw_rew = reward_func()
                
                # Convert to tensor if it's a float (handles _reward_alive which returns 1.0)
                if isinstance(raw_rew, float):
                    raw_rew = torch.ones_like(self.rew_buf) * raw_rew
                    
                # Add clipping to prevent extreme values before scaling
                # Increased clipping threshold to match reward multiplier
                clipped_rew = torch.clamp(raw_rew, -25.0, 25.0)  
                
                rew = clipped_rew * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew
        
        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()  # Back to 1N
        
        # Compute relative foot positions
        feet_pos_rel = self.feet_pos - self.base_pos.unsqueeze(1)

        # Compute observations - simplify to match official implementation
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,  # Using first 3 values of commands (x, y, yaw)
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
            self.sin_phase,
            self.cos_phase
        ], axis=-1)
        
        self.obs_buf = torch.clip(self.obs_buf, -self.env_cfg["clip_observations"], self.env_cfg["clip_observations"])
        self.last_actions[:] = self.actions[:]

        # Update self.extras
        self.extras["observations"] = {
            "critic": self.obs_buf,
            "privileged": None
        }

        # Update last_dof_vel for acceleration calculation
        self.last_dof_vel = self.dof_vel.clone()

        # Update feet_quat and feet_quat_euler
        self.feet_quat = self.links_quat[:, self.feet_indices, :]
        self.feet_quat_euler = quat_to_xyz(self.feet_quat)

        # Update leg_phase
        self.leg_phase[:, 0] = self.phase
        self.leg_phase[:, 1] = (self.phase + 0.5) % 1.0

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
            
        # Reset joint positions and velocities
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        
        # Reset base pose and velocities
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
        
        # Reset action buffer
        self.last_actions[envs_idx] = 0.0
        
        # Reset episode stats and track them
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
            
        # Reset phase
        self.phase[envs_idx] = gs_rand_float(0.0, 0.2, (len(envs_idx),), self.device)  # Start early in phase
        
        # Resample commands
        self._resample_commands(envs_idx)

        # Reset consecutive contact tracking
        self.consecutive_contact_count[envs_idx] = 0

        # Reset feet air time
        self.feet_air_time[envs_idx] = 0.0
        self.last_contacts[envs_idx] = False
        self.last_dof_vel[envs_idx] = 0.0

        # Start with higher base height and less noise
        self.base_pos[envs_idx, 2] = self.base_init_pos[2] + 0.03  # 0.12 -> 0.03

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

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 0.5  # Lowered threshold
        pos_error = torch.square(self.feet_pos[:, :, 2] - self.reward_cfg["feet_height_target"]) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_knee_angle(self):
        return torch.sum(torch.square(self.dof_pos[:, self.knee_indices]), dim=1)

    def _reward_feet_angle(self):
        return torch.sum(torch.square(self.feet_quat_euler[:,:,2] - self.feet_quat_euler_ref[:,:,2]), dim=1)

    def _register_reward_functions(self):
        # Register only the reward functions we actually use
        reward_function_names = [
            "tracking_lin_vel",
            "tracking_ang_vel",
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
        ]
        
        # First register all functions that exist
        for name in reward_function_names:
            if hasattr(self, f"_reward_{name}"):
                self.reward_functions[name] = getattr(self, f"_reward_{name}")
            else:
                print(f"Warning: No reward function found for {name}")
        
        # Then ensure all needed scales exist
        for key in list(self.reward_functions.keys()):
            if key not in self.reward_scales:
                print(f"Adding missing reward scale for {key}")
                self.reward_scales[key] = 0.0  # Disable by default
        
        # And remove any functions for disabled rewards to save computation
        for key in list(self.reward_functions.keys()):
            if self.reward_scales.get(key, 0.0) == 0.0:
                print(f"Disabling unused reward function: {key}")
                del self.reward_functions[key]  # Actually remove disabled functions

    def close(self):
        print("Environment closed.")