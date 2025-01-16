import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np

 # Highlevel Policy
class HighLevelPolicy:
        def __init__(self, num_envs, state_dim, action_dim, hidden_dim=256, num_layers=2, device="cuda"):
            """
            High-level controller for generating intermediate goals.

            Args:
                num_envs (int): Number of parallel environments.
                state_dim (int): Dimension of the state space.
                action_dim (int): Dimension of the action space (goals).
                hidden_dim (int): Number of hidden units in each layer.
                num_layers (int): Number of hidden layers.
                device (str): Device for computation (e.g., "cuda" or "cpu").
            """

            self.device = torch.device(device)
            self.state_dim = state_dim
            self.action_dim = action_dim

            # define the policy network
            layers = []
            layers.append(nn.Linear(state_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, action_dim))
            self.policy_net = nn.Sequential(*layers).to(self.device)

        def act(self, state):
            """
            Generate high-level actions (intermediate goals) based on state.

            Args: 
                state(np.ndarray or torch.Tensor): Current state of the environment.

            Returns:
                np.ndarray: High-level actions(goals).
            """

            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
            else:
                state = state.to(self.device)

            with torch.no_grad():
                action = self.policy_net(state)

            return action.cpu().numpy()
        
# LowLevelController:
class LowLevelController:
        def __init__(self, num_envs, state_dim, action_dim, goal_dim, hidden_dim=256, num_layers=2, device="cuda"):
            """
            Low-level controller for generating fine-grained actions.

            Args:
                num_envs (int): Number of parallel environments.
                state_dim (int): Dimension of the state space.
                action_dim (int): Dimension of the action space (goals).
                goal_dim(int): Dimension of the goal space.
                hidden_dim (int): Number of hidden units in each layer.
                num_layers (int): Number of hidden layers.
                device (str): Device for computation (e.g., "cuda" or "cpu").
            """
            
            self.num_envs = num_envs
            self.device = torch.device(device)
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.goal_dim = goal_dim
            
            # define the policy network 
            layers = []
            layers.append(nn.Linear(state_dim + goal_dim, hidden_dim)) # input layer
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim)) # hidden layer 
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, action_dim)) # output layer
            self.policy_net = nn.Sequential(*layers).to(self.device)

        def act(self, state, goal):
            """
            Generate fine-grained actions to achieve the goal.

            Args: 
                state(np.ndarray or torch.Tensor): Current state of the environment.
                goal(np.ndarray or torch.Tensor): High-level goal.

            Returns:
                np.ndarray: Low-level actions.
            """

            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
            else:
                state = state.to(self.device)

            if not isinstance(goal, torch.Tensor):
                goal = torch.tensor(goal, device=self.device, dtype=torch.float32)
            else:
                goal = goal.to(self.device)

            # combine state and goal
            input_data = torch.cat([state, goal], dim=-1)

            with torch.no_grad():
                action = self.policy_net(input_data)

            return action.cpu().numpy()

def gs_rand_float(lower, upper, shape, device):
    """
    generate random numbers in the range [lower, uppper) using pytorch.

    Args: 
        lower(float): Lower bound of the range.
        upper(float): Upper bound of the range.
        shape(tuple): Shape of the output tensor.
        device(torch.device): Device for the output tensor.

    Returns:
        torch.Tensor: Random numbers in the range [lower, upper).
    """
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class G1DeeplocoEnv:
    def __init__ (self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda" ):
        """
        Initialize the G1 Deeploco environment.

        Args: 
            num_envs(int): Number of parallel environments
            env_cfg(dict): Configuration for the environment(e.g. episode length, control parameters).
            obs_cfg(dict): Configuration for the observation space.
            reward_config(dict): Configuration for reward  functions.
            command_cfg(dict): Configuration for command sampling.
            show_viewer(bool): Whether to show the simulation viewer.
            device(str): Device for tensor computation(e.g., "cuda" or "cpu").
        """

        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_command = command_cfg["num_commands"]

        self.dt = 0.02 # control frequency(50 Hz)
        self.max_episode_length = math.ceil(env_cfg ["episode_length_s"] / self.dt)
        
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.show_viewer = show_viewer

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        
        self.torques = torch.zeros((num_envs, self.num_actions), device=device)
        self.actions = torch.zeros((num_envs, self.num_actions), device=device)

        # create scene
        self.scene = gs.Scene(
            sim_options= gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options= gs.options.ViewerOptions(
                max_FPS= int(0.5 / self.dt),
                camera_pos= (2.0, 0.0, 2.5),
                camera_lookat= (0.0, 0.0, 0.5),
                camera_fov= 40,
            ),
            vis_options= gs.options.VisOptions(n_rendered_envs=1),
            rigid_options= gs.options.RigidOptions(
                dt= self.dt,
                constraint_solver= gs.constraint_solver.Newton,
                enable_collision= True,
                enable_joint_limit= True,
            ),
            show_viewer= show_viewer,
        )

        # add plane 
        self.scene.add_entity(gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True))

        # add G1 robot
        self.base_init_pos= torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat= torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_int_quat= inv_quat(self.base_init_quat)
        self.robot= self.scene.add_entity(
            gs.morphs.URDF(
                file="/home/kuntao/Desktop/My_deeploco/my_deeploco/urdf/g1_29dof.urdf",
                pos= self.base_init_pos.cpu().numpy(),
                quat= self.base_init_quat.cpu().numpy(),
            ),
        )

        # build scene
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs= [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]]* self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]]* self.num_actions, self.motor_dofs)

        # prepare reward functions
        self.reward_functions, self.episode_sum = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_"+ name)
            self.episode_sum[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self._initialize_buffers()

        # initialize controllers
        state_dim = self.num_obs
        goal_dim = 6 # example: desired COM velocity, torso orientation
        action_dim = self.num_actions # dimension of the action space

        # Debugging: Print the values of state_dim, action_dim, and goal_dim
        print(f"state_dim: {state_dim}, action_dim: {action_dim}, goal_dim: {goal_dim}")

        self.high_level_controller = HighLevelPolicy(num_envs, state_dim, goal_dim, device=device)
        self.low_level_controller = LowLevelController(
            num_envs=num_envs, 
            state_dim=state_dim, 
            action_dim=action_dim, 
            goal_dim=goal_dim, 
            device=device
        )

        """self.command_cfg = {
            "lin_vel_x_range" : [0.5, 1.5], # desired forward velocity (vx) in m/s
            "lin_vel_y_range" : [-0.2, 0.2], # desired lateral velocity (vy) in m/s
            "ang_vel_yaw_range": [-0.5, 0.5], # desired yaw velocity (wz) in rad/s
            "gait_mode_range": [0, 2], # gait modes : 0 = walk, 1 = run, 2 = turn
            "locomotion_param_range": [0.1, 0.3], # additional parameters (e.g. step height in meters)
        } """ 

    def _resample_commands(self, envs_idx):
        """resample commands for specified environments.
        commands include desired linear velocities(vx, vy), angular velocity(wz), and gait modes.

        Args:
            envs_idx(torch.Tensor): Indices of envirnments to resample commands for.
        """

        # resample linear velocities (vx, vy) in the robot's local frame
        self.commands[envs_idx, 0] = gs_rand_float(
            # vx (forward velocity)
            self.command_cfg["lin_vel_x_range"][0],
            self.command_cfg["lin_vel_x_range"][1],
            (len(envs_idx),),
            self.device,
        )

        self.commands[envs_idx, 1] = gs_rand_float(
            # vy (lateral velocity)
            self.command_cfg["lin_vel_x_range"][0],
            self.command_cfg["lin_vel_x_range"][1],
            (len(envs_idx),),
            self.device,
        )

        self.commands[envs_idx, 2] = gs_rand_float(
            # wz (yaw velocity)
            self.command_cfg["lin_vel_x_range"][0],
            self.command_cfg["lin_vel_x_range"][1],
            (len(envs_idx),),
            self.device,
        )

        # resample gait modes(e.g., walk, run, turn)
        if "gait_mode_range" in self.command_cfg:
            gait_modes = torch.randint(
                self.command_cfg["gait_mode_range"][0],
                self.command_cfg["gait_mode_range"][1] + 1, # inclusive range
                (len(envs_idx),),
                device = self.device
            )
            self.command[envs_idx, 3] = gait_modes # assign gait modes to command buffer 

        # resample additional locomotion parameters (e.g., step height, stride length)
        if "locomotion_param_range" in self.command_cfg:
            self.command[envs_idx, 4:] = gs_rand_float(
                self.command_cfg["locomotion_param_range"][0],
                self.command_cfg["locomotion_param_range"][1],
                (len(envs_idx), self.num_commands -4), # adjust based on the number of additional parameters
                self.device,
            )

    def _initialize_buffers(self):
        """initialize tensors for observations, rewards, and other buffers."""
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        self.rest_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        self.episode_length_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)
        self.commands = torch.zeros((self.num_envs, self.num_command), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device = self.device,
            dtype = gs.tc_float,
        )
        #self.num_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict() # extra information for logging

    def step(self, action):
        """
        Advance the simulation by one step.

        Args:
            actions (torch.Tensor): Actions for all environments.

        Returns:
            obs_buf (torch:Tensor): Obsercations after the step.
            None: Placeholder for into dictionary.
            rew_buf(torch.Tensor): Rewards for step.
            reset_buf(torch.Tensor): Reset flags for environments.
            extra(dict): Additional info for debugging or logging.
        """

        # generate high-level goals
        state = self.obs_buf # current state 
        goals = self.high_level_controller.act(state)

        # generate low-level actions
        actions = self.low_level_controller.act(state, goals)

        # apply actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        
        # compute torques using PD control
        self.torques = self.compute_torques(actions)

        # set DOF position using genesis API
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        """update buffers"""
        # incement the episode length buffer for all environments by 1, tracking how many steps eacn environment has taken in the current episode.
        self.episode_length_buf += 1 

        # updates the position of the robot's base (typically the root or torso of the robot) in the world frame
        self.base_pos[:] = self.robot.get_pos()

        # updates the orientation of the robot's base in the world frame as quaternion.
        self.base_quat[:] = self.robot.get_quat()

        # convers the base quaternion into euler angles(roll, pitch, yaw) for easier interpolation and calculations.
        """
        transform_quat_by_quat:combines the current base quaternion,
        (self.base_quat) with the inverse of the initial base quaternion,(self.inv_base_int_quat) 
        this transforms the orientation relative to the initial state
        quat_to_xyz: converts the resulting quaternion into euler angles(roll, pitch, yaw)
        """
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat)*self.inv_base_int_quat, self.base_quat)
        )

        # computes the inverse (conjugate) of the current base quaternion, this is used to transform vectors from the world frame into the robot's local frame.
        inv_base_quat = inv_quat(self.base_quat)

        # transforms the robot's base linear velocity (in the world frame ) into the robot's local frame using the inverse base quaternion. This allows the linear velocity to be expressed relative to the robot's orientation.
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)

        # transforms the robot's angular velocity (in the world frame) into the local frame, useful for understanding the rotational dynamics relative to the robot's orientation.
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        
        # projects the global gravity vector (typically [0,0,-1] in the world frame) into the robot's local frame, helps the controller understand the effect of the gravity relative to the robot's current orientation.
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        """
        determines which environments within a parallelized simulation need to have their commands resampled
        
        (self.episode_length_buf % int (self.env_cfg["resampling_time_s"] / self.dt) == 0)
        identifies environments where the current simulation step is a multiple of the resampling interval

        key components: 
            self.episode_length_buf: A buffer that tracks the number of steps each environment has take in the current episode.
            self.env_cfg["resampling_time_s"]: The duration(in seconds) after which commands should be resampled.
            self.dt: the time step of the simulation
            self.env_cfg["resampling_time_s"] / self.dt: Converts the resampling time from seconds into the corresponding number of simulation steps.
            self.episode_length_buf % ... == 0: Checks if the current step count for an environment is a multiple of the resampling interval, indicating it's time to resample.
        """
        envs_idx = (
            (self.episode_length_buf % int (self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False) # as_tuple = False, ensures the result is returned as a 2D tensor(with each row containing and index), rather than a tuple. 
            .flatten() 
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        """
        self.reset_buf = self.episode_length_buf > self.max_episode_length : reset environment where the episode has exceeded the maximum allowed length.
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]  : Adds a reset based on the robot's pitch angle.
        ... : adds a reset condition based on the robot's roll angle.

        |= : for result , environment where the pitch/roll exceeds the threshold are added to the reset buffer using the biwise OR assignment

        """
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        """
        time_out_idx = (self.episode_length_buf > self.max_episode_length). nonzero(as_tuple=False).flatten() : identifies which environments have exceeded the maximum episode length(timed out)
        
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float) : initializes a tensor to track timeouts for each environment.
        
        self.extras["time_outs"][time_out_idx] = 1.0 : marks environments that timed out by setting their corresponding indices in self.extras["time_outs"] to 1.0.
        
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten()) : resets the environments that need to be reset.
        """
        time_out_idx = (self.episode_length_buf > self.max_episode_length). nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward 
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"], #3
                self.projected_gravity, #3
                self.commands * self.commands_scale, #3 (linear x, linear y, angular yaw), adjust if you include additional commands, like desired torso height or heading.
                (self.dof_vel - self.default_dof_pos) * self.obs_scales["dof_pos"], #29 
                self.dof_vel * self.obs_scales["dof_vel"], #29 
                self.actions, #29
            ],
            axis = -1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # for clearance
        """
         # Convert rewards and done to scalars for Stable-Baselines3
        reward = float(np.mean(self.rew_buf.cpu().numpy()))  # Mean reward across environments
        done = self.reset_buf.any().item()  # Whether any environment has terminated
        info = self.extras  # Additional info

        """

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras
    
    def reset_idx(self, envs_idx):
        """reset specified environments."""
        if len(envs_idx) == 0:
            return
        
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base position adn orientation
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat
        self.robot.set_pos(self.base_pos[envs_idx], envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], envs_idx)

        # reset buffers
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 0
        self.rew_buf[envs_idx] = 0.0
        for key in self.episode_sum.keys():
            self.episode_sums[key][envs_idx] = 0.0

        # resample commands for reset environments
        self._resample_commands(envs_idx)

    def compute_torques(self, actions):
        """
        compute torques using PD control.

        Args:
            actions (torch.Tensor): Actions for all environments.

        Returns:
            torque (torch.Tensor): Computed torques for all joints.
        """

        # compute desired joint position based on action 
        desired_joint_pos = self.default_dof_pos + actions + self.env_cfg["action_scale"]

        # compute torques using PD control
        torques = self.env_cfg["kp"] * (desired_joint_pos - self.dof_pos) - self.env_cfg["kd"] + self.dof_vel

        return torques

        
    # placeholder : sample linear velocity (x, y) and angular velocity (yaw)

    def _reward_tracking_lin_vel(self):
        """reward for tracking linear velocity commands."""
        lin_vel_error = torch.sum(torch.square(self.commands[:, : 2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """reward for tracking angular velocity commands."""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_base_height(self):
        """reward for maintaining a desired base height."""
        desired_height = self.env_cfg.get("desired_base_height", 0.5) # default desired height
        height_error = torch.square(self.base_pos[:, 2] - desired_height)
        return torch.exp(-height_error / self.reward_cfg["height_sigma"])
    
    def _reward_foot_clearance(self):
        """reward for lifting feet during swing phases."""
        foot_clearance = self.foot_position[:, :, 2] # z-coordinate of foot positions
        desired_clearance = self.env_cfg.get("desired_foot_clearance", 0.1) # default desired clearance 

        clearance_error = torch.sum(torch.sqaure(foot_clearance - desired_clearance), dim=1)
        return torch.exp(-clearance_error / self.reward_cfg["clearance_sigma"])
    
    def _reward_joint_limits(self):
        """penalize joint position close to their limits."""
        joint_pos = self.dof_pos
        joint_limits_lower = torch.tensor(self.env_cfg["joint_limits"]["lower"], device=self.device)
        joint_limits_upper = torch.tensor(self.env_cfg["joint_limits"]["upper"], device=self.device)

        limit_violation = torch.sum(
            torch.maximum(joint_pos - joint_limits_upper, torch.tensor(0.0, device=self.device)) +
            torch.maximum(joint_limits_lower - joint_pos, torch.tensor(0.0, device=self.device)),
            dim=1,
        )

        return -limit_violation
    
    def _reward_torque(self):
        """penalize high torque usage."""
        torque = torch.sum(torch.square(self.torques), dim=1)
        return -torque
    
    def _reward_contact_consistency(self):
        """reward for consistent foot contact patterns."""
        contact_states = self.foot_contacts # binary tensor indicating foot contact
        desired_contact_states = self.desired_contact_states # desired contact pattern
        contact_error = torch.sum(torch.square(contact_states - desired_contact_states), dim=1)
        return torch.exp(-contact_error / self.reward_cfg["contact_sigma"])

    def _reward_gait_mode(self):
        """reward for matching the desired gait mode."""
        gait_mode_error = torch.abs(self.commands[:, 3] - self.current_gait_mode)
        return torch.exp(-gait_mode_error / self.reward_cfg["tracking_sigma"])

    def _reward_orientation(self):
        """reward for mainting a stable torso orientation."""
        pitch_roll_error = torch.sum(torch.square(self.base_euler[:, :2]), dim= 1)
        return torch.exp(-pitch_roll_error / self.reward_cfg["orientation_sigma"])

    def _reward_energy(self):
        """penalize energy consumption."""
        power = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        return -power

    def _reward_action_smoothness(self):
        """penalize jerky actions."""
        action_diff = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return -action_diff
    
    def _reward_action_rate(self):
        """penalize high action rates (jerky actions)."""
        action_diff = torch.sum(torch.sqaure(self.actions - self.last_actions), dim=1)
        return -action_diff
    
    def compute_reward(self): 
        """compute the total reward."""
        rewards = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(),
            "tracking_ang_vel": self._reward_tracking_ang_vel(),
            "gait_mode": self._reward_gait_mode(),
            "orientation": self._reward_orientation(),
            "energy": self._reward_energy(),
            "action_smoothness": self._reward_action_smoothness(),
            "base_height": self._reward_base_height(),
            "foot_clearance": self._reward_foot_clearance(),
            "joint_limits": self._reward_joint_limits(),
            "torque": self._reward_torque(),
            "contact_consistency": self._reward_contact_consistency(),
            "action_rate": self._reward_action_rate,
        }

        total_reward = torch.zero_like(rewards["tracking_lin_vel"])
        for name, reward in rewards.items():
            total_reward += self.reward_scales[name] * reward

        return total_reward

    def close(self):
        """clean up the environment."""
        self.scene.close()


    
