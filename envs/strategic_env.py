"""
Complete Strategic Push-Grasp Environment for ME5418 Project.
Implements full state space, action space, and reward function from proposal.

State Space: 28 + 22N + N² dimensions
- Robot: 22D (7 joints + velocities + EE pose + gripper)
- Objects: N×21D (pose, velocities, 8D shape descriptors)
- Spatial: Distance matrix (N×N) + occlusion masks (N)
- Environment: 6D (goal, bounds, count)

Action Space: 4D continuous [-1, 1]
- α_skill: Pick-and-place (>0) vs Push (≤0)
- α_x, α_y: Contact point in object's local frame
- α_θ: Push direction

Reward: 8 components from proposal
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
import pybullet_data
import pybullet as p

# Import utility functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.object_util import (
    compute_shape_descriptors,
    compute_pairwise_distance_matrix,
    compute_occlusion_masks,
    analyze_scene_occlusions,
    update_object_colors,
    select_target_heuristic,
    get_safe_spawn_position,
    check_object_in_goal
)

from utils.robot_util import (
    execute_pick_and_place,
    execute_push
)

from utils.physics_util import (
    check_workspace_violation,
    check_collision_with_table,
    check_object_collision
)


class StrategicPushAndGraspEnv(gym.Env):
    """
    Strategic Push-Grasp Environment with complete state space and reward function.
    
    Features:
    - 7-DOF Panda robot for manipulation
    - 5-10 objects (cubes and spheres) per episode
    - Dynamic occlusion detection and color coding
    - Heuristic target selection (nearest to goal)
    - Complete observable state space
    - All 8 reward components from ME5418 proposal
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    # Color constants for object states
    COLOR_GREEN = [0.1, 0.8, 0.1, 1.0]   # Graspable cubes
    COLOR_YELLOW = [0.8, 0.8, 0.1, 1.0]  # Spheres (harder to grasp)
    COLOR_RED = [0.8, 0.1, 0.1, 1.0]     # Occluded objects
    
    # Environment parameters
    MAX_OBJECTS = 10
    WORKSPACE_BOUNDS = (-0.3, 0.3, -0.3, 0.3)  # x_min, x_max, y_min, y_max
    OCCLUSION_THRESHOLD = 0.05  # 5cm
    MIN_OBJECT_SEPARATION = 0.08  # 8cm

    def __init__(self, render_mode: str = "human"):
        """
        Initialize the Strategic Push-Grasp Environment.
        
        Args:
            render_mode: "human" for GUI visualization, None for headless
        """
        print("=" * 70)
        print("Initializing Strategic Push-Grasp Environment (ME5418)")
        print("=" * 70)
        
        # 1. Create simulation and robot
        self.sim = PyBullet(render_mode=render_mode)
        self.robot = Panda(
            self.sim,
            block_gripper=False,  # Enable gripper control
            base_position=np.array([0.4, -0.3, 0.0])
        )
        print("✓ Robot: 7-DOF Panda at position [0.4, -0.3, 0.0]")
        
        # 2. Action space: A = (α_skill, α_x, α_y, α_θ) ∈ [-1, 1]^4
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        print("✓ Action space: 4D continuous [-1, 1]")
        
        # 3. Scene parameters
        self.objects = {}  # Dictionary: name -> properties
        self.goal_pos = np.array([-0.2, -0.2])
        self.goal_size = 0.1
        self.table_bounds = np.array([0.4, 0.4])
        self.scene_setup = False
        print(f"✓ Goal zone: center={self.goal_pos}, size={self.goal_size}m")
        
        # 4. Episode tracking
        self.current_target = None
        self.collected_objects = set()
        self.episode_step = 0
        self.max_episode_steps = 100
        
        # 5. Reward tracking variables
        self.previous_joint_positions = None
        self.previous_object_distances = {}
        self.previous_occlusion_states = {}
        self.action_was_successful = True
        
        # 6. Define observation space with FIXED dimensions based on MAX_OBJECTS
        # Formula: 28 + MAX_OBJECTS*21 + MAX_OBJECTS² + MAX_OBJECTS
        obs_dim = 28 + self.MAX_OBJECTS * 21 + self.MAX_OBJECTS**2 + self.MAX_OBJECTS
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        print(f"✓ Observation space: {obs_dim}D (28 + {self.MAX_OBJECTS}×21 + {self.MAX_OBJECTS}² + {self.MAX_OBJECTS})")
        print("=" * 70 + "\n")

    def _draw_goal_square(self):
        """Draw green square outline for goal zone."""
        half_size = self.goal_size / 2
        color = [0, 1, 0]  # Green
        z = 0.001
        cx, cy = self.goal_pos
        
        corners = [
            [cx - half_size, cy - half_size, z],
            [cx + half_size, cy - half_size, z],
            [cx + half_size, cy + half_size, z],
            [cx - half_size, cy + half_size, z]
        ]
        
        client = self.sim.physics_client
        for i in range(4):
            client.addUserDebugLine(
                corners[i],
                corners[(i + 1) % 4],
                color,
                lineWidth=3
            )

    # def _get_robot_state(self) -> Dict[str, np.ndarray]:
    #     """
    #     Get complete robot state (22D).
        
    #     Returns:
    #         Dictionary with:
    #         - joint_positions: 7D
    #         - joint_velocities: 7D
    #         - ee_position: 3D
    #         - ee_orientation: 4D (quaternion)
    #         - gripper_width: 1D
    #     """
    #     robot_obs = self.robot.get_obs()
        
    #     return {
    #         'joint_positions': robot_obs[:7],
    #         'joint_velocities': robot_obs[7:14],
    #         'ee_position': self.robot.get_ee_position(),
    #         'ee_orientation': self.robot.get_ee_orientation(),
    #         'gripper_width': np.array([np.mean(robot_obs[14:16])])
    #     }

    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()

        # --- 取 EE 位姿（位置仍用现有接口；姿态增加兜底） ---
        ee_pos = self.robot.get_ee_position()

        try:
            ee_quat = self.robot.get_ee_orientation()  # 先尝试官方接口
        except AttributeError:
            # 兜底：直接用 PyBullet 取链接姿态四元数
            panda_uid = self.sim._bodies_idx.get("panda")  # 你的工程里 body 名一般就是 'panda'
            ee_link = getattr(self.robot, "ee_link", None) # Panda 类通常有 ee_link 索引
            if panda_uid is not None and ee_link is not None:
                ee_quat = np.array(
                    p.getLinkState(panda_uid, ee_link, computeForwardKinematics=1)[1],
                    dtype=np.float32
                )
            else:
                # 最保守的兜底：单位四元数
                ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)

        return {
            'joint_positions':    robot_obs[:7],
            'joint_velocities':   robot_obs[7:14],
            'ee_position':        ee_pos,
            'ee_orientation':     ee_quat,
            'gripper_width':      np.array([np.mean(robot_obs[14:16])], dtype=np.float32),
        }



    # def _get_robot_state(self) -> Dict[str, np.ndarray]:
    #     # 1) 机器人原始观测
    #     robot_obs = self.robot.get_obs()
    #     print("[DBG] robot_obs type:", type(robot_obs), "shape:", getattr(robot_obs, "shape", None))
    #     print("[DBG] robot_obs[:7] (q):", robot_obs[:7])
    #     print("[DBG] robot_obs[7:14] (dq):", robot_obs[7:14])
    #     # 14:16 常用作两指位姿，如果版本不同可能偏移，请留意输出
    #     print("[DBG] robot_obs[14:16] (gripper fingers raw):", robot_obs[14:16])

    #     # 2) 末端位置
    #     try:
    #         ee_pos = self.robot.get_ee_position()
    #         print("[DBG] ee_position from Panda:", ee_pos)
    #     except Exception as e:
    #         print("[DBG] get_ee_position() failed:", repr(e))
    #         ee_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    #     # 3) 末端姿态（首选 Panda 接口，失败则 PyBullet 兜底）
    #     try:
    #         ee_quat = self.robot.get_ee_orientation()
    #         print("[DBG] ee_orientation from Panda:", ee_quat)
    #     except AttributeError as e:
    #         print("[DBG] get_ee_orientation() missing on Panda:", repr(e))
    #         # 3a) 打印 sim 的 body 索引表，确认机器人 body 名称
    #         try:
    #             print("[DBG] sim._bodies_idx keys:", list(getattr(self.sim, "_bodies_idx", {}).keys()))
    #         except Exception as e2:
    #             print("[DBG] cannot read sim._bodies_idx keys:", repr(e2))

    #         panda_uid = getattr(self.sim, "_bodies_idx", {}).get("panda")
    #         print("[DBG] panda_uid (by key 'panda'):", panda_uid)

    #         ee_link = getattr(self.robot, "ee_link", None)
    #         print("[DBG] robot.ee_link:", ee_link)

    #         try:
    #             if panda_uid is not None and ee_link is not None:
    #                 ls = p.getLinkState(panda_uid, ee_link, computeForwardKinematics=1)
    #                 # ls[1] 是世界系四元数 (x,y,z,w)
    #                 print("[DBG] getLinkState return len:", len(ls) if ls else None)
    #                 print("[DBG] getLinkState[0] world pos:", ls[0] if ls else None)
    #                 print("[DBG] getLinkState[1] world quat:", ls[1] if ls else None)
    #                 ee_quat = np.array(ls[1], dtype=np.float32)
    #             else:
    #                 print("[DBG] panda_uid/ee_link is None, fallback to unit quat.")
    #                 ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)
    #         except Exception as e3:
    #             print("[DBG] getLinkState() failed:", repr(e3))
    #             ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)
    #     except Exception as e:
    #         print("[DBG] unexpected exception in get_ee_orientation():", repr(e))
    #         ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)

    #     # 4) 夹爪宽度（两指平均）
    #     try:
    #         gripper_raw = robot_obs[14:16]
    #         gripper_width = float(np.mean(gripper_raw))
    #     except Exception as e:
    #         print("[DBG] gripper width parse failed:", repr(e))
    #         gripper_width = float("nan")
    #     print("[DBG] gripper_width (mean of 14:16):", gripper_width)

    #     # 5) 组装返回
    #     jp = np.asarray(robot_obs[:7], dtype=np.float32)
    #     jv = np.asarray(robot_obs[7:14], dtype=np.float32)
    #     ee_pos = np.asarray(ee_pos, dtype=np.float32)
    #     ee_quat = np.asarray(ee_quat, dtype=np.float32)
    #     gw = np.array([gripper_width], dtype=np.float32)

    #     print("[DBG] final shapes -> jp:", jp.shape, "jv:", jv.shape,
    #         "ee_pos:", ee_pos.shape, "ee_quat:", ee_quat.shape, "gw:", gw.shape)

    #     return {
    #         "joint_positions": jp,
    #         "joint_velocities": jv,
    #         "ee_position": ee_pos,
    #         "ee_orientation": ee_quat,
    #         "gripper_width": gw,
    #     }



    def _get_object_states(self) -> Dict[str, np.ndarray]:
        """
        Get complete object states (N×21D).
        
        Per object: position (3) + orientation (4) + velocity (3) + 
                   angular_velocity (3) + shape_descriptor (8) = 21D
        
        Returns:
            Dictionary with arrays for all objects
        """
        object_names = sorted(self.objects.keys())
        N = len(object_names)
        
        if N == 0:
            return {
                'positions': np.array([]).reshape(0, 3),
                'orientations': np.array([]).reshape(0, 4),
                'velocities': np.array([]).reshape(0, 3),
                'angular_velocities': np.array([]).reshape(0, 3),
                'shape_descriptors': np.array([]).reshape(0, 8)
            }
        
        positions = np.zeros((N, 3), dtype=np.float32)
        orientations = np.zeros((N, 4), dtype=np.float32)
        velocities = np.zeros((N, 3), dtype=np.float32)
        angular_velocities = np.zeros((N, 3), dtype=np.float32)
        shape_descriptors = np.zeros((N, 8), dtype=np.float32)
        
        for i, name in enumerate(object_names):
            positions[i] = self.sim.get_base_position(name)
            orientations[i] = self.sim.get_base_orientation(name)
            velocities[i] = self.sim.get_base_velocity(name)
            angular_velocities[i] = self.sim.get_base_angular_velocity(name)
            shape_descriptors[i] = self.objects[name]['shape_descriptor']
        
        return {
            'positions': positions,
            'orientations': orientations,
            'velocities': velocities,
            'angular_velocities': angular_velocities,
            'shape_descriptors': shape_descriptors
        }

    def _get_spatial_relationships(self) -> Dict[str, np.ndarray]:
        """
        Compute spatial relationships between objects.
        
        Returns:
            Dictionary with:
            - distance_matrix: N×N pairwise distances
            - occlusion_mask: N binary flags
        """
        distance_matrix = compute_pairwise_distance_matrix(self.sim, self.objects)
        occlusion_mask = compute_occlusion_masks(
            self.sim,
            self.objects,
            threshold=self.OCCLUSION_THRESHOLD
        )
        
        return {
            'distance_matrix': distance_matrix,
            'occlusion_mask': occlusion_mask
        }

    def _get_obs(self) -> np.ndarray:
        """
        Construct complete observation vector.
        
        Structure (28 + 22N + N²):
        - Robot state: 22D
        - Environment info: 6D
        - Object states: N×21D
        - Distance matrix: N×N
        - Occlusion mask: N
        
        Returns:
            Flattened observation array
        """
        # 1. Robot state (22D)
        robot = self._get_robot_state()
        robot_vector = np.concatenate([
            robot['joint_positions'],      # 7D
            robot['joint_velocities'],     # 7D
            robot['ee_position'],          # 3D
            robot['ee_orientation'],       # 4D
            robot['gripper_width']         # 1D
        ])
        
        # 2. Object states (N×21D)
        objects = self._get_object_states()
        N = len(self.objects)
        
        # Create padded arrays
        positions_padded = np.zeros((self.MAX_OBJECTS, 3), dtype=np.float32)
        orientations_padded = np.zeros((self.MAX_OBJECTS, 4), dtype=np.float32)
        velocities_padded = np.zeros((self.MAX_OBJECTS, 3), dtype=np.float32)
        angular_velocities_padded = np.zeros((self.MAX_OBJECTS, 3), dtype=np.float32)
        shape_descriptors_padded = np.zeros((self.MAX_OBJECTS, 8), dtype=np.float32)
        
        if N > 0:
            positions_padded[:N] = objects['positions']
            orientations_padded[:N] = objects['orientations']
            velocities_padded[:N] = objects['velocities']
            angular_velocities_padded[:N] = objects['angular_velocities']
            shape_descriptors_padded[:N] = objects['shape_descriptors']
        
        object_vector = np.concatenate([
            positions_padded.flatten(),           # MAX_OBJECTS×3
            orientations_padded.flatten(),        # MAX_OBJECTS×4
            velocities_padded.flatten(),          # MAX_OBJECTS×3
            angular_velocities_padded.flatten(),  # MAX_OBJECTS×3
            shape_descriptors_padded.flatten()    # MAX_OBJECTS×8
        ])
        
        # 3. Spatial relationships
        spatial = self._get_spatial_relationships()
    
        # Pad distance matrix to MAX_OBJECTS × MAX_OBJECTS
        distance_matrix_padded = np.zeros((self.MAX_OBJECTS, self.MAX_OBJECTS), dtype=np.float32)
        if N > 0:
            distance_matrix_padded[:N, :N] = spatial['distance_matrix']
        
        # Pad occlusion mask to MAX_OBJECTS
        occlusion_mask_padded = np.zeros(self.MAX_OBJECTS, dtype=np.float32)
        if N > 0:
            occlusion_mask_padded[:N] = spatial['occlusion_mask'].astype(np.float32)
        
        # 4. Environment information (6D)
        env_info = np.array([
            self.goal_pos[0],      # Goal X
            self.goal_pos[1],      # Goal Y
            self.goal_size,        # Goal size
            self.table_bounds[0],  # Table X bound
            self.table_bounds[1],  # Table Y bound
            float(N)               # Actual number of objects
        ], dtype=np.float32)
        
        # 5. Concatenate all components
        obs = np.concatenate([
            robot_vector,                      # 22D
            env_info,                          # 6D
            object_vector,                     # MAX_OBJECTS×21D
            distance_matrix_padded.flatten(),  # MAX_OBJECTS²
            occlusion_mask_padded              # MAX_OBJECTS
        ])
        
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment to initial state for new episode.
        
        Returns:
            observation: Complete state vector
            info: Empty dictionary
        """
        super().reset(seed=seed)
        
        with self.sim.no_rendering():
            # Setup static scene (table)
            if not self.scene_setup:
                self.sim.create_box(
                    body_name="table",
                    half_extents=np.array([0.4, 0.4, 0.01]),
                    mass=0.0,
                    position=np.array([0, 0, -0.01]),
                    rgba_color=np.array([0.8, 0.8, 0.8, 1])
                )
                self.scene_setup = True
            
            # Remove old objects
            for body_name in self.objects:
                body_id = self.sim._bodies_idx.get(body_name)
                if body_id is not None:
                    self.sim.physics_client.removeBody(body_id)
            
            self.objects.clear()
            self.collected_objects.clear()
            
            # Reset tracking variables
            self.previous_joint_positions = None
            self.previous_object_distances.clear()
            self.previous_occlusion_states.clear()
            
            # Draw goal area
            self._draw_goal_square()
            
            # Reset robot
            self.robot.reset()
            
            # Spawn random objects (5-10)
            num_objects = np.random.randint(5, self.MAX_OBJECTS + 1)
            object_types = ["cube", "sphere"]
            
            for i in range(num_objects):
                body_name = f"object_{i}"
                object_type = np.random.choice(object_types)
                
                spawn_pos = get_safe_spawn_position(
                    self.sim,
                    list(self.objects.keys()),
                    self.goal_pos,
                    self.goal_size,
                    min_separation=self.MIN_OBJECT_SEPARATION,
                    workspace_bounds=self.WORKSPACE_BOUNDS
                )
                
                default_color = self.COLOR_GREEN if object_type == "cube" else self.COLOR_YELLOW
                
                # Create object and compute shape descriptor
                if object_type == "cube":
                    half_extents = np.array([0.02, 0.02, 0.02])
                    self.sim.create_box(
                        body_name=body_name,
                        half_extents=half_extents,
                        mass=1.0,
                        position=spawn_pos,
                        rgba_color=default_color
                    )
                    shape_desc = compute_shape_descriptors("cube", half_extents=half_extents)
                
                elif object_type == "sphere":
                    radius = 0.02
                    self.sim.create_sphere(
                        body_name=body_name,
                        radius=radius,
                        mass=1.0,
                        position=spawn_pos,
                        rgba_color=default_color
                    )
                    shape_desc = compute_shape_descriptors("sphere", radius=radius)
                
                # Store object metadata with shape descriptor
                self.objects[body_name] = {
                    "type": object_type,
                    "is_occluded": False,
                    "shape_descriptor": shape_desc
                }
                
                # Initialize distance tracking
                obj_pos = np.array(self.sim.get_base_position(body_name)[:2])
                self.previous_object_distances[body_name] = np.linalg.norm(obj_pos - self.goal_pos)
            
            # Analyze scene and update colors
            analyze_scene_occlusions(self.sim, self.objects, self.OCCLUSION_THRESHOLD)
            
            # Store initial occlusion states
            for name in self.objects:
                self.previous_occlusion_states[name] = self.objects[name]["is_occluded"]
            
            update_object_colors(
                self.sim, self.objects,
                self.COLOR_GREEN, self.COLOR_YELLOW, self.COLOR_RED
            )
            
            # Select initial target using heuristic
            self.current_target = select_target_heuristic(
                self.sim, self.objects, self.goal_pos, self.collected_objects
            )
        
        # Initialize joint positions for trajectory penalty
        self.previous_joint_positions = self.robot.get_obs()[:7]
        self.episode_step = 0
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """
        Execute one environment step with complete action and reward.
        
        Args:
            action: 4D array [α_skill, α_x, α_y, α_θ]
        
        Returns:
            observation: Complete state vector
            reward: Scalar reward (sum of 8 components)
            terminated: True if episode complete
            truncated: True if max steps reached
            info: Dictionary with reward breakdown
        """
        # Parse action: A = (α_skill, α_x, α_y, α_θ)
        a_skill, alpha_x, alpha_y, alpha_theta = action
        
        # Determine action type based on α_skill
        action_type = "grasp" if a_skill > 0 else "push"
        
        # Execute action primitive as specified in proposal
        if self.current_target is None:
            # No target selected - action automatically fails
            self.action_was_successful = False
            print(f"⚠ No target selected - action failed")
        
        elif action_type == "grasp":
            # Execute pick-and-place primitive
            print(f"\nExecuting PICK-AND-PLACE on {self.current_target}")
            print(f"  Contact: α_x={alpha_x:.2f}, α_y={alpha_y:.2f}")
            self.action_was_successful = execute_pick_and_place(
                self.sim, self.robot, self.current_target,
                alpha_x, alpha_y, self.goal_pos
            )
        
        elif action_type == "push":
            # Execute push primitive
            print(f"\nExecuting PUSH on {self.current_target}")
            print(f"  Contact: α_x={alpha_x:.2f}, α_y={alpha_y:.2f}")
            print(f"  Direction: α_θ={alpha_theta:.2f} (angle={alpha_theta*np.pi:.2f} rad)")
            self.action_was_successful = execute_push(
                self.sim, self.robot, self.current_target,
                alpha_x, alpha_y, alpha_theta
            )
        
        # Update scene analysis
        analyze_scene_occlusions(self.sim, self.objects, self.OCCLUSION_THRESHOLD)
        update_object_colors(
            self.sim, self.objects,
            self.COLOR_GREEN, self.COLOR_YELLOW, self.COLOR_RED
        )
        
        # Get current joint positions for trajectory penalty
        current_joints = self.robot.get_obs()[:7]
        
        # Compute COMPLETE reward (8 components)
        reward, reward_info = self._compute_complete_reward(
            action_type=action_type,
            current_joints=current_joints
        )
        
        # Update tracking variables
        self.previous_joint_positions = current_joints
        self.episode_step += 1
        
        # Update target selection (heuristic)
        self.current_target = select_target_heuristic(
            self.sim, self.objects, self.goal_pos, self.collected_objects
        )
        
        # Get new observation
        obs = self._get_obs()
        
        # Check termination conditions
        terminated = len(self.collected_objects) >= len(self.objects) * 0.95
        truncated = self.episode_step >= self.max_episode_steps
        
        # Prepare info dictionary
        info = {
            "is_success": terminated,
            "collected": len(self.collected_objects),
            "total": len(self.objects),
            "reward_breakdown": reward_info,
            "action_type": action_type,
            "episode_step": self.episode_step
        }
        
        return obs, float(reward), bool(terminated), truncated, info

    def _compute_complete_reward(self, action_type: str,
                                 current_joints: np.ndarray) -> Tuple[float, dict]:
        """
        Compute complete reward with all 8 components from proposal.
        
        Components:
        1. +5: Correctly placing one object
        2. +25: More than 95% objects collected
        3. +0.5: Successful push reduces distance/occlusion
        4. -3: Failed grasp/push attempt
        5. -10: Object leaves workspace
        6. -5: Collision with fixed obstacles
        7. -0.1: Each action (step penalty)
        8. -0.01/rad: Trajectory length penalty
        
        Returns:
            total_reward: Sum of all components
            reward_info: Dictionary with breakdown
        """
        reward_info = {}
        
        # ========== Component 1: Object placement (+5) ==========
        placement_reward = 0.0
        if self.current_target and self.current_target in self.objects:
            obj_pos = self.sim.get_base_position(self.current_target)
            if check_object_in_goal(obj_pos, self.goal_pos, self.goal_size):
                if self.current_target not in self.collected_objects:
                    placement_reward = 5.0
                    self.collected_objects.add(self.current_target)
                    print(f"✓ Object {self.current_target} placed in goal! (+5)")
        reward_info['placement'] = placement_reward
        
        # ========== Component 2: Task completion (+25) ==========
        completion_reward = 0.0
        completion_threshold = len(self.objects) * 0.95
        if len(self.collected_objects) >= completion_threshold:
            completion_reward = 25.0
            print(f"✓ Task completed! {len(self.collected_objects)}/{len(self.objects)} objects (+25)")
        reward_info['completion'] = completion_reward
        
        # ========== Component 3: Successful push (+0.5) ==========
        push_success_reward = 0.0
        if action_type == "push" and self.current_target and self.action_was_successful:
            # Check if push moved object closer to goal
            obj_pos = np.array(self.sim.get_base_position(self.current_target)[:2])
            current_dist = np.linalg.norm(obj_pos - self.goal_pos)
            previous_dist = self.previous_object_distances.get(self.current_target, current_dist)
            
            distance_reduced = previous_dist - current_dist > 0.01  # Moved >1cm
            
            # Check if push cleared occlusion
            was_occluded = self.previous_occlusion_states.get(self.current_target, False)
            now_occluded = self.objects[self.current_target]["is_occluded"]
            occlusion_cleared = was_occluded and not now_occluded
            
            if distance_reduced or occlusion_cleared:
                push_success_reward = 0.5
                if distance_reduced:
                    print(f"✓ Push moved object closer ({previous_dist:.3f}→{current_dist:.3f}m) (+0.5)")
                if occlusion_cleared:
                    print(f"✓ Push cleared occlusion (+0.5)")
            
            # Update distance tracking
            self.previous_object_distances[self.current_target] = current_dist
        
        # Update occlusion tracking for all objects
        for name in self.objects:
            self.previous_occlusion_states[name] = self.objects[name]["is_occluded"]
        
        reward_info['push_success'] = push_success_reward
        
        # ========== Component 4: Failed action (-3) ==========
        failure_penalty = 0.0
        if not self.action_was_successful:
            failure_penalty = -3.0
            print(f"✗ Action failed! ({action_type}) (-3)")
        reward_info['failure'] = failure_penalty
        
        # ========== Component 5: Workspace violation (-10) ==========
        workspace_penalty = 0.0
        for obj_name in list(self.objects.keys()):
            if check_workspace_violation(
                self.sim, obj_name,
                self.WORKSPACE_BOUNDS,
                z_min=-0.05
            ):
                workspace_penalty = -10.0
                print(f"✗ Object {obj_name} left workspace! (-10)")
                # Remove from tracking
                if obj_name in self.objects:
                    del self.objects[obj_name]
                break
        reward_info['workspace_violation'] = workspace_penalty
        
        # ========== Component 6: Collision penalty (-5) ==========
        collision_penalty = 0.0
        
        # Check collision with table
        if check_collision_with_table(self.sim, 'panda', 'table'):
            collision_penalty = -5.0
            print(f"✗ Robot collided with table! (-5)")
        
        # Check collision with other objects
        if collision_penalty == 0.0:
            for obj_name in self.objects:
                if obj_name != self.current_target:
                    if check_object_collision(self.sim, 'panda', obj_name):
                        collision_penalty = -5.0
                        print(f"✗ Robot collided with {obj_name}! (-5)")
                        break
        
        reward_info['collision'] = collision_penalty
        
        # ========== Component 7: Step penalty (-0.1) ==========
        step_penalty = -0.1
        reward_info['step'] = step_penalty
        
        # ========== Component 8: Trajectory penalty (-0.01/rad) ==========
        trajectory_penalty = 0.0
        if self.previous_joint_positions is not None:
            # Sum absolute joint displacements across all 7 DOF
            joint_displacement = np.abs(current_joints - self.previous_joint_positions)
            total_movement = np.sum(joint_displacement)  # Radians
            
            # Penalty: -0.01 per radian moved
            trajectory_penalty = -0.01 * total_movement
            
            if total_movement > 0.1:
                print(f"Trajectory: {total_movement:.3f} rad → {trajectory_penalty:.4f}")
        reward_info['trajectory'] = trajectory_penalty
        
        # ========== TOTAL REWARD ==========
        total_reward = sum(reward_info.values())
        
        return total_reward, reward_info

    def close(self):
        """Clean up environment resources."""
        self.sim.close()
        print("\nEnvironment closed.")

