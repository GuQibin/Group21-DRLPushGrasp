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
    
    This environment implements the ME5418 project for learning push-grasp coordination
    in cluttered tabletop scenarios using hierarchical reinforcement learning.
    
    Features:
    - 7-DOF Panda robot for manipulation
    - 5-10 objects (cubes and spheres) per episode
    - Dynamic occlusion detection and color coding
    - Heuristic target selection (nearest to goal)
    - Complete observable state space (no partial observability)
    - All 8 reward components from ME5418 proposal
    
    Key Design Decisions:
    - Uses FIXED observation space with padding to handle variable object counts
    - Separates high-level action selection (push/grasp) from low-level control
    - Implements complete state observability for simplified learning
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    # ========================================================================
    # COLOR CONSTANTS - Visual feedback for object states
    # ========================================================================
    COLOR_GREEN = [0.1, 0.8, 0.1, 1.0]   # Graspable cubes (easy targets)
    COLOR_YELLOW = [0.8, 0.8, 0.1, 1.0]  # Spheres (harder to grasp, prefer push)
    COLOR_RED = [0.8, 0.1, 0.1, 1.0]     # Occluded objects (need clearing first)
    
    # ========================================================================
    # ENVIRONMENT PARAMETERS
    # ========================================================================
    MAX_OBJECTS = 10 # Maximum objects for padding observation space
    WORKSPACE_BOUNDS = (-0.3, 0.3, -0.3, 0.3)  # (x_min, x_max, y_min, y_max) in meters
    OCCLUSION_THRESHOLD = 0.05  # 5cm - objects closer than this may occlude each other
    MIN_OBJECT_SEPARATION = 0.08  # 8cm - minimum distance between spawned objects

    def __init__(self, render_mode: str = "human"):
        """
        Initialize the Strategic Push-Grasp Environment.
        
        Sets up:
        1. PyBullet physics simulation
        2. Panda robot with 7-DOF arm + 2-finger gripper
        3. Action space: 4D continuous control
        4. Observation space: Fixed-size padded vector
        5. Goal zone and workspace boundaries
        
        Args:
            render_mode: "human" for GUI visualization, None for headless training
        """
        print("=" * 70)
        print("Initializing Strategic Push-Grasp Environment (ME5418)")
        print("=" * 70)
        
        # ====================================================================
        # 1. CREATE SIMULATION AND ROBOT
        # ====================================================================
        # PyBullet simulation with optional GUI rendering
        
        self.sim = PyBullet(render_mode=render_mode)
        
        # Panda robot: 7-DOF arm + parallel-jaw gripper
        # Base positioned at [0.4, -0.3, 0.0] to reach table center
        # block_gripper=False allows continuous gripper control
        self.robot = Panda(
            self.sim,
            block_gripper=False, # Enable gripper actuation
            base_position=np.array([0.4, -0.3, 0.0])
        )
        print("✓ Robot: 7-DOF Panda at position [0.4, -0.3, 0.0]")
        
        # ====================================================================
        # 2. ACTION SPACE - Continuous 4D control
        # ====================================================================
        # Action vector: A = (α_skill, α_x, α_y, α_θ) ∈ [-1, 1]^4
        # 
        # α_skill: Skill selection (sign determines push vs. grasp)
        #   - α_skill > 0 → Pick-and-Place
        #   - α_skill ≤ 0 → Push
        # 
        # α_x, α_y: Contact point in target object's local frame
        #   - For grasping: defines grasp approach point
        #   - For pushing: defines push contact location
        # 
        # α_θ: Push direction angle (only used when α_skill ≤ 0)
        #   - Mapped to [-π, π] for 360° push directions
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        print("✓ Action space: 4D continuous [-1, 1]")
        
        # ====================================================================
        # 3. SCENE PARAMETERS
        # ====================================================================
        self.objects = {}  # Dictionary: {name: {info dict}}
        
        # Goal zone: Green square where objects should be collected
        self.goal_pos = np.array([-0.2, -0.2])  # Center position (x, y)
        self.goal_size = 0.1  # Square side length in meters
        
        # Table workspace bounds
        self.table_bounds = np.array([0.4, 0.4])  # Half-extents in x, y
        
        # Setup flag
        self.scene_setup = False
        print(f"✓ Goal zone: center={self.goal_pos}, size={self.goal_size}m")
        
        # ====================================================================
        # 4. EPISODE TRACKING VARIABLES
        # ====================================================================
        self.current_target = None  # Currently selected target object (heuristic)
        self.collected_objects = set()  # Set of object names already in goal
        self.episode_step = 0  # Current step count
        self.max_episode_steps = 100  # Episode truncation limit
        
        # ====================================================================
        # 5. REWARD TRACKING VARIABLES
        # ====================================================================
        # These track previous states to compute shaped rewards
        
        # For trajectory length penalty (-0.01 per unit radian/move)
        self.previous_joint_positions = None
        
        # For distance-based shaping (reward approaching goal)
        self.previous_object_distances = {}  # {obj_name: distance_to_goal}
        
        # For occlusion tracking (reward clearing occlusions)
        self.previous_occlusion_states = {}  # {obj_name: was_occluded}
        
        # For failure penalties (failed grasp/push: -3)
        self.action_was_successful = True
        
        # ====================================================================
        # 6. OBSERVATION SPACE - FIXED SIZE WITH PADDING
        # ====================================================================
        # Challenge: Variable number of objects (5-10) → variable obs size
        # Solution: Pad to MAX_OBJECTS with zeros for fixed NN input size
        #
        # Observation structure (total 348D):
        # ┌─────────────────────────────────────────────────────────────┐
        # │ 1. Robot state (22D)                                        │
        # │    - Joint positions: 7D                                    │
        # │    - Joint velocities: 7D                                   │
        # │    - End-effector position: 3D                              │
        # │    - End-effector orientation (quaternion): 4D              │
        # │    - Gripper width: 1D                                      │
        # ├─────────────────────────────────────────────────────────────┤
        # │ 2. Environment info (6D)                                    │
        # │    - Goal position: 2D                                      │
        # │    - Goal size: 1D                                          │
        # │    - Table bounds: 2D                                       │
        # │    - Number of objects: 1D                                  │
        # ├─────────────────────────────────────────────────────────────┤
        # │ 3. Object states (MAX_OBJECTS × 21D = 210D)                │
        # │    Per object (21D):                                        │
        # │      - Position: 3D                                         │
        # │      - Orientation (quaternion): 4D                         │
        # │      - Linear velocity: 3D                                  │
        # │      - Angular velocity: 3D                                 │
        # │      - Shape descriptor: 8D (type, dimensions, etc.)        │
        # ├─────────────────────────────────────────────────────────────┤
        # │ 4. Distance matrix (MAX_OBJECTS² = 100D)                   │
        # │    - Pairwise Euclidean distances between all objects       │
        # │    - Flattened N×N matrix, padded to 10×10                  │
        # ├─────────────────────────────────────────────────────────────┤
        # │ 5. Occlusion mask (MAX_OBJECTS = 10D)                      │
        # │    - Binary flag per object: 1 if occluded, 0 otherwise    │
        # └─────────────────────────────────────────────────────────────┘
        #
        # Formula: 22 + 6 + (21×10) + 10² + 10 = 348D
        obs_dim = 28 + self.MAX_OBJECTS * 21 + self.MAX_OBJECTS**2 + self.MAX_OBJECTS
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        print(f"✓ Observation space: {obs_dim}D")
        print(f"  Formula: 28 + {self.MAX_OBJECTS}×21 + {self.MAX_OBJECTS}² + {self.MAX_OBJECTS}")
        print(f"  = 22 (robot) + 6 (env) + 210 (objects) + 100 (dist) + 10 (occl)")
        print("=" * 70 + "\n")

    def _draw_goal_square(self):
        """
        Draw green square outline for goal zone visualization.
        Uses PyBullet's debug drawing API to render a persistent green square
        on the table surface indicating where objects should be collected.
        """
        half_size = self.goal_size / 2
        color = [0, 1, 0]  # Green RGB
        z = 0.001 # Slightly above table to avoid z-fighting
        cx, cy = self.goal_pos

        # Define 4 corners of the square
        corners = [
            [cx - half_size, cy - half_size, z],
            [cx + half_size, cy - half_size, z],
            [cx + half_size, cy + half_size, z],
            [cx - half_size, cy + half_size, z]
        ]

        # Draw 4 edges connecting corners
        client = self.sim.physics_client
        for i in range(4):
            client.addUserDebugLine(
                corners[i],
                corners[(i + 1) % 4], # Connect to next corner (wrap around)
                color,
                lineWidth=3
            )

    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        """
        Get complete robot state (22D total).
        
        Extracts all robot-related observations from the Panda robot and PyBullet.
        
        Returns:
            Dictionary with robot state components:
            - joint_positions (7D): Current joint angles [rad]
            - joint_velocities (7D): Joint angular velocities [rad/s]
            - ee_position (3D): End-effector Cartesian position [m]
            - ee_orientation (4D): End-effector orientation as quaternion [x,y,z,w]
            - gripper_width (1D): Distance between gripper fingers [m]
        
        Implementation Notes:
        - Joint positions come from robot.get_obs()
        - Joint velocities require direct PyBullet query (not in robot.get_obs())
        - End-effector pose uses forward kinematics via getLinkState()
        """
        # Get observation from Panda robot wrapper
        # Structure: [joint_pos (7), gripper_fingers (2), ...]
        robot_obs = self.robot.get_obs()
        
        # Extract joint positions (first 7 elements)
        joint_positions = np.array(robot_obs[:7], dtype=np.float32)
        
        # ====================================================================
        # JOINT VELOCITIES - Requires direct PyBullet access
        # ====================================================================
        # robot.get_obs() doesn't include velocities, so we query PyBullet
        
        joint_velocities = np.zeros(7, dtype=np.float32)
        panda_uid = self.sim._bodies_idx.get("panda")
        
        if panda_uid is not None:
            # Panda has 7 revolute arm joints (indices 0-6 in URDF)
            for i in range(7):
                try:
                    joint_velocities[i] = self.sim.get_joint_velocity("panda", i)
                except:
                    # Fallback to zero if query fails
                    joint_velocities[i] = 0.0
        
        # ====================================================================
        # GRIPPER STATE
        # ====================================================================
        # Gripper has 2 prismatic finger joints (indices 7-8 in robot_obs)
        # Average their positions to get approximate gripper width
        if len(robot_obs) >= 9:
            gripper_state = np.array(robot_obs[7:9], dtype=np.float32)
        else:
            gripper_state = np.zeros(2, dtype=np.float32)
        
        # ====================================================================
        # END-EFFECTOR POSE - Forward kinematics
        # ====================================================================
        # Get EE link index (typically link 11 for Panda's EE frame)
        ee_link = getattr(self.robot, 'ee_link', 11)
        
        try:
            if panda_uid is not None:
                # Query link state with forward kinematics enabled
                link_state = self.sim.physics_client.getLinkState(
                    panda_uid, 
                    ee_link, 
                    computeForwardKinematics=1
                )
                ee_pos = np.array(link_state[0], dtype=np.float32)   # World position
                ee_quat = np.array(link_state[1], dtype=np.float32)  # World orientation
            else:
                # Fallback if robot UID not found
                ee_pos = np.zeros(3, dtype=np.float32)
                ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)  # Identity quaternion
        except Exception as e:
            print(f"Warning: Could not get EE pose: {e}")
            ee_pos = np.zeros(3, dtype=np.float32)
            ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)
    
        return {
            'joint_positions':    joint_positions,      # 7D
            'joint_velocities':   joint_velocities,     # 7D
            'ee_position':        ee_pos,               # 3D
            'ee_orientation':     ee_quat,              # 4D (x, y, z, w)
            'gripper_width':      np.array([np.mean(gripper_state)], dtype=np.float32),  # 1D
        }

    def _get_object_states(self) -> Dict[str, np.ndarray]:
        """
        Get complete object states (N × 21D).
        
        Queries PyBullet for the state of all objects in the scene and computes
        their shape descriptors.
        
        Per object (21D):
        - Position: 3D Cartesian coordinates [m]
        - Orientation: 4D quaternion [x, y, z, w]
        - Linear velocity: 3D [m/s]
        - Angular velocity: 3D [rad/s]
        - Shape descriptor: 8D (type encoding, dimensions, graspability, etc.)
        
        Returns:
            Dictionary with stacked arrays for all N objects:
            - positions: (N, 3)
            - orientations: (N, 4)
            - velocities: (N, 3)
            - angular_velocities: (N, 3)
            - shape_descriptors: (N, 8)
        
        Note: Returns empty arrays if no objects exist (N=0)
        """
        # Sort object names for consistent ordering across observations
        object_names = sorted(self.objects.keys())
        N = len(object_names)
        
        # Handle empty scene
        if N == 0:
            return {
                'positions': np.array([]).reshape(0, 3),
                'orientations': np.array([]).reshape(0, 4),
                'velocities': np.array([]).reshape(0, 3),
                'angular_velocities': np.array([]).reshape(0, 3),
                'shape_descriptors': np.array([]).reshape(0, 8)
            }
        
        # Initialize arrays for N objects
        positions = np.zeros((N, 3), dtype=np.float32)
        orientations = np.zeros((N, 4), dtype=np.float32)
        velocities = np.zeros((N, 3), dtype=np.float32)
        angular_velocities = np.zeros((N, 3), dtype=np.float32)
        shape_descriptors = np.zeros((N, 8), dtype=np.float32)
        
        # Query PyBullet for each object's state
        for i, name in enumerate(object_names):
            positions[i] = self.sim.get_base_position(name)
            orientations[i] = self.sim.get_base_orientation(name)
            velocities[i] = self.sim.get_base_velocity(name)
            angular_velocities[i] = self.sim.get_base_angular_velocity(name)
            
            # Shape descriptor computed during object creation (cached in self.objects)
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
        
        Calculates:
        1. Distance matrix: Pairwise Euclidean distances between all objects
        2. Occlusion mask: Binary flags indicating which objects are occluded
        
        Occlusion detection:
        - An object is "occluded" if another object is between it and the goal
        - Uses geometric reasoning with OCCLUSION_THRESHOLD (5cm)
        - Occluded objects should ideally be pushed aside first
        
        Returns:
            Dictionary with:
            - distance_matrix: (N, N) float array of pairwise distances [m]
            - occlusion_mask: (N,) binary array (1=occluded, 0=free)
        """
        # Compute N×N matrix of all pairwise distances
        distance_matrix = compute_pairwise_distance_matrix(self.sim, self.objects)
        
        # Determine which objects are blocked from goal by other objects
        occlusion_mask = compute_occlusion_masks(
            self.sim,
            self.objects,
            threshold=self.OCCLUSION_THRESHOLD  # 5cm proximity threshold
        )
        
        return {
            'distance_matrix': distance_matrix,
            'occlusion_mask': occlusion_mask
        }

    def _get_obs(self) -> np.ndarray:
        """
        Construct complete observation vector WITH PADDING to MAX_OBJECTS.
        
        This is the main observation function called by the RL algorithm.
        Assembles all state components into a fixed-size vector by padding
        to MAX_OBJECTS, enabling use with standard neural network architectures.
        
        Padding Strategy:
        - Actual objects: Fill first N slots with real data
        - Remaining slots: Fill with zeros (MAX_OBJECTS - N)
        - Network can learn to ignore padded entries via the "num_objects" field
        
        Observation Structure (348D total):
        ┌──────────────────────────────────────────────────────────┐
        │ [0:22]    Robot state (22D)                              │
        │ [22:28]   Environment info (6D)                          │
        │ [28:238]  Object states (210D = MAX_OBJECTS×21)          │
        │ [238:338] Distance matrix (100D = MAX_OBJECTS²)          │
        │ [338:348] Occlusion mask (10D = MAX_OBJECTS)             │
        └──────────────────────────────────────────────────────────┘
        
        Returns:
            Flattened observation array of shape (348,)
        """
        # ====================================================================
        # 1. ROBOT STATE (22D)
        # ====================================================================
        robot = self._get_robot_state()
        robot_vector = np.concatenate([
            robot['joint_positions'],      # 7D - joint angles [rad]
            robot['joint_velocities'],     # 7D - joint velocities [rad/s]
            robot['ee_position'],          # 3D - EE position [m]
            robot['ee_orientation'],       # 4D - EE orientation [quaternion]
            robot['gripper_width']         # 1D - gripper opening [m]
        ])
        
        # ====================================================================
        # 2. ENVIRONMENT INFORMATION (6D)
        # ====================================================================
        N = len(self.objects)  # Actual number of objects in current episode
        env_info = np.array([
            self.goal_pos[0],      # Goal zone center X [m]
            self.goal_pos[1],      # Goal zone center Y [m]
            self.goal_size,        # Goal zone side length [m]
            self.table_bounds[0],  # Table X half-extent [m]
            self.table_bounds[1],  # Table Y half-extent [m]
            float(N)               # Number of active objects (for mask learning)
        ], dtype=np.float32)
        
        # ====================================================================
        # 3. OBJECT STATES (MAX_OBJECTS×21D = 210D) - PADDED
        # ====================================================================
        objects = self._get_object_states()
        
        # Create zero-padded arrays for MAX_OBJECTS
        positions_padded = np.zeros((self.MAX_OBJECTS, 3), dtype=np.float32)
        orientations_padded = np.zeros((self.MAX_OBJECTS, 4), dtype=np.float32)
        velocities_padded = np.zeros((self.MAX_OBJECTS, 3), dtype=np.float32)
        angular_velocities_padded = np.zeros((self.MAX_OBJECTS, 3), dtype=np.float32)
        shape_descriptors_padded = np.zeros((self.MAX_OBJECTS, 8), dtype=np.float32)
        
        # Fill first N slots with actual object data
        if N > 0:
            positions_padded[:N] = objects['positions']
            orientations_padded[:N] = objects['orientations']
            velocities_padded[:N] = objects['velocities']
            angular_velocities_padded[:N] = objects['angular_velocities']
            shape_descriptors_padded[:N] = objects['shape_descriptors']
        
        # Flatten into single vector: 21 × MAX_OBJECTS = 210D
        object_vector = np.concatenate([
            positions_padded.flatten(),           # MAX_OBJECTS×3 = 30D
            orientations_padded.flatten(),        # MAX_OBJECTS×4 = 40D
            velocities_padded.flatten(),          # MAX_OBJECTS×3 = 30D
            angular_velocities_padded.flatten(),  # MAX_OBJECTS×3 = 30D
            shape_descriptors_padded.flatten()    # MAX_OBJECTS×8 = 80D
        ])
        
        # ====================================================================
        # 4. SPATIAL RELATIONSHIPS (PADDED)
        # ====================================================================
        spatial = self._get_spatial_relationships()
        
        # Pad distance matrix to MAX_OBJECTS × MAX_OBJECTS (10×10 = 100D)
        distance_matrix_padded = np.zeros((self.MAX_OBJECTS, self.MAX_OBJECTS), dtype=np.float32)
        if N > 0:
            distance_matrix_padded[:N, :N] = spatial['distance_matrix']
        
        # Pad occlusion mask to MAX_OBJECTS (10D)
        occlusion_mask_padded = np.zeros(self.MAX_OBJECTS, dtype=np.float32)
        if N > 0:
            occlusion_mask_padded[:N] = spatial['occlusion_mask'].astype(np.float32)
        
        # ====================================================================
        # 5. CONCATENATE ALL COMPONENTS
        # ====================================================================
        # Total dimension check: 22 + 6 + 210 + 100 + 10 = 348D ✓
        obs = np.concatenate([
            robot_vector,                      # 22D
            env_info,                          # 6D
            object_vector,                     # 210D
            distance_matrix_padded.flatten(),  # 100D
            occlusion_mask_padded              # 10D
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










