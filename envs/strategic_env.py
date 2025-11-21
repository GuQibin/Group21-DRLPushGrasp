import numpy as np
import random
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

# NOTE: Assuming necessary imports from utils are available
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


# ======================================================================
# EXPLORATION BONUS (GLOBAL)
# ======================================================================
def compute_exploration_bonus(
        global_steps: int,  # total training steps completed so far
        total_steps: int,  # total planned training steps (for scaling)
        collected_count: int,  # number of objects successfully collected
        total_objects: int,  # total number of objects in the current scene
        current_target: str,  # name of the current target object
        objects: dict  # metadata dictionary for all objects
) -> float:
    if total_objects == 0:
        return 0.0

    decay_steps = 50_000
    progress = min(global_steps / decay_steps, 1.0)
    base_bonus = 2.0 * (1.0 - progress)

    milestone_bonus = 0.0
    if collected_count == 1:
        milestone_bonus = 3.0
    elif collected_count % 3 == 0:
        milestone_bonus = 1.0

    hard_bonus = 0.0
    if current_target in objects:
        meta = objects[current_target]
        if meta.get("is_occluded", False):
            hard_bonus += 0.5 * (1.0 - progress)

    size_factor = min(total_objects / 5.0, 2.0)
    total_bonus = (base_bonus + milestone_bonus + hard_bonus) * size_factor

    return round(float(total_bonus), 3)


class StrategicPushAndGraspEnv(gym.Env):
    """
    Strategic Push-Grasp Environment with RETRY LOGIC:
    - Priority 1: Rule-Based Grasp.
      - If FAIL < 3 times: Minor penalty (-1.0), retry next step.
      - If FAIL >= 3 times: Remove object, penalty (-5.0).
    - Priority 2: RL-Trained Discrete Push.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Color constants
    COLOR_GREEN = [0.1, 0.8, 0.1, 1.0]  # Non-occluded
    COLOR_YELLOW = [0.8, 0.8, 0.1, 1.0]  # (unused)
    COLOR_RED = [0.8, 0.1, 0.1, 1.0]  # Occluded objects

    # Environment parameters
    MAX_OBJECTS = 12

    # --- WORKSPACE CONFIGURATION ---
    SPAWN_BOUNDS = (-0.15, 0.15, -0.15, 0.15)

    # Expanded reach bounds
    WORKSPACE_BOUNDS = (-0.45, 0.45, -0.45, 0.45)

    OCCLUSION_THRESHOLD = 0.075
    MIN_OBJECT_SEPARATION = 0.08

    PUSH_DIRECTIONS = {
        0: 0.0,  # 0 deg: +X
        1: 0.25,  # 45 deg
        2: 0.5,  # 90 deg: +Y
        3: 0.75,  # 135 deg
        4: 1.0,  # 180 deg: -X
        5: -0.75,  # -135 deg
        6: -0.5,  # -90 deg: -Y
        7: -0.25,  # -45 deg
    }

    def __init__(self, render_mode: str = "human", motion_scale: float = 1.0):
        print("=" * 70)
        print("Initializing Strategic Env (3-Try Grasp Mechanism)")
        print("=" * 70)

        self.sim = PyBullet(render_mode=render_mode)
        self.robot = Panda(
            self.sim,
            block_gripper=False,
            base_position=np.array([-0.5, 0, 0])
        )

        self.objects = {}
        self.collected_objects = set()
        self.current_target = None
        self.action_was_successful = False
        self.scene_setup = False

        # [NEW] Track grasp attempts per object
        self.grasp_attempt_counts = {}

        self.episode_step = 0
        self.max_episode_steps = 100

        self.previous_joint_positions = None
        self.previous_object_distances = {}
        self.previous_occlusion_states = {}
        self._prev_distance_to_target = None
        self.num_occluded_objects_prev = 0

        # Goal Zone
        self.goal_pos = np.array([0.0, 0.35], dtype=np.float32)
        self.goal_size = 0.12
        self.table_bounds = np.array([0.4, 0.4], dtype=np.float32)

        self.global_steps = 0
        self.total_training_steps = 200_000
        self.total_objects_at_start = 0

        self.motion_scale = float(max(0.1, motion_scale))
        self.action_space = spaces.Discrete(8)

        obs_dim = 22 + 6 + (self.MAX_OBJECTS * 21) + (self.MAX_OBJECTS * self.MAX_OBJECTS) + self.MAX_OBJECTS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    # ======================================================================
    # HELPER METHODS
    # ======================================================================

    def _remove_object(self, body_name: str, mark_as_collected: bool = True):
        if body_name in self.objects:
            body_id = self.sim._bodies_idx.get(body_name)
            if body_id is not None:
                p.removeBody(bodyUniqueId=body_id)

            del self.objects[body_name]
            # Clean up tracking dicts
            if body_name in self.grasp_attempt_counts:
                del self.grasp_attempt_counts[body_name]

            if body_name in self.sim._bodies_idx:
                del self.sim._bodies_idx[body_name]
            if body_name in self.previous_object_distances:
                del self.previous_object_distances[body_name]
            if body_name in self.previous_occlusion_states:
                del self.previous_occlusion_states[body_name]

            if mark_as_collected:
                self.collected_objects.add(body_name)
            else:
                print(f"🧹 Object {body_name} removed from scene.")

    def _check_and_remove_collected_objects(self) -> int:
        newly_collected_count = 0
        objects_to_remove = []

        for obj_name in list(self.objects.keys()):
            if obj_name not in self.collected_objects:
                obj_pos = self.sim.get_base_position(obj_name)
                if check_object_in_goal(obj_pos, self.goal_pos, self.goal_size):
                    self.collected_objects.add(obj_name)
                    newly_collected_count += 1
                    objects_to_remove.append(obj_name)
                    print(f"✓ Object {obj_name} placed on goal platform!")

        for obj_name in objects_to_remove:
            self._remove_object(obj_name, mark_as_collected=True)

        return newly_collected_count

    def _validate_and_select_target(self) -> Optional[str]:
        if self.current_target is not None:
            if self.current_target not in self.objects:
                self.current_target = None
            elif self.current_target in self.collected_objects:
                self.current_target = None

        if self.current_target is None:
            self.current_target = select_target_heuristic(
                self.sim, self.objects, self.goal_pos, self.collected_objects
            )

        return self.current_target

    # ======================================================================
    # STEP FUNCTION (RETRY LOGIC)
    # ======================================================================

    def step(self, action: int):
        # --- PHASE 0: DETERMINE HIGH-LEVEL ACTION (RULE-BASED) ---
        uncollected_non_occluded = [
            name for name, meta in self.objects.items()
            if name not in self.collected_objects and not meta.get("is_occluded", False)
        ]

        self.num_occluded_objects_prev = sum(
            1 for meta in self.objects.values() if meta.get("is_occluded", False)
        )

        target_name = None
        alpha_x, alpha_y, alpha_theta = 0.0, 0.0, 0.0

        if len(uncollected_non_occluded) > 0:
            # PRIORITY 1: GRASP (Rule-Based)
            action_type = "grasp"
            target_name = select_target_heuristic(
                self.sim,
                {k: v for k, v in self.objects.items() if not v.get("is_occluded", False)},
                self.goal_pos, self.collected_objects
            )
            if target_name is None:
                target_name = select_target_heuristic(self.sim, self.objects, self.goal_pos, self.collected_objects)
        else:
            # PRIORITY 2: PUSH (RL-Trained)
            action_type = "push"
            discrete_dir_index = action
            alpha_x, alpha_y = 0.0, 0.0
            alpha_theta = self.PUSH_DIRECTIONS.get(discrete_dir_index, 0.0)
            target_name = self._validate_and_select_target()

        self.current_target = target_name

        if self.current_target is None:
            obs = self._get_obs()
            reward = -0.5
            terminated = True
            info = {
                "is_success": False, "collected": len(self.collected_objects),
                "total": self.total_objects_at_start,
                "reward_breakdown": {"no_target_penalty": -0.5},
                "action_type": "none", "episode_step": self.episode_step
            }
            return obs, float(reward), bool(terminated), False, info

        # ==================================================================
        # EXECUTE
        # ==================================================================
        manual_drop_penalty = 0.0

        if action_type == "grasp":
            self.action_was_successful = execute_pick_and_place(
                self.sim, self.robot, self.current_target, alpha_x, alpha_y,
                self.goal_pos, workspace_bounds=self.WORKSPACE_BOUNDS,
                motion_scale=self.motion_scale
            )

            # [MODIFIED] Retry Logic
            if not self.action_was_successful:
                # Increment failure count
                fails = self.grasp_attempt_counts.get(self.current_target, 0) + 1
                self.grasp_attempt_counts[self.current_target] = fails

                print(f"⚠️ Grasp failed for {self.current_target} (Attempt {fails}/3)")

                if fails >= 3:
                    # Max attempts reached -> Remove and Penalize (-5)
                    print(f"💀 Max grasp attempts reached. Removing {self.current_target} (-5.0).")
                    self._remove_object(self.current_target, mark_as_collected=False)
                    manual_drop_penalty = -5.0
                else:
                    # Retry allowed -> Minor penalty (-1.0) to discourage bad pushes
                    # The loop will naturally select it again next step since it's still closest/unoccluded
                    manual_drop_penalty = -1.0
            else:
                # Success -> Clear counter (object will be removed by check_collected anyway)
                if self.current_target in self.grasp_attempt_counts:
                    del self.grasp_attempt_counts[self.current_target]

        elif action_type == "push":
            self.action_was_successful = execute_push(
                self.sim, self.robot, self.current_target, alpha_x, alpha_y,
                alpha_theta, workspace_bounds=self.WORKSPACE_BOUNDS,
                motion_scale=self.motion_scale
            )

        # ==================================================================
        # POST-ACTION CHECKS
        # ==================================================================
        num_newly_collected = self._check_and_remove_collected_objects()

        workspace_penalty = 0.0
        TABLE_LIMITS = (-0.45, 0.45, -0.45, 0.45)

        for obj_name in list(self.objects.keys()):
            if check_workspace_violation(self.sim, obj_name, TABLE_LIMITS, z_min=-0.05):
                workspace_penalty = -10.0
                print(f"✗ Object {obj_name} fell off the table! (-10)")
                self._remove_object(obj_name, mark_as_collected=False)
                break

        total_workspace_penalty = workspace_penalty + manual_drop_penalty

        analyze_scene_occlusions(self.sim, self.objects, self.OCCLUSION_THRESHOLD)
        update_object_colors(self.sim, self.objects, self.COLOR_GREEN, self.COLOR_YELLOW, self.COLOR_RED)

        current_joints = self.robot.get_obs()[:7]

        reward, reward_info = self._compute_complete_reward(
            action_type=action_type, current_joints=current_joints,
            num_newly_collected=num_newly_collected,
            workspace_penalty=total_workspace_penalty
        )

        self.previous_joint_positions = current_joints
        self.episode_step += 1

        for name in list(self.objects.keys()):
            self.previous_occlusion_states[name] = self.objects[name]["is_occluded"]

        self.num_occluded_objects_prev = sum(
            1 for meta in self.objects.values() if meta.get("is_occluded", False)
        )

        obs = self._get_obs()

        total_objects_at_start = self.total_objects_at_start
        terminated = False

        if total_objects_at_start > 0:
            terminated = len(self.collected_objects) >= total_objects_at_start * 0.95

        if not terminated and self.current_target is None and len(self.objects) > 0:
            terminated = True
            print(f" Episode terminated: no valid targets but {len(self.objects)} objects remain")

        truncated = self.episode_step >= self.max_episode_steps

        info = {
            "is_success": terminated and len(self.collected_objects) >= total_objects_at_start * 0.60,
            "collected": len(self.collected_objects),
            "total": total_objects_at_start,
            "reward_breakdown": reward_info,
            "action_type": action_type,
            "episode_step": self.episode_step,
            "current_target": self.current_target
        }

        return obs, float(reward), bool(terminated), truncated, info

    def _compute_complete_reward(self, action_type: str,
                                 current_joints: np.ndarray,
                                 num_newly_collected: int,
                                 workspace_penalty: float = 0.0) -> Tuple[float, dict]:
        reward_info = {}

        num_occluded_objects_curr = sum(
            1 for meta in self.objects.values() if meta.get("is_occluded", False)
        )
        delta_occlusion = self.num_occluded_objects_prev - num_occluded_objects_curr

        reward_info['placement'] = 10.0 * num_newly_collected

        occlusion_reward = 0.0
        if action_type == "push":
            if delta_occlusion > 0:
                occlusion_reward = delta_occlusion * 5.0
                print(f" Push cleared {delta_occlusion} occluded objects! (+{occlusion_reward:.1f})")
            elif delta_occlusion < 0:
                occlusion_reward = delta_occlusion * 3.0
                print(f" Push created {-delta_occlusion} new occlusions! ({occlusion_reward:.1f})")
            else:
                occlusion_reward = -1.0

        reward_info['occlusion_change'] = occlusion_reward

        completion_reward = 0.0
        total_objects_at_start = self.total_objects_at_start
        if total_objects_at_start > 0:
            completion_threshold = total_objects_at_start * 0.95
            prev_collected_count = len(self.collected_objects) - num_newly_collected
            if prev_collected_count < completion_threshold and len(self.collected_objects) >= completion_threshold:
                completion_reward = 25.0
                print(f"✓ Task completion milestone achieved! (+25)")
        reward_info['completion'] = completion_reward

        fail_penalty = 0.0
        if action_type == "push" and not self.action_was_successful:
            fail_penalty = -1.0
        reward_info['failure'] = fail_penalty

        reward_info['workspace_violation'] = workspace_penalty

        collision_penalty = 0.0
        if check_collision_with_table(self.sim, 'panda', 'table'):
            collision_penalty = -2.0
        else:
            for obj_name in self.objects:
                if obj_name != self.current_target:
                    if check_object_collision(self.sim, 'panda', obj_name):
                        collision_penalty = -2.0
                        break
        reward_info['collision'] = collision_penalty

        reward_info['step'] = -0.05 if action_type == "push" else 0.0

        trajectory_penalty = 0.0
        if self.previous_joint_positions is not None:
            joint_displacement = np.abs(current_joints - self.previous_joint_positions)
            total_movement = np.sum(joint_displacement)
            trajectory_penalty = -0.005 * total_movement
        reward_info['trajectory'] = trajectory_penalty

        total_reward = sum(reward_info.values())
        return total_reward, reward_info

    # ======================================================================
    # OBSERVATION
    # ======================================================================

    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()
        joint_positions = np.array(robot_obs[:7], dtype=np.float32)
        joint_velocities = np.zeros(7, dtype=np.float32)
        panda_uid = self.sim._bodies_idx.get("panda")
        if panda_uid is not None:
            for i in range(7):
                try:
                    joint_velocities[i] = self.sim.get_joint_velocity("panda", i)
                except:
                    joint_velocities[i] = 0.0

        try:
            width = float(self.robot.get_fingers_width())
        except Exception:
            width = 0.0
        gripper_state = np.array([width], dtype=np.float32)

        ee_link = getattr(self.robot, 'ee_link', 11)
        try:
            if panda_uid is not None:
                link_state = self.sim.physics_client.getLinkState(
                    panda_uid, ee_link, computeForwardKinematics=1
                )
                ee_pos = np.array(link_state[0], dtype=np.float32)
                ee_quat = np.array(link_state[1], dtype=np.float32)
            else:
                ee_pos = np.zeros(3, dtype=np.float32)
                ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)
        except Exception:
            ee_pos = np.zeros(3, dtype=np.float32)
            ee_quat = np.array([0, 0, 0, 1], dtype=np.float32)

        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'ee_position': ee_pos,
            'ee_orientation': ee_quat,
            'gripper_width': np.array([np.mean(gripper_state)], dtype=np.float32),
        }

    def _get_object_states(self) -> Dict[str, np.ndarray]:
        object_names = sorted(self.objects.keys())
        N = len(object_names)

        if N == 0:
            return {
                'positions': np.array([]).reshape(0, 3),
                'orientations': np.array([]).reshape(0, 4),
                'velocities': np.array([]).reshape(0, 3),
                'angular_velocities': np.array([]).reshape(0, 3),
                'shape_descriptors': np.array([]).reshape(0, 8),
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
            'shape_descriptors': shape_descriptors,
        }

    def _get_spatial_relationships(self) -> Dict[str, np.ndarray]:
        distance_matrix = compute_pairwise_distance_matrix(self.sim, self.objects)
        occlusion_mask = compute_occlusion_masks(self.sim, self.objects, threshold=self.OCCLUSION_THRESHOLD)
        return {'distance_matrix': distance_matrix, 'occlusion_mask': occlusion_mask}

    def _get_obs(self) -> np.ndarray:
        robot = self._get_robot_state()
        robot_vector = np.concatenate([
            robot['joint_positions'], robot['joint_velocities'],
            robot['ee_position'], robot['ee_orientation'], robot['gripper_width'],
        ])

        N = len(self.objects)
        env_info = np.array([
            self.goal_pos[0], self.goal_pos[1], self.goal_size,
            self.table_bounds[0], self.table_bounds[1], float(N),
        ], dtype=np.float32)

        objects = self._get_object_states()
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
            positions_padded.flatten(), orientations_padded.flatten(),
            velocities_padded.flatten(), angular_velocities_padded.flatten(),
            shape_descriptors_padded.flatten(),
        ])

        spatial = self._get_spatial_relationships()
        distance_matrix_padded = np.zeros((self.MAX_OBJECTS, self.MAX_OBJECTS), dtype=np.float32)
        if N > 0:
            distance_matrix_padded[:N, :N] = spatial['distance_matrix']
        occlusion_mask_padded = np.zeros(self.MAX_OBJECTS, dtype=np.float32)
        if N > 0:
            occlusion_mask_padded[:N] = spatial['occlusion_mask'].astype(np.float32)

        obs = np.concatenate([
            robot_vector, env_info, object_vector,
            distance_matrix_padded.flatten(), occlusion_mask_padded,
        ]).astype(np.float32)

        return obs

    def _draw_goal_square(self):
        """Draw goal zone on the separate platform."""
        half = self.goal_size / 2
        color = [0, 1, 0]
        z = 0.001  # On the platform
        cx, cy = self.goal_pos
        corners = [
            [cx - half, cy - half, z], [cx + half, cy - half, z],
            [cx + half, cy + half, z], [cx - half, cy + half, z],
        ]
        client = self.sim.physics_client
        for i in range(4):
            client.addUserDebugLine(
                lineFromXYZ=corners[i], lineToXYZ=corners[(i + 1) % 4],
                lineColorRGB=color, lineWidth=3
            )

    def _spawn_object(self, body_name: str, object_type: str, position: np.ndarray,
                      orientation: Optional[np.ndarray] = None):
        """
        MODIFIED: Reduced friction to prevent objects from standing on edges/corners.
        """
        object_type = "cube"
        default_color = self.COLOR_GREEN

        half_extents = np.array([0.02, 0.02, 0.02])
        self.sim.create_box(body_name=body_name, half_extents=half_extents, mass=1.0,
                            position=position, rgba_color=default_color)
        shape_desc = compute_shape_descriptors("cube", half_extents=half_extents)

        body_id = self.sim._bodies_idx.get(body_name)
        if body_id is not None:
            # Apply dynamics with LOWER FRICTION
            p.changeDynamics(
                bodyUniqueId=body_id,
                linkIndex=-1,
                lateralFriction=0.4,  # Reduced from 1.2
                spinningFriction=0.001,  # Very low
                rollingFriction=0.001,  # Very low
                restitution=0.0,  # No bouncing
                linearDamping=0.04,
                angularDamping=0.04
            )

            # Apply orientation if provided
            if orientation is not None:
                p.resetBasePositionAndOrientation(body_id, position, orientation)

        self.objects[body_name] = {"type": object_type, "is_occluded": False, "shape_descriptor": shape_desc}
        obj_pos = np.array(self.sim.get_base_position(body_name)[:2])
        self.previous_object_distances[body_name] = np.linalg.norm(obj_pos - self.goal_pos)

    def close(self):
        self.sim.close()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.grasp_attempt_counts.clear()  # Clear attempts

        with self.sim.no_rendering():
            if not self.scene_setup:
                # 1. Main Table (Center 0,0)
                self.sim.create_box(
                    body_name="table", half_extents=np.array([0.4, 0.4, 0.01]),
                    mass=0.0, position=np.array([0, 0, -0.01]),
                    rgba_color=np.array([0.8, 0.8, 0.8, 1])
                )

                # 2. Goal Platform (To the side)
                platform_half_size = self.goal_size / 2.0

                self.sim.create_box(
                    body_name="goal_platform",
                    half_extents=np.array([platform_half_size, platform_half_size, 0.01]),
                    mass=0.0,
                    position=np.array([0.0, 0.35, -0.01]),
                    rgba_color=np.array([0.7, 0.7, 0.9, 1])
                )

                self.scene_setup = True

            table_id = self.sim._bodies_idx.get("table")
            if table_id is not None:
                p.changeDynamics(table_id, -1, lateralFriction=1.2)

            for body_name in list(self.objects.keys()):
                body_id = self.sim._bodies_idx.get(body_name)
                if body_id is not None:
                    self.sim.physics_client.removeBody(body_id)
            self.objects.clear()
            self.collected_objects.clear()

            self.previous_joint_positions = None
            self.previous_object_distances.clear()
            self.previous_occlusion_states.clear()

            try:
                self.sim.physics_client.removeAllUserDebugItems()
            except Exception:
                pass

            self._draw_goal_square()
            self.robot.reset()

            if options and options.get("single_object_demo", False):
                if "goal_pos" in options: self.goal_pos = np.array(options["goal_pos"], dtype=np.float32)
                if "goal_size" in options: self.goal_size = float(options["goal_size"])
                self._draw_goal_square()

                obj_xy = np.array(options.get("object_pos", [0.45, 0.00]), dtype=np.float32)
                half = float(options.get("cube_half", 0.02))
                spawn_pos = np.array([obj_xy[0], obj_xy[1], half], dtype=np.float32)
                self._spawn_object("object_0", "cube", spawn_pos)
                self.total_objects_at_start = len(self.objects)

            else:
                # Spawn 8-12 objects in High Density Mode
                num_objects = np.random.randint(8, self.MAX_OBJECTS + 1)

                object_names = []
                for i in range(num_objects):
                    object_names.append(f"object_{i}")

                    # Occlusion generation
                occlusion_dist = self.OCCLUSION_THRESHOLD - 0.01

                spawned_count = 0
                for i in range(0, len(object_names) - 1, 2):
                    name1 = object_names[i]
                    name2 = object_names[i + 1]

                    # MODIFIED: Separation must account for new larger OCCLUSION_THRESHOLD
                    pos1 = get_safe_spawn_position(
                        self.sim, list(self.objects.keys()), self.goal_pos, self.goal_size,
                        min_separation=self.MIN_OBJECT_SEPARATION * 1.1,
                        workspace_bounds=self.SPAWN_BOUNDS
                    )
                    yaw1 = random.uniform(-np.pi, np.pi)
                    ori1 = p.getQuaternionFromEuler([0, 0, yaw1])
                    self._spawn_object(name1, "cube", pos1, orientation=ori1)
                    spawned_count += 1

                    # Random angle relative to obj1
                    rel_angle = random.uniform(-np.pi, np.pi)

                    pos2 = pos1.copy()
                    pos2[0] += occlusion_dist * np.cos(rel_angle)
                    pos2[1] += occlusion_dist * np.sin(rel_angle)

                    yaw2 = random.uniform(-np.pi, np.pi)
                    ori2 = p.getQuaternionFromEuler([0, 0, yaw2])

                    self._spawn_object(name2, "cube", pos2, orientation=ori2)
                    spawned_count += 1

                if spawned_count < num_objects:
                    name_rem = object_names[num_objects - 1]
                    pos_rem = get_safe_spawn_position(
                        self.sim, list(self.objects.keys()), self.goal_pos, self.goal_size,
                        min_separation=self.MIN_OBJECT_SEPARATION, workspace_bounds=self.SPAWN_BOUNDS
                    )
                    yaw_rem = random.uniform(-np.pi, np.pi)
                    ori_rem = p.getQuaternionFromEuler([0, 0, yaw_rem])
                    self._spawn_object(name_rem, "cube", pos_rem, orientation=ori_rem)

                self.total_objects_at_start = len(self.objects)

            analyze_scene_occlusions(self.sim, self.objects, self.OCCLUSION_THRESHOLD)

            self.num_occluded_objects_prev = sum(
                1 for meta in self.objects.values() if meta.get("is_occluded", False)
            )

            for name in self.objects:
                self.previous_occlusion_states[name] = self.objects[name]["is_occluded"]
            update_object_colors(self.sim, self.objects, self.COLOR_GREEN, self.COLOR_YELLOW, self.COLOR_RED)

            self.current_target = select_target_heuristic(self.sim, self.objects, self.goal_pos, self.collected_objects)

        self.previous_joint_positions = self.robot.get_obs()[:7]
        self.episode_step = 0

        return self._get_obs(), {}