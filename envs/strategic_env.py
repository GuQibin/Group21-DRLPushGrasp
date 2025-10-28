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
    Strategic Push-Grasp Environment with FIXED target selection handling.
    
    CRITICAL CHANGES FROM ORIGINAL:
    - Robust handling of "No Target Selected" scenario
    - Objects that fall off workspace are marked as collected
    - Episode auto-terminates when no valid targets remain
    - Target validation before every action execution
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Color constants for visual feedback
    COLOR_GREEN = [0.1, 0.8, 0.1, 1.0]   # Graspable cubes
    COLOR_YELLOW = [0.8, 0.8, 0.1, 1.0]  # Spheres
    COLOR_RED = [0.8, 0.1, 0.1, 1.0]     # Occluded objects

    # Environment parameters
    MAX_OBJECTS = 10
    WORKSPACE_BOUNDS = (-0.25, 0.25, -0.25, 0.25)
    OCCLUSION_THRESHOLD = 0.05
    MIN_OBJECT_SEPARATION = 0.08

    def __init__(self, render_mode: str = "human"):
        """Initialize the environment with robust target selection."""
        print("=" * 70)
        print("Initializing Strategic Push-Grasp Environment (FIXED)")
        print("=" * 70)

        # Create simulation
        self.sim = PyBullet(render_mode=render_mode, background_color=np.array([200, 220, 230]))
        self.robot = Panda(self.sim, block_gripper=False, base_position=np.array([0, 0, 0]))

        # State tracking
        self.objects = {}
        self.collected_objects = set()
        self.current_target = None
        self.action_was_successful = False
        self.scene_setup = False
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 100
        
        # Reward shaping variables
        self.previous_joint_positions = None
        self.previous_object_distances = {}
        self.previous_occlusion_states = {}
        
        # Goal zone
        self.goal_pos = np.array([0.15, 0.15], dtype=np.float32)
        self.goal_size = 0.12
        self.table_bounds = np.array([0.4, 0.4], dtype=np.float32)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = 22 + 6 + (self.MAX_OBJECTS * 21) + (self.MAX_OBJECTS * self.MAX_OBJECTS) + self.MAX_OBJECTS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space}")
        print("=" * 70 + "\n")

    def _remove_object(self, body_name: str, mark_as_collected: bool = True):
        """
        FIXED: Remove object and optionally mark as collected to prevent re-selection.
        
        Args:
            body_name: Name of object to remove
            mark_as_collected: If True, add to collected_objects set
                             This prevents the object from being selected again
        """
        if body_name in self.objects:
            body_id = self.sim._bodies_idx.get(body_name)
            if body_id is not None:
                p.removeBody(bodyUniqueId=body_id)

            # Clean up from all tracking dictionaries
            del self.objects[body_name]
            if body_name in self.sim._bodies_idx:
                del self.sim._bodies_idx[body_name]
            if body_name in self.previous_object_distances:
                del self.previous_object_distances[body_name]
            if body_name in self.previous_occlusion_states:
                del self.previous_occlusion_states[body_name]

            # CRITICAL FIX: Mark as collected to prevent re-selection
            if mark_as_collected:
                self.collected_objects.add(body_name)
                print(f"🧹 Object {body_name} removed and marked as collected.")
            else:
                print(f"🧹 Object {body_name} removed from scene.")

    def _check_and_remove_collected_objects(self) -> int:
        """
        FIXED: Check for objects in goal zone and properly track them.
        Returns count of newly collected objects.
        """
        newly_collected_count = 0
        objects_to_remove = []
        
        for obj_name in list(self.objects.keys()):
            if obj_name not in self.collected_objects:
                obj_pos = self.sim.get_base_position(obj_name)
                if check_object_in_goal(obj_pos, self.goal_pos, self.goal_size):
                    self.collected_objects.add(obj_name)
                    newly_collected_count += 1
                    objects_to_remove.append(obj_name)
                    print(f"✓ Object {obj_name} entered goal! Marked for removal.")

        # Remove collected objects
        for obj_name in objects_to_remove:
            self._remove_object(obj_name, mark_as_collected=True)

        return newly_collected_count

    def _validate_and_select_target(self) -> Optional[str]:
        """
        FIXED: Robust target selection with validation.
        
        Returns:
            Valid target object name, or None if no valid targets exist
        """
        # First, ensure current target is still valid
        if self.current_target is not None:
            if self.current_target not in self.objects:
                print(f"⚠️ Current target {self.current_target} no longer exists, re-selecting...")
                self.current_target = None
            elif self.current_target in self.collected_objects:
                print(f"⚠️ Current target {self.current_target} already collected, re-selecting...")
                self.current_target = None

        # Select new target if needed
        if self.current_target is None:
            self.current_target = select_target_heuristic(
                self.sim, self.objects, self.goal_pos, self.collected_objects
            )
            
            if self.current_target is not None:
                print(f"🎯 Selected target: {self.current_target}")
            else:
                print(f"⚠️ No valid targets available (objects={len(self.objects)}, collected={len(self.collected_objects)})")

        return self.current_target

    def step(self, action: np.ndarray):
        """
        FIXED: Execute environment step with robust target handling.
        
        Key improvements:
        1. Validates target before action execution
        2. Auto-terminates episode if no targets remain
        3. Properly handles object removal scenarios
        4. Re-selects target after object collection/removal
        """
        # Parse action vector
        a_skill, alpha_x, alpha_y, alpha_theta = action
        action_type = "grasp" if a_skill > 0 else "push"
        
        # ==================================================================
        # FIX 1: VALIDATE TARGET BEFORE ACTION EXECUTION
        # ==================================================================
        self._validate_and_select_target()
        
        if self.current_target is None:
            # No valid targets - this should trigger episode termination
            self.action_was_successful = False
            print(f"⚠️ No target selected - marking action as failed")
            
            # Give small negative reward but don't penalize heavily
            # (not the agent's fault if no objects remain)
            obs = self._get_obs()
            reward = -0.5  # Small penalty instead of -3
            
            # Auto-terminate episode when no targets available
            terminated = True
            truncated = False
            
            info = {
                "is_success": False,
                "collected": len(self.collected_objects),
                "total": len(self.objects) + len(self.collected_objects),
                "reward_breakdown": {"no_target_penalty": -0.5},
                "action_type": "none",
                "episode_step": self.episode_step,
                "termination_reason": "no_valid_targets"
            }
            
            return obs, float(reward), bool(terminated), truncated, info

        # ==================================================================
        # FIX 2: FORCE PUSH FOR PROBLEMATIC OBJECTS
        # ==================================================================
        if self._should_force_push(self.current_target):
            action_type = "push"

        # ==================================================================
        # EXECUTE ACTION PRIMITIVE
        # ==================================================================
        if action_type == "grasp":
            print(f"\n🤏 Executing PICK-AND-PLACE on {self.current_target}")
            self.action_was_successful = execute_pick_and_place(
                self.sim,
                self.robot,
                self.current_target,
                alpha_x,
                alpha_y,
                self.goal_pos,
                workspace_bounds=self.WORKSPACE_BOUNDS
            )
        elif action_type == "push":
            print(f"\n👉 Executing PUSH on {self.current_target}")
            self.action_was_successful = execute_push(
                self.sim,
                self.robot,
                self.current_target,
                alpha_x,
                alpha_y,
                alpha_theta,
                workspace_bounds=self.WORKSPACE_BOUNDS
            )

        # ==================================================================
        # CHECK FOR COLLECTED OBJECTS
        # ==================================================================
        num_newly_collected = self._check_and_remove_collected_objects()

        # ==================================================================
        # FIX 3: CHECK FOR WORKSPACE VIOLATIONS (with proper marking)
        # ==================================================================
        workspace_penalty = 0.0
        for obj_name in list(self.objects.keys()):
            if check_workspace_violation(self.sim, obj_name, self.WORKSPACE_BOUNDS, z_min=-0.05):
                workspace_penalty = -10.0
                print(f"✗ Object {obj_name} left workspace! (-10)")
                # CRITICAL: Mark as collected to prevent re-selection
                self._remove_object(obj_name, mark_as_collected=True)
                break

        # ==================================================================
        # UPDATE SCENE ANALYSIS
        # ==================================================================
        analyze_scene_occlusions(self.sim, self.objects, self.OCCLUSION_THRESHOLD)
        update_object_colors(self.sim, self.objects, self.COLOR_GREEN, self.COLOR_YELLOW, self.COLOR_RED)

        # ==================================================================
        # GET JOINT STATE FOR TRAJECTORY PENALTY
        # ==================================================================
        current_joints = self.robot.get_obs()[:7]

        # ==================================================================
        # COMPUTE REWARD
        # ==================================================================
        reward, reward_info = self._compute_complete_reward(
            action_type=action_type,
            current_joints=current_joints,
            num_newly_collected=num_newly_collected,
            workspace_penalty=workspace_penalty
        )

        # ==================================================================
        # UPDATE TRACKING VARIABLES
        # ==================================================================
        self.previous_joint_positions = current_joints
        self.episode_step += 1

        # ==================================================================
        # FIX 4: RE-SELECT TARGET AFTER OBJECT CHANGES
        # ==================================================================
        self._validate_and_select_target()

        # ==================================================================
        # GET NEW OBSERVATION
        # ==================================================================
        obs = self._get_obs()

        # ==================================================================
        # CHECK TERMINATION CONDITIONS
        # ==================================================================
        total_objects_at_start = len(self.objects) + len(self.collected_objects)
        terminated = False
        
        # Success: collected 95% or more
        if total_objects_at_start > 0:
            terminated = len(self.collected_objects) >= total_objects_at_start * 0.95
        
        # Also terminate if no valid targets remain (prevents getting stuck)
        if not terminated and self.current_target is None and len(self.objects) > 0:
            terminated = True
            print(f"⚠️ Episode terminated: no valid targets but {len(self.objects)} objects remain")

        truncated = self.episode_step >= self.max_episode_steps

        # ==================================================================
        # PREPARE INFO DICTIONARY
        # ==================================================================
        info = {
            "is_success": terminated and len(self.collected_objects) >= total_objects_at_start * 0.95,
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
        """
        FIXED: Compute reward with workspace penalty passed in (to avoid double-checking).
        """
        reward_info = {}

        # 1. Object placement (+5 per object)
        reward_info['placement'] = 5.0 * num_newly_collected

        # 2. Task completion (+25)
        completion_reward = 0.0
        total_objects = len(self.objects) + len(self.collected_objects)
        if total_objects > 0:
            completion_threshold = total_objects * 0.95
            prev_collected_count = len(self.collected_objects) - num_newly_collected
            if prev_collected_count < completion_threshold and len(self.collected_objects) >= completion_threshold:
                completion_reward = 25.0
                print(f"✓ Task completed! {len(self.collected_objects)}/{total_objects} objects (+25)")
        reward_info['completion'] = completion_reward

        # 3. Successful push (+0.5)
        push_success_reward = 0.0
        if action_type == "push" and self.current_target and self.action_was_successful:
            if self.current_target in self.objects:
                obj_pos = np.array(self.sim.get_base_position(self.current_target)[:2])
                current_dist = np.linalg.norm(obj_pos - self.goal_pos)
                previous_dist = self.previous_object_distances.get(self.current_target, current_dist)

                distance_reduced = previous_dist - current_dist > 0.01

                was_occluded = self.previous_occlusion_states.get(self.current_target, False)
                now_occluded = self.objects[self.current_target]["is_occluded"]
                occlusion_cleared = was_occluded and not now_occluded

                if distance_reduced or occlusion_cleared:
                    push_success_reward = 0.5
                    if distance_reduced:
                        print(f"✓ Push moved object closer ({previous_dist:.3f}→{current_dist:.3f}m) (+0.5)")
                    if occlusion_cleared:
                        print(f"✓ Push cleared occlusion (+0.5)")

                self.previous_object_distances[self.current_target] = current_dist

        # Update occlusion tracking for all remaining objects
        for name in list(self.objects.keys()):
            self.previous_occlusion_states[name] = self.objects[name]["is_occluded"]

        reward_info['push_success'] = push_success_reward

        # 4. Failed action (-3)
        reward_info['failure'] = -3.0 if not self.action_was_successful else 0.0

        # 5. Workspace violation (-10) - passed in to avoid double-checking
        reward_info['workspace_violation'] = workspace_penalty

        # 6. Collision penalty (-5)
        collision_penalty = 0.0
        if check_collision_with_table(self.sim, 'panda', 'table'):
            collision_penalty = -5.0
            print(f"✗ Robot collided with table! (-5)")

        if collision_penalty == 0.0:
            for obj_name in self.objects:
                if obj_name != self.current_target:
                    if check_object_collision(self.sim, 'panda', obj_name):
                        collision_penalty = -5.0
                        print(f"✗ Robot collided with {obj_name}! (-5)")
                        break
        reward_info['collision'] = collision_penalty

        # 7. Step penalty (-0.1)
        reward_info['step'] = -0.1

        # 8. Trajectory length penalty (-0.01/rad)
        trajectory_penalty = 0.0
        if self.previous_joint_positions is not None:
            joint_displacement = np.abs(current_joints - self.previous_joint_positions)
            total_movement = np.sum(joint_displacement)
            trajectory_penalty = -0.01 * total_movement
        reward_info['trajectory'] = trajectory_penalty

        # Total reward
        total_reward = sum(reward_info.values())
        return total_reward, reward_info

    def _should_force_push(self, target_name: Optional[str]) -> bool:
        """Force push for occluded objects or spheres."""
        if not target_name or target_name not in self.objects:
            return True

        meta = self.objects[target_name]
        is_occluded = bool(meta.get("is_occluded", False))
        typ = meta.get("type", "unknown")
        non_graspable_types = {"sphere"}
        return is_occluded or (typ in non_graspable_types)

    def close(self):
        """Clean up environment resources."""
        self.sim.close()
        print("\nEnvironment closed.")


