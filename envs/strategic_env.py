import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
import pybullet_data
import pybullet as p

# Import utility functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.object_utils import (
    compute_shape_descriptors,
    compute_pairwise_distance_matrix,
    compute_occlusion_masks,
    analyze_scene_occlusions,
    update_object_colors,
    select_target_heuristic,
    get_safe_spawn_position,
    check_object_in_goal
)
from utils.robot_utils import (
    execute_pick_and_place,
    execute_push
)
from utils.physics_utils import (
    check_workspace_violation,
    check_collision_with_table,
    check_object_collision
)

class StrategicPushAndGraspEnv(gym.Env):
    """
    一个为ME5418项目定制的Gym环境。
    该环境继承自基础的gym.Env，并完全自定义场景的创建与交互逻辑。
    - 机器人固定在桌子一角。
    - 场景中有一个方形的目标区域。
    - 每回合会随机生成多个方块和球体。
    - 物体的颜色会根据其是否被遮挡而动态变化。
    """
    # Gymnasium环境的元数据，声明支持“human”渲染模式
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # 定义颜色常量，方便统一管理和修改
    COLOR_GREEN = [0.1, 0.8, 0.1, 1.0]  # 默认方块颜色
    COLOR_YELLOW = [0.8, 0.8, 0.1, 1.0]  # 默认球体颜色，且球体默认不可被抓取
    COLOR_RED = [0.8, 0.1, 0.1, 1.0]  # 遮挡物体的颜色，遮挡物不可被抓取

    def __init__(self, render_mode: str = "human"):
        """
        环境的构造函数，在创建环境实例时被调用一次。
        负责初始化模拟器、机器人、动作/状态空间以及其他必要的参数。
        """
        print("Initializing Strategic Environment...")
        # 1. 创建我们自己完全管理的模拟器和机器人
        self.sim = PyBullet(render_mode=render_mode)
        self.robot = Panda(self.sim, block_gripper=True, base_position=np.array([0.4, -0.3, 0.0]))

        # 2. 定义我们自己的高层动作空间 (a_skill, ax, ay, a_theta)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # 3. 初始化场景和物体相关的参数
        self.objects = {}  # 使用字典来存储场景中所有动态物体及其属性
        self.goal_pos = np.array([-0.2, -0.2])  # 目标区域的中心坐标
        self.goal_size = 0.1  # 目标区域的边长
        self.scene_setup = False  # 一个标志，用于确保静态场景（如桌子）只被加载一次
        self.occlusion_threshold = 0.05  # 判断两个物体是否相互遮挡的距离阈值

        # 4. 通过调用一次reset来获取真实的观测维度，并以此定义观测空间
        initial_obs, _ = self.reset()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=initial_obs.shape, dtype=np.float32)

    def _draw_goal_square(self):
        """使用PyBullet的调试工具在目标位置画一个绿色的方形轮廓"""
        half_size, color, z = self.goal_size / 2, [0, 1, 0], 0.001
        cx, cy = self.goal_pos
        p1, p2 = [cx - half_size, cy - half_size, z], [cx + half_size, cy - half_size, z]
        p3, p4 = [cx + half_size, cy + half_size, z], [cx - half_size, cy + half_size, z]
        client = self.sim.physics_client
        # 使用四条调试线段连接四个角点，形成一个正方形
        client.addUserDebugLine(p1, p2, color, 3);
        client.addUserDebugLine(p2, p3, color, 3)
        client.addUserDebugLine(p3, p4, color, 3);
        client.addUserDebugLine(p4, p1, color, 3)

    def _is_success(self, achieved_goal: np.ndarray) -> bool:
        """判断一个坐标点是否在目标方形区域内"""
        half_size = self.goal_size / 2
        object_xy = achieved_goal[:2]  # 只关心x,y坐标
        # 检查点的x和y坐标是否都落在正方形的边界之内
        return (
                (self.goal_pos[0] - half_size < object_xy[0] < self.goal_pos[0] + half_size) and
                (self.goal_pos[1] - half_size < object_xy[1] < self.goal_pos[1] + half_size)
        )

    def _get_safe_spawn_position(self, existing_objects):
        """生成一个安全的、不重叠的物体生成位置"""
        is_valid = False
        while not is_valid:
            # 1. 在一个大的桌面区域内随机生成一个候选位置
            pos_xy = np.random.uniform(low=-0.3, high=0.3, size=2)

            # 2. 检查该位置是否在目标区域内
            is_in_goal = self._is_success(pos_xy)

            # 3. 检查该位置是否与其他已存在的物体过于接近
            is_overlapping = False
            if existing_objects:
                distances = [np.linalg.norm(pos_xy - np.array(self.sim.get_base_position(name)[:2])) for name in
                             existing_objects]
                if min(distances) < 0.08:
                    is_overlapping = True

            # 4. 如果既不在目标区内，也不与其他物体重叠，则该位置有效
            if not is_in_goal and not is_overlapping:
                is_valid = True
        return [pos_xy[0], pos_xy[1], 0.02]

    def _analyze_scene_states(self):
        """分析场景中物体间的状态，例如是否相互遮挡"""
        object_names = list(self.objects.keys())

        # 每次分析前，先重置所有物体的遮挡状态为False，以反映当前帧的真实情况
        for name in object_names:
            self.objects[name]["is_occluded"] = False

        # 通过双重循环遍历所有独一无二的物体对
        for i in range(len(object_names)):
            for j in range(i + 1, len(object_names)):
                name1, name2 = object_names[i], object_names[j]
                pos1 = np.array(self.sim.get_base_position(name1)[:2])
                pos2 = np.array(self.sim.get_base_position(name2)[:2])
                distance = np.linalg.norm(pos1 - pos2)

                # 如果距离小于阈值，则将两个物体都标记为“被遮挡”
                if distance < self.occlusion_threshold:
                    self.objects[name1]["is_occluded"] = True
                    self.objects[name2]["is_occluded"] = True

    def _update_object_colors(self):
        """根据物体的类型和当前状态（是否被遮挡）来更新其在模拟器中的颜色"""
        for name, properties in self.objects.items():
            color_to_set = None
            # 遮挡状态的优先级最高，被遮挡一律为红色
            if properties["is_occluded"]:
                color_to_set = self.COLOR_RED
            # 如果不被遮挡，则根据类型设置默认颜色
            elif properties["type"] == "cube":
                color_to_set = self.COLOR_GREEN
            elif properties["type"] == "sphere":
                color_to_set = self.COLOR_YELLOW

            if color_to_set is not None:
                body_id = self.sim._bodies_idx.get(name)
                if body_id is not None:
                    # 调用PyBullet API来改变物体的视觉属性
                    self.sim.physics_client.changeVisualShape(
                        body_id, -1, rgbaColor=color_to_set
                    )

    def _get_obs(self):
        """收集所有状态信息，构建并返回一个扁平化的观测向量给智能体"""
        # 获取机器人自身的状态（末端执行器位置、速度等）
        robot_obs = self.robot.get_obs()

        object_states = []
        # 遍历场景中所有物体，收集它们的状态
        for name in sorted(self.objects.keys()):
            pos, ori = self.sim.get_base_position(name), self.sim.get_base_orientation(name)
            vel, ang_vel = self.sim.get_base_velocity(name), self.sim.get_base_angular_velocity(name)

            # 将遮挡状态转换为0或1的数字标签
            is_occluded_label = 1.0 if self.objects[name]["is_occluded"] else 0.0

            # 将该物体的所有状态信息拼接成一个向量
            obj_state = np.concatenate([pos, ori, vel, ang_vel, [is_occluded_label]])
            object_states.append(obj_state)

        # 将机器人状态和所有物体的状态拼接成一个最终的、扁平化的观测向量
        return np.concatenate([robot_obs] + object_states) if object_states else robot_obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境到新一回合的初始状态。
        这是每个episode开始时调用的核心函数。
        """
        super().reset(seed=seed)
        # 在no_rendering模式下进行场景设置，可以加快重置速度
        with self.sim.no_rendering():
            # 仅在第一次reset时加载静态场景（桌子）
            if not self.scene_setup:
                self.sim.create_box(body_name="table", half_extents=np.array([0.4, 0.4, 0.01]), mass=0.0,
                                    position=np.array([0, 0, -0.01]), rgba_color=np.array([0.8, 0.8, 0.8, 1]))
                self.scene_setup = True

            # 移除上一回合留下的所有动态物体
            for body_name in self.objects:
                self.sim.physics_client.removeBody(self.sim._bodies_idx.get(body_name))
            self.objects.clear()

            # 画出终点区域
            self._draw_goal_square()
            # 重置机器人姿态
            self.robot.reset()

            # 随机生成新一回合的物体
            num_objects = np.random.randint(5, 9)
            object_types = ["cube", "sphere"]
            for i in range(num_objects):
                body_name = f"object_{i}"
                object_type = np.random.choice(object_types)
                spawn_pos = self._get_safe_spawn_position(self.objects.keys())

                # 创建时赋予默认颜色
                default_color = self.COLOR_GREEN if object_type == "cube" else self.COLOR_YELLOW

                if object_type == "cube":
                    self.sim.create_box(body_name=body_name, half_extents=np.array([0.02] * 3), mass=1.0,
                                        position=spawn_pos, rgba_color=default_color)
                elif object_type == "sphere":
                    self.sim.create_sphere(body_name=body_name, radius=0.02, mass=1.0,
                                           position=spawn_pos, rgba_color=default_color)

                # 将物体信息存入字典
                self.objects[body_name] = {"type": object_type, "is_occluded": False}

            # 在所有物体生成后，统一分析场景状态并更新颜色
            self._analyze_scene_states()
            self._update_object_colors()

        # 返回初始观测和空的info字典
        return self._get_obs(), {}

#TODO 详细定义动作的执行，我这里只是简单的写了一些
    def step(self, action: np.ndarray):
        """
        Execute one step with COMPLETE reward function.
        
        Action Space (from proposal):
        - A = (α_skill, α_x, α_y, α_θ) ∈ [-1, 1]^4
        - α_skill > 0  → execute_pick_and_place()
        - α_skill ≤ 0  → execute_push()
        - α_x, α_y: Contact point in target object's local frame
        - α_θ: Push direction (only used for push)
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
            print(f"Executing PICK-AND-PLACE on {self.current_target}")
            print(f"  α_x={alpha_x:.2f}, α_y={alpha_y:.2f}")
            self.action_was_successful = execute_pick_and_place(
                self.sim, self.robot, self.current_target,
                alpha_x, alpha_y, self.goal_pos
            )
        
        elif action_type == "push":
            # Execute push primitive
            print(f"Executing PUSH on {self.current_target}")
            print(f"  α_x={alpha_x:.2f}, α_y={alpha_y:.2f}, α_θ={alpha_theta:.2f}")
            self.action_was_successful = execute_push(
                self.sim, self.robot, self.current_target,
                alpha_x, alpha_y, alpha_theta
            )
        
        # Update scene
        analyze_scene_occlusions(self.sim, self.objects, self.OCCLUSION_THRESHOLD)
        update_object_colors(
            self.sim, self.objects,
            self.COLOR_GREEN, self.COLOR_YELLOW, self.COLOR_RED
        )
        
        # Get current joint positions for trajectory penalty
        current_joints = self.robot.get_obs()[:7]
        
        # Compute COMPLETE reward
        reward, reward_info = self._compute_complete_reward(
            action_type=action_type,
            current_joints=current_joints
        )
        
        # Update tracking variables
        self.previous_joint_positions = current_joints
        self.episode_step += 1
        
        # Update target selection
        self.current_target = select_target_heuristic(
            self.sim, self.objects, self.goal_pos, self.collected_objects
        )
        
        # Get new observation
        obs = self._get_obs()
        
        # Check termination
        terminated = len(self.collected_objects) >= len(self.objects) * 0.95
        truncated = self.episode_step >= self.max_episode_steps
        
        info = {
            "is_success": terminated,
            "collected": len(self.collected_objects),
            "total": len(self.objects),
            "reward_breakdown": reward_info,
            "action_type": action_type
        }
        
        return obs, float(reward), bool(terminated), truncated, info
        
    def _compute_complete_reward(self, action_type: str, 
                                 current_joints: np.ndarray) -> Tuple[float, dict]:
        """
        COMPLETE REWARD FUNCTION - All 8 components from proposal.
        
        Returns:
            total_reward: Sum of all reward components
            reward_info: Dictionary with breakdown of each component
        """
        reward_info = {}
        
        # ========== Component 1: Correctly placing one object (+5) ==========
        placement_reward = 0.0
        if self.current_target and self.current_target in self.objects:
            obj_pos = self.sim.get_base_position(self.current_target)
            if check_object_in_goal(obj_pos, self.goal_pos, self.goal_size):
                if self.current_target not in self.collected_objects:
                    placement_reward = 5.0
                    self.collected_objects.add(self.current_target)
                    print(f"✓ Object {self.current_target} placed in goal! (+5)")
        reward_info['placement'] = placement_reward
        
        # ========== Component 2: More than 95% objects collected (+25) ==========
        completion_reward = 0.0
        completion_threshold = len(self.objects) * 0.95
        if len(self.collected_objects) >= completion_threshold:
            completion_reward = 25.0
            print(f"✓ Task completed! {len(self.collected_objects)}/{len(self.objects)} objects (+25)")
        reward_info['completion'] = completion_reward
        
        # ========== Component 3: Successful push reduces distance/occlusion (+0.5) ==========
        push_success_reward = 0.0
        if action_type == "push" and self.current_target and self.action_was_successful:
            # Check if push moved object closer to goal
            obj_pos = np.array(self.sim.get_base_position(self.current_target)[:2])
            current_dist = np.linalg.norm(obj_pos - self.goal_pos)
            previous_dist = self.previous_object_distances.get(self.current_target, current_dist)
            
            distance_reduced = previous_dist - current_dist > 0.01  # Moved >1cm closer
            
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
        
        # ========== Component 4: Failed grasp/push attempt (-3) ==========
        failure_penalty = 0.0
        if not self.action_was_successful:
            failure_penalty = -3.0
            print(f"✗ Action failed! ({action_type}) (-3)")
        reward_info['failure'] = failure_penalty
        
        # ========== Component 5: Object leaves workspace (-10) ==========
        workspace_penalty = 0.0
        for obj_name in list(self.objects.keys()):
            if check_workspace_violation(
                self.sim, obj_name,
                self.WORKSPACE_BOUNDS,
                z_min=-0.05
            ):
                workspace_penalty = -10.0
                print(f"✗ Object {obj_name} left workspace! (-10)")
                # Remove object from tracking
                if obj_name in self.objects:
                    del self.objects[obj_name]
                break
        reward_info['workspace_violation'] = workspace_penalty
        
        # ========== Component 6: Collision with fixed obstacles (-5) ==========
        collision_penalty = 0.0
        
        # Check collision with table
        if check_collision_with_table(self.sim, 'panda', 'table'):
            collision_penalty = -5.0
            print(f"✗ Robot collided with table! (-5)")
        
        # Check collision with other objects (not target)
        if collision_penalty == 0.0:  # Only check if no table collision
            for obj_name in self.objects:
                if obj_name != self.current_target:
                    if check_object_collision(self.sim, 'panda', obj_name):
                        collision_penalty = -5.0
                        print(f"✗ Robot collided with {obj_name}! (-5)")
                        break
        
        reward_info['collision'] = collision_penalty
        
        # ========== Component 7: Each action (-0.1) ==========
        step_penalty = -0.1
        reward_info['step'] = step_penalty
        
        # ========== Component 8: Trajectory length penalty (-0.01 per unit radian) ==========
        trajectory_penalty = 0.0
        if self.previous_joint_positions is not None:
            # Calculate total joint displacement across all 7 DOF
            joint_displacement = np.abs(current_joints - self.previous_joint_positions)
            total_movement = np.sum(joint_displacement)  # Sum in radians
            
            # Penalty: -0.01 per radian moved
            trajectory_penalty = -0.01 * total_movement
            
            if total_movement > 0.1:  # Only print if significant movement
                print(f"Trajectory: {total_movement:.3f} rad → penalty: {trajectory_penalty:.4f}")
        reward_info['trajectory'] = trajectory_penalty
        
        # ========== TOTAL REWARD ==========
        total_reward = sum(reward_info.values())
        
        return total_reward, reward_info

    def close(self):
        """Clean up environment resources."""
        self.sim.close()
