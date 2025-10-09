import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

# 导入所有需要的 panda_gym 和 pybullet 组件
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
import pybullet_data
import pybullet as p


# ‼️ 关键架构：我们现在直接继承最基础的 gym.Env
class StrategicPushAndGraspEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: str = "human"):
        print("Initializing Strategic Environment...")
        # 1. 创建我们自己完全管理的模拟器和机器人
        self.sim = PyBullet(render_mode=render_mode)
        self.robot = Panda(self.sim, block_gripper=True, base_position=np.array([0.4, -0.3, 0.0]))

        # 2. 定义我们自己的动作空间
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # 3. 初始化场景和物体参数
        self.objects = {}
        self.goal_pos = np.array([-0.2, -0.2])
        self.goal_size = 0.1
        self.scene_setup = False

        # 4. 通过一次reset来确定观测空间的维度
        initial_obs, _ = self.reset()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=initial_obs.shape, dtype=np.float32)

    def _draw_goal_square(self):
        # ... (此函数无需改动) ...
        half_size, color, z = self.goal_size / 2, [0, 1, 0], 0.001
        cx, cy = self.goal_pos
        p1, p2 = [cx - half_size, cy - half_size, z], [cx + half_size, cy - half_size, z]
        p3, p4 = [cx + half_size, cy + half_size, z], [cx - half_size, cy + half_size, z]
        client = self.sim.physics_client
        client.addUserDebugLine(p1, p2, color, 3);
        client.addUserDebugLine(p2, p3, color, 3)
        client.addUserDebugLine(p3, p4, color, 3);
        client.addUserDebugLine(p4, p1, color, 3)

    def _is_success(self, achieved_goal: np.ndarray) -> bool:
        # ... (此函数无需改动) ...
        half_size = self.goal_size / 2
        return (
                (self.goal_pos[0] - half_size < achieved_goal[0] < self.goal_pos[0] + half_size) and
                (self.goal_pos[1] - half_size < achieved_goal[1] < self.goal_pos[1] + half_size)
        )

    def _get_safe_spawn_position(self, existing_objects):
        # ... (此函数无需改動) ...
        is_valid = False
        while not is_valid:
            pos_xy = np.random.uniform(low=-0.3, high=0.3, size=2)
            is_in_goal = self._is_success(pos_xy)
            is_overlapping = False
            if existing_objects:
                distances = [np.linalg.norm(pos_xy - np.array(self.sim.get_base_position(name)[:2])) for name in
                             existing_objects]
                if min(distances) < 0.08: is_overlapping = True
            if not is_in_goal and not is_overlapping: is_valid = True
        return [pos_xy[0], pos_xy[1], 0.02]

    def _get_obs(self):
        robot_obs = self.robot.get_obs()
        object_states = []
        for name in sorted(self.objects.keys()):
            pos, ori = self.sim.get_base_position(name), self.sim.get_base_orientation(name)
            vel, ang_vel = self.sim.get_base_velocity(name), self.sim.get_base_angular_velocity(name)
            obj_state = np.concatenate([pos, ori, vel, ang_vel])
            object_states.append(obj_state)
        return np.concatenate([robot_obs] + object_states) if object_states else robot_obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        with self.sim.no_rendering():
            if not self.scene_setup:
                self.sim.create_box(body_name="table", half_extents=np.array([0.4, 0.4, 0.01]), mass=0.0,
                                    position=np.array([0, 0, -0.01]), rgba_color=np.array([0.8, 0.8, 0.8, 1]))
                self.scene_setup = True

            for body_name in self.objects:
                self.sim.physics_client.removeBody(self.sim._bodies_idx.get(body_name))
            self.objects.clear()

            self._draw_goal_square()
            self.robot.reset()

            num_objects = np.random.randint(5, 9)
            object_types = ["cube", "sphere"]
            for i in range(num_objects):
                body_name = f"object_{i}"
                object_type = np.random.choice(object_types)
                spawn_pos = self._get_safe_spawn_position(self.objects.keys())

                if object_type == "cube":
                    self.sim.create_box(
                        body_name=body_name, half_extents=np.array([0.02] * 3), mass=1.0,
                        position=spawn_pos, rgba_color=np.random.uniform(0, 1, size=4))
                elif object_type == "sphere":
                    self.sim.create_sphere(
                        body_name=body_name, radius=0.02, mass=1.0,
                        position=spawn_pos, rgba_color=np.random.uniform(0, 1, size=4))
                self.objects[body_name] = {"type": object_type}

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        a_skill, dx, dy, dz, gripper_ctrl = action[0], action[1], action[2], -0.1, 1.0
        self.robot.set_action(np.array([dx, dy, dz, gripper_ctrl]))
        self.sim.step()

        obs = self._get_obs()

        # 简化版的奖励和成功判断 (假设关心第一个物体)
        achieved_goal = self.sim.get_base_position(list(self.objects.keys())[0])
        terminated = self._is_success(achieved_goal)
        reward = -np.linalg.norm(achieved_goal[:2] - self.goal_pos)
        if terminated: reward += 10.0

        truncated = False
        info = {"is_success": bool(terminated)}
        return obs, float(reward), bool(terminated), truncated, info

    def close(self):
        self.sim.close()