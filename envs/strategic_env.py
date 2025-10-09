import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
import pybullet_data
import pybullet as p

from panda_gym.envs.tasks.push import Push
from panda_gym.envs.tasks.pick_and_place import PickAndPlace


class StrategicPushAndGraspEnv(RobotTaskEnv):
    def __init__(self, render_mode: str = "rgb_array", control_type: str = "ee"):
        sim = PyBullet(render_mode=render_mode)
        robot_base_position = np.array([0.4, -0.3, 0.0])
        robot = Panda(sim, block_gripper=True, control_type=control_type, base_position=robot_base_position)

        self.robot = robot
        self.sim = sim
        self.objects_body_names = []
        self.scene_setup = False

        # 定义目标区域的中心和大小
        self.goal_pos = np.array([-0.2, -0.2])  # 方形区域的中心点
        self.goal_size = 0.1  # 方形区域的边长

        # 真实的逻辑任务也在这里创建
        self.task_push = Push(sim)
        self.task_pick_and_place = PickAndPlace(sim)

        pseudo_task = self.task_pick_and_place
        super().__init__(self.robot, pseudo_task)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def _draw_goal_square(self):
        """使用PyBullet的调试工具在目标位置画一个绿色的方形轮廓"""
        half_size = self.goal_size / 2
        color = [0, 1, 0]  # Green
        z_offset = 0.001  # 略高于桌面

        # 定义正方形的四个角点
        center_x, center_y = self.goal_pos
        p1 = [center_x - half_size, center_y - half_size, z_offset]
        p2 = [center_x + half_size, center_y - half_size, z_offset]
        p3 = [center_x + half_size, center_y + half_size, z_offset]
        p4 = [center_x - half_size, center_y + half_size, z_offset]

        # 用四条线连接角点
        client = self.sim.physics_client
        client.addUserDebugLine(p1, p2, lineColorRGB=color, lineWidth=3)
        client.addUserDebugLine(p2, p3, lineColorRGB=color, lineWidth=3)
        client.addUserDebugLine(p3, p4, lineColorRGB=color, lineWidth=3)
        client.addUserDebugLine(p4, p1, lineColorRGB=color, lineWidth=3)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        with self.sim.no_rendering():
            for body_name in self.objects_body_names:
                body_id = self.sim._bodies_idx.get(body_name)
                if body_id is not None: self.sim.physics_client.removeBody(body_id)
            self.objects_body_names.clear()

            if not self.scene_setup:
                self.sim.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
                self.sim.loadURDF(body_name="plane", fileName="plane.urdf", basePosition=[0, 0, -0.001])
                self.scene_setup = True

            self._draw_goal_square()

            self.robot.reset()

            num_objects = np.random.randint(2, 5)
            for i in range(num_objects):
                body_name = f"object_{i}"
                self.sim.create_box(
                    body_name=body_name, half_extents=np.array([0.02, 0.02, 0.02]), mass=1.0,
                    position=np.random.uniform([-0.2, -0.2, 0.02], [0.1, 0.2, 0.02]),
                    rgba_color=np.random.uniform(0, 1, size=4),
                )
                self.objects_body_names.append(body_name)

        self.task_push.goal = self.goal_pos
        self.task_pick_and_place.goal = self.goal_pos
        observation = self._get_obs()
        info = {"is_success": False}
        return observation, info

    def _is_success(self, achieved_goal: np.ndarray) -> bool:
        """判断物体是否在目标方形区域内"""
        half_size = self.goal_size / 2
        # achieved_goal 是物体的 (x, y, z) 坐标
        # 我们只关心 x 和 y
        object_xy = achieved_goal[:2]

        # 判断物体的 xy 坐标是否都在方形区域的边界内
        return (
                (self.goal_pos[0] - half_size < object_xy[0] < self.goal_pos[0] + half_size) and
                (self.goal_pos[1] - half_size < object_xy[1] < self.goal_pos[1] + half_size)
        )

    def step(self, action: np.ndarray):
        a_skill = action[0]
        dx, dy, dz, gripper_ctrl = action[1], action[2], -0.1, 1.0
        low_level_action = np.array([dx, dy, dz, gripper_ctrl])
        self.robot.set_action(low_level_action)
        self.sim.step()

        observation = self._get_obs()

        # achieved_goal 通常是其中一个物体的位置，由父类的 _get_obs 决定
        achieved_goal = observation["achieved_goal"]

        terminated = self._is_success(achieved_goal)

        # 我们可以继续使用 panda-gym 的奖励函数，因为它通常是基于到目标点距离的
        # 这对于引导 agent 仍然是有效的
        desired_goal = self.goal_pos
        reward = self.task_push.compute_reward(achieved_goal, desired_goal, {})

        # 如果成功，可以给予一个额外的巨大奖励
        if terminated:
            reward += 10.0  # 成功奖励

        truncated = False
        info = {"is_success": bool(terminated)}
        return observation, float(reward), bool(terminated), truncated, info