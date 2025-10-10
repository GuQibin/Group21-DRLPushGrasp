import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda
import pybullet_data
import pybullet as p


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
        执行一个动作，并让环境前进一个时间步。
        """
        # 1. 解析并执行高层动作
        a_skill, dx, dy, dz, gripper_ctrl = action[0], action[1], action[2], -0.1, 1.0
        self.robot.set_action(np.array([dx, dy, dz, gripper_ctrl]))
        self.sim.step()

        # 2. 动作执行后，重新分析场景并更新物体颜色
        self._analyze_scene_states()
        self._update_object_colors()

        # 3. 获取新的观测
        obs = self._get_obs()
# TODO 详细完善所有的奖励和惩罚机制，详见proposal
        # 4. 计算奖励和成功/结束条件 (简化版逻辑)
        # 假设我们只关心第一个物体的状态
        achieved_goal = self.sim.get_base_position(list(self.objects.keys())[0])
        terminated = self._is_success(achieved_goal)
        reward = -np.linalg.norm(achieved_goal[:2] - self.goal_pos)  # 距离奖励
        if terminated:
            reward += 10.0  # 成功时给予额外奖励

        truncated = False  # 本环境中没有超时结束的条件
        info = {"is_success": bool(terminated)}

        # 5. 返回Gymnasium标准的5个值
        return obs, float(reward), bool(terminated), truncated, info

    def close(self):
        """关闭环境，释放资源。"""
        self.sim.close()