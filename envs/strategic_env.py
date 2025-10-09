import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.envs.tasks.push import Push
from panda_gym.envs.tasks.pick_and_place import PickAndPlace


# 我们的主环境，继承自 panda_gym 的 RobotTaskEnv
class StrategicPushAndGraspEnv(RobotTaskEnv):
    def __init__(self, render_mode: str = "rgb_array", control_type: str = "ee"):
        # 1. 创建共享的模拟器和机器人
        # 注意：这里我们让Panda的夹爪默认不锁定，因为抓取任务需要它
        sim = PyBullet(render_mode=render_mode)
        robot_base_position = np.array([0.4, -0.3, 0.0])
        robot = Panda(sim, block_gripper=False, control_type=control_type, base_position=robot_base_position)

        # 2. 创建一个“逻辑”任务实例，用于初始化父类和获取观测空间结构
        #    我们用 PickAndPlace，因为它有更完整的状态
        self.default_task = PickAndPlace(sim)

        # 3. 创建两个“策略”任务，用于在step中计算奖励和成功条件
        self.task_push = Push(sim)
        self.task_pick_and_place = PickAndPlace(sim)

        # 4. 使用默认任务来初始化父类
        super().__init__(robot, self.default_task)

        # 5. 覆盖掉父类的动作空间，使用我们自己的4D策略空间
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # ‼️ 我们不再需要自定义的_get_obs, reset, close, _setup_scene 等方法
        #    因为父类 RobotTaskEnv 已经为我们处理好了！

    # 我们唯一需要重写的就是 step 方法
    def step(self, action: np.ndarray):
        # --- 动作转换逻辑 ---
        # action: (a_skill, a_x, a_y, a_theta)
        a_skill = action[0]

        # 将 a_x, a_y 转换为底层动作 (dx, dy, dz, gripper)
        dx, dy = action[1], action[2]
        dz = -0.1  # 默认稍微向下

        # 根据技能决定夹爪的开合
        if a_skill > 0:  # PickAndPlace 逻辑
            gripper_ctrl = -1.0  # 尝试闭合
        else:  # Push 逻辑
            # 推的时候，我们希望夹爪是锁定的，但Panda机器人不能动态锁定
            # 所以我们让它保持张开
            gripper_ctrl = 1.0

            # Panda 机器人的底层动作是 (dx, dy, dz, gripper_ctrl)
        low_level_action = np.array([dx, dy, dz, gripper_ctrl])
        self.robot.set_action(low_level_action)
        self.sim.step()

        observation = self._get_obs()  # 使用父类的 _get_obs 方法

        # --- 根据 a_skill 选择奖励计算方式 ---
        achieved_goal = observation["achieved_goal"]
        desired_goal = observation["desired_goal"]

        # ‼️ 关键改动：调用 is_success 时不再传入 info
        if a_skill > 0:  # 使用 PickAndPlace 的奖励逻辑
            reward = self.task_pick_and_place.compute_reward(achieved_goal, desired_goal, {})
            terminated = self.task_pick_and_place.is_success(achieved_goal, desired_goal)
        else:  # 使用 Push 的奖励逻辑
            reward = self.task_push.compute_reward(achieved_goal, desired_goal, {})
            terminated = self.task_push.is_success(achieved_goal, desired_goal)

        truncated = False
        info = {"is_success": bool(terminated)}

        return observation, float(reward), bool(terminated), truncated, info