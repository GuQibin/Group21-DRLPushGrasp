# scripts/demo_single_cube_env.py
"""
Demo: single-cube pick-and-place using StrategicPushAndGraspEnv.

Run:
    python -m scripts.demo_single_cube_env
    # 可选参数：
    # python -m scripts.demo_single_cube_env --cube-x 0.45 --cube-y 0.00 --goal-x 0.25 --goal-y 0.25 --alpha-x 0.0 --alpha-y 0.0
"""

import time
import numpy as np
import gymnasium as gym
import envs  # 触发注册

def main():
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")

    # —— 这里切换你想要的物体和位置 ——
    # object_type: "cube" 或 "sphere"
    # object_pos : 物体平面位置
    # goal_pos   : 奖励区域中心
    obs, info = env.reset(options={
        "single_object_demo": True,
        "object_type": "sphere",          # 改成 "sphere" 就能测试球体
        "object_pos":  [0.1, 0.1],
        #"cube_half":   0.02,            # 方体半边长（m）
        "sphere_radius": 0.02,        # 球体半径（m），若用球体时可打开
        "goal_pos":    [0.25, 0.25],
        "goal_size":   0.10
    })

    # 构造一次“抓取”动作：a_skill>0 → 调用 execute_pick_and_place()
    # alpha_x, alpha_y 是物体系偏移 ∈ [-1,1]，先从中心抓起
    action = np.array([+1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    print(">>> sending grasp action:", action.tolist())
    obs, reward, terminated, truncated, info = env.step(action)
    print(">>> reward:", reward, "| terminated:", terminated, "| truncated:", truncated)
    print(">>> info:", info)

    time.sleep(20)
    env.close()

if __name__ == "__main__":
    main()