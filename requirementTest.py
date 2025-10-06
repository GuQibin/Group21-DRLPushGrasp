import gymnasium as gym
import panda_gym
import time
import numpy as np

# 1. 使用 gym.make() 创建 PandaReach-v3 环境
#    这是 panda-gym 库中推荐的最新版本。
env = gym.make("PandaReach-v3", render_mode="human")

print("--- 环境信息 ---")
print(f"环境 ID: {env.spec.id}")
print(f"动作空间: {env.action_space}")
print(f"观察空间: {env.observation_space}")
print("----------------")

# 2. 重置环境，获取初始状态观测和额外信息
observation, info = env.reset()

# 3. 循环运行回合 (例如 5 个回合)
num_episodes = 5
max_steps_per_episode = 100
current_episode = 0

while current_episode < num_episodes:

    # 4. 从动作空间中随机采样一个动作
    #    动作是一个 4 维向量 (x, y, z 移动, 抓取动作)
    action = env.action_space.sample()

    # 5. 执行动作
    #    step() 返回: observation, reward, terminated, truncated, info
    observation, reward, terminated, truncated, info = env.step(action)

    # 6. 打印关键信息
    # print(f"Episode: {current_episode}, Step: {env.unwrapped.current_step}, Reward: {reward:.4f}, Success: {info.get('is_success')}")

    # 7. 检查回合是否结束 (无论是正常结束还是被截断)
    if terminated or truncated:
        print(f"\n--- Episode {current_episode} finished. Resetting environment. ---")
        observation, info = env.reset()
        current_episode += 1

    # 为了让渲染有时间显示，可以短暂等待
    time.sleep(0.01)

# 8. 关闭环境
env.close()

print("\nPanda-gym test script finished successfully! (5 episodes completed)")