import gymnasium as gym
import envs  # 这一行非常重要，它会执行__init__.py中的注册代码


def main():
    # 使用 render_mode="human" 来开启可视化
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")

    # reset 返回 obs 和 info
    obs, info = env.reset()

    print("Starting test with random actions...")
    for _ in range(1000):
        action = env.action_space.sample()

        # step 返回5个值
        obs, reward, terminated, truncated, info = env.step(action)

        # 'done' 现在是 terminated 或 truncated
        done = terminated or truncated

        if done:
            obs, info = env.reset()

    env.close()
    print("Test finished.")


if __name__ == "__main__":
    main()