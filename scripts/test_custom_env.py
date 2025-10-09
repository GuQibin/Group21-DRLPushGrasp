import gymnasium as gym
import envs  # 这一行非常重要，它会执行__init__.py中的注册代码
import time


def main():
    """
    Initializes the custom environment and runs it indefinitely with random actions.
    The user can observe the simulation and close it by pressing Ctrl+C.
    """
    # 使用 render_mode="human" 来开启可视化
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")

    # reset 返回 obs 和 info
    obs, info = env.reset()

    print("\n" + "=" * 50)
    print("Starting test with random actions...")
    print("The simulation will run indefinitely.")
    print("‼To exit, please click on the terminal and press Ctrl+C.")
    print("=" * 50 + "\n")

    try:
        while True:
            # 从我们自定义的4D动作空间中采样一个随机动作
            action = env.action_space.sample()

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            # done 现在是 terminated 或 truncated
            done = terminated or truncated

            if done:
                print("Episode finished. Resetting environment.")
                obs, info = env.reset()

            # 短暂休眠，让渲染更平滑，也降低CPU占用
            # 注意：物理模拟的步进由env.step()内部处理，这里只是为了视觉效果
            time.sleep(1. / 240.)

    except KeyboardInterrupt:
        # 当用户在终端按下 Ctrl+C 时，程序会捕捉到这个异常
        print("\nInterrupted by user. Closing environment...")

    finally:
        # 无论如何，最后都要确保环境被干净地关闭
        env.close()
        print("Test finished.")


if __name__ == "__main__":
    main()