import gymnasium as gym
import envs  # 这一行非常重要，它会执行__init__.py中的注册代码


def main():
    """
    Initializes the custom environment, resets it once to show the
    initial state, and then waits for the user to exit.
    This is useful for debugging the reset() function and scene setup.
    """
    # 1. 创建环境，并开启可视化
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")

    # 2. 重置环境。这一步会创建并渲染出初始场景
    print("Resetting the environment to show the initial frame...")
    obs, info = env.reset()

    # --- 关键改动：不再执行动作循环，而是等待用户输入 ---
    print("\n" + "=" * 50)
    print("The simulation is now paused at the initial frame.")
    print("You can inspect the initial placement of the robot and objects.")
    # input() 函数会在这里暂停脚本，直到您按下回车
    input("Press Enter in this terminal to close the environment...")
    print("=" * 50 + "\n")

    # 3. 干净地关闭环境
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
