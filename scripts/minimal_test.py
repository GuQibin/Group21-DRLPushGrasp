import pybullet as p
import time

try:
    # 方案A: 尝试默认连接 (这可能会失败)
    # print("Attempting default GUI connection...")
    # p.connect(p.GUI)

    # 方案B: 尝试强制使用OpenGL 2 (这很可能会成功)
    print("Attempting GUI connection with --opengl2 option...")
    p.connect(p.GUI, options="--opengl2")

    print("\nConnection successful! The window should be visible.")
    print("If you see this message and a window, the fix is confirmed.")

    # 让窗口保持开启几秒钟
    time.sleep(5)
    p.disconnect()

except Exception as e:
    print("\nConnection failed.")
    print(e)
