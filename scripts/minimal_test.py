import pybullet as p
import time

try:
    # Option A: Attempt default connection (this might fail)
    # print("Attempting default GUI connection...")
    # p.connect(p.GUI)

    # Option B: Force OpenGL 2 usage (this is much more likely to succeed)
    print("Attempting GUI connection with --opengl2 option...")
    p.connect(p.GUI, options="--opengl2")

    print("\nConnection successful! The window should be visible.")
    print("If you see this message and a window, the fix is confirmed.")

    # Keep the window open for a few seconds to verify it's working
    time.sleep(5)
    
    # Properly disconnect from the physics server
    p.disconnect()

except Exception as e:
    print("\nConnection failed.")
    print(e)
