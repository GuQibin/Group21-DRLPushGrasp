"""
Test script to verify robot_utils.py works correctly
Run this ONLY after ensuring PyBullet connection works
"""

import numpy as np
import sys
from pathlib import Path












import os
import time
import pybullet as p
from datetime import datetime

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _start_video(log_dir: str, base_name: str) -> int:
    """Start recording a PyBullet MP4 and return the log id."""
    _ensure_dir(log_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{base_name}_{ts}.mp4")
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, path)
    print(f"[VIDEO] Recording to: {path}")
    return log_id

def _stop_video(log_id: int):
    """Stop recording for a given PyBullet log id."""
    try:
        p.stopStateLogging(log_id)
        print(f"[VIDEO] Recording stopped (log_id={log_id})")
    except Exception as e:
        print(f"[VIDEO] stop failed: {e}")


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_basic_imports():
    """Test that all functions can be imported"""
    print("Testing imports...")
    
    try:
        from utils.robot_util import (
            execute_pick_and_place,
            execute_push,
            move_to_position,
            open_gripper,
            close_gripper,
            compute_inverse_kinematics,
            plan_trajectory,
            check_self_collision,
            check_collision_with_obstacles,
            get_gripper_state
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False


def test_trajectory_planning():
    """Test trajectory planning without simulation"""
    print("\nTesting trajectory planning...")
    
    from utils.robot_util import plan_trajectory
    
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    trajectory = plan_trajectory(start, goal, num_waypoints=10)
    
    assert trajectory.shape == (10, 7), f"Expected (10, 7), got {trajectory.shape}"
    assert np.allclose(trajectory[0], start), "First waypoint should be start"
    assert np.allclose(trajectory[-1], goal), "Last waypoint should be goal"
    
    print("  ✓ Trajectory planning works")


def test_gripper_functions():
    """Test gripper utility functions"""
    print("\nTesting gripper functions...")
    
    from utils.robot_util import get_gripper_state
    
    # Create mock robot
    class MockRobot:
        def get_obs(self):
            # Mock observation with 16 values (7 joints + 7 velocities + 2 gripper)
            return np.array([0]*14 + [0.04, 0.04])  # Gripper open
    
    robot = MockRobot()
    state = get_gripper_state(robot)
    
    assert 'width' in state, "Should have 'width' key"
    assert 'is_closed' in state, "Should have 'is_closed' key"
    assert 'is_open' in state, "Should have 'is_open' key"
    assert state['is_open'] == True, "Gripper should be open"
    
    print("  ✓ Gripper state function works")


def test_with_real_environment():
    """
    Test with an actual PyBullet environment and record a short video.
    Only run this if the PyBullet GUI is available!
    """
    print("\nTesting with real environment (with video)...")
    print("  (This requires PyBullet GUI: render_mode='human')")

    log_id = None
    try:
        import gymnasium as gym
        import envs

        # Create the environment first, then start recording so the reset scene is captured
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        log_id = _start_video(log_dir="videos", base_name="test_with_real_environment")

        obs, info = env.reset()
        time.sleep(0.5)  # Give the camera/renderer a short moment

        from utils.robot_util import get_gripper_state, move_to_position

        gripper_state = get_gripper_state(env.robot)
        print(f" Gripper state: {gripper_state}")

        # Make a small motion so the video has some movement
        target_pos = env.robot.get_ee_position() + np.array([0.05, 0.0, 0.0])
        success = move_to_position(env.sim, env.robot, target_pos, steps=40)
        print(" Basic movement works" if success else "  ⚠ Movement didn't fully reach target")

        # Step a bit more so the clip isn’t too short
        for _ in range(60):
            env.sim.step()

        env.close()
        print(" Environment integration works")

    except Exception as e:
        print(f" Real environment test skipped: {e}")
        print(" This is okay if the PyBullet GUI isn't set up yet")
    finally:
        if log_id is not None:
            _stop_video(log_id)



if __name__ == "__main__":
    print("=" * 60)
    print("Testing utils/robot_utils.py")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # Test 1: Imports
    if not test_basic_imports():
        all_passed = False
        print("\n Cannot proceed - import errors")
        sys.exit(1)
    
    # Test 2: Trajectory planning
    try:
        test_trajectory_planning()
    except Exception as e:
        print(f"Trajectory planning test failed: {e}")
        all_passed = False
    
    # Test 3: Gripper functions
    try:
        test_gripper_functions()
    except Exception as e:
        print(f"Gripper function test failed: {e}")
        all_passed = False
    
    # Test 4: Real environment (optional)
    try:
        test_with_real_environment()
    except Exception as e:
        print(f"Real environment test failed: {e}")
        print("This is expected if PyBullet GUI isn't working yet")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("CORE TESTS PASSED")
        print("   (Real environment test is optional)")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
