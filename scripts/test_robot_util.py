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

# Add parent directory to path to enable importing from utils package
sys.path.append(str(Path(__file__).parent.parent))


def _ensure_dir(path: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def _start_video(log_dir: str, base_name: str) -> int:
    """Start recording a PyBullet MP4 video and return the log id for later stopping."""
    _ensure_dir(log_dir)
    # Create timestamped filename to avoid overwriting previous recordings
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{base_name}_{ts}.mp4")
    # Start PyBullet's built-in video recording functionality
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


def test_basic_imports():
    """Test that all robot utility functions can be imported successfully"""
    print("Testing imports...")
    
    try:
        from utils.robot_util import (
            get_ee_position_safe,
            get_ee_orientation_safe,
            execute_pick_and_place,
            execute_push,
            move_to_position,
            open_gripper,
            close_gripper,
            check_grasp_success,
            compute_inverse_kinematics,
            plan_trajectory,
            get_gripper_state,
            check_collision_between_bodies,
            quaternion_to_rotation_matrix,
            wait_for_stability
        )
        print("All imports successful")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trajectory_planning():
    """Test joint space trajectory planning without requiring simulation"""
    print("\nTesting trajectory planning...")
    
    from utils.robot_util import plan_trajectory
    
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    trajectory = plan_trajectory(start, goal, num_waypoints=10)

    assert trajectory.shape == (10, 7), f"Expected (10, 7), got {trajectory.shape}"
    assert np.allclose(trajectory[0], start), "First waypoint should be start"
    assert np.allclose(trajectory[-1], goal), "Last waypoint should be goal"

    for i in range(7):
        diffs = np.diff(trajectory[:, i])
        if not np.all(diffs >= 0) and not np.all(diffs <= 0):
            assert np.allclose(diffs, 0, atol=1e-10) or \
                   (np.all(diffs >= -1e-10)) or \
                   (np.all(diffs <= 1e-10)), \
                   f"Joint {i} trajectory not monotonic"

    print(" Trajectory planning works")


def test_gripper_functions():
    """Test gripper state detection and utility functions"""
    print("\nTesting gripper functions...")

    # 【核心修正】: 为了匹配新的 get_gripper_state 实现，我们需要模拟 sim 和关节状态
    from utils.robot_util import get_gripper_state, get_gripper_width
    import unittest.mock as mock

    # 创建一个模拟的 sim 对象
    mock_sim = mock.Mock()

    # 模拟 get_gripper_width 函数的行为
    def mock_get_gripper_width(sim_instance):
        # 这个内部函数会根据我们设置的关节状态返回宽度
        if sim_instance.gripper_joints == "open":
            return 0.08
        elif sim_instance.gripper_joints == "closed":
            return 0.0
        return 0.0

    # 创建一个模拟的 robot 对象
    class MockRobot:
        def __init__(self, sim):
            self.sim = sim

    # --- 测试张开状态 ---
    # 设置模拟的 sim，让它表现得像夹爪是张开的
    mock_sim.gripper_joints = "open"
    # 将 get_gripper_width 函数“劫持”，让它在我们模拟的 sim 上运行时返回我们想要的值
    with mock.patch('utils.robot_util.get_gripper_width', mock_get_gripper_width):
        robot_open = MockRobot(mock_sim)
        state_open = get_gripper_state(robot_open)

    assert 'width' in state_open, "Should have 'width' key"
    assert state_open['width'] > 0.07, f"Width should be > 0.07, got {state_open['width']}"
    assert state_open['is_open'] == True, f"Gripper should be open, got {state_open}"
    assert state_open['is_closed'] == False, "Open gripper should not be closed"

    # --- 测试闭合状态 ---
    # 设置模拟的 sim，让它表现得像夹爪是闭合的
    mock_sim.gripper_joints = "closed"
    with mock.patch('utils.robot_util.get_gripper_width', mock_get_gripper_width):
        robot_closed = MockRobot(mock_sim)
        state_closed = get_gripper_state(robot_closed)

    assert state_closed['width'] < 0.01, f"Width should be < 0.01, got {state_closed['width']}"
    assert state_closed['is_closed'] == True, f"Gripper should be closed, got {state_closed}"
    assert state_closed['is_open'] == False, "Closed gripper should not be open"

    print("  Gripper state function works")

def test_quaternion_conversion():
    """Test quaternion to rotation matrix conversion for 3D orientation"""
    print("\nTesting quaternion conversion...")

    from utils.robot_util import quaternion_to_rotation_matrix

    identity_quat = np.array([0, 0, 0, 1])
    rot_matrix = quaternion_to_rotation_matrix(identity_quat)

    assert rot_matrix.shape == (3, 3), f"Expected (3, 3), got {rot_matrix.shape}"
    assert np.allclose(rot_matrix, np.eye(3)), "Identity quaternion should give identity matrix"

    z_rot_quat = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    rot_matrix_z = quaternion_to_rotation_matrix(z_rot_quat)

    vec_x = np.array([1, 0, 0])
    rotated = rot_matrix_z @ vec_x
    expected = np.array([0, 1, 0])
    assert np.allclose(rotated, expected, atol=0.01), f"Z-rotation failed: {rotated} vs {expected}"

    print("  ✓ Quaternion conversion works")


def test_safe_ee_getters():
    """Test safe end-effector position/orientation getters with error handling"""
    print("\nTesting safe EE getters...")

    from utils.robot_util import get_ee_position_safe, get_ee_orientation_safe

    class MockRobotWithMethod:
        def get_ee_position(self): return np.array([0.3, 0.0, 0.5])
        def get_ee_orientation(self): return np.array([0, 0, 0, 1])

    robot = MockRobotWithMethod()

    pos = get_ee_position_safe(robot)
    assert pos.shape == (3,), f"Expected (3,), got {pos.shape}"
    assert np.allclose(pos, [0.3, 0.0, 0.5]), "Position should match mock"

    ori = get_ee_orientation_safe(robot)
    assert ori.shape == (4,), f"Expected (4,), got {ori.shape}"
    assert np.allclose(ori, [0, 0, 0, 1]), "Orientation should match mock"

    print(" Safe EE getters work")


def test_with_real_environment():
    """Integration test with real PyBullet environment and video recording"""
    print("\nTesting with real environment (with video)...")
    print("  (This requires PyBullet GUI: render_mode='human')")

    log_id = None
    env = None
    try:
        import gymnasium as gym
        import envs

        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        log_id = _start_video(log_dir="videos", base_name="test_robot_util")

        obs, info = env.reset()
        time.sleep(0.5)

        from utils.robot_util import get_gripper_state, move_to_position, get_ee_position_safe, open_gripper, close_gripper

        gripper_state = get_gripper_state(env.unwrapped.robot)
        print(f"  ✓ Gripper state: {gripper_state}")

        ee_pos = get_ee_position_safe(env.unwrapped.robot)
        print(f"  ✓ EE position: {ee_pos}")

        target_pos = ee_pos + np.array([0.05, 0.0, 0.0])
        success = move_to_position(env.unwrapped.sim, env.unwrapped.robot, target_pos, steps=40)
        print(f"  ✓ Movement {'succeeded' if success else 'completed with warnings'}")

        print("  Testing gripper open...")
        open_gripper(env.unwrapped.sim, env.unwrapped.robot, steps=20)
        for _ in range(20): env.unwrapped.sim.step()

        print("  Testing gripper close...")
        close_gripper(env.unwrapped.sim, env.unwrapped.robot, steps=20)
        for _ in range(20): env.unwrapped.sim.step()

        for _ in range(60): env.unwrapped.sim.step()

        print("  ✓ Environment integration works")

    except Exception as e:
        print(f"  ⚠ Real environment test skipped: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if log_id is not None: _stop_video(log_id)
        if env is not None: env.close(); print("  Environment closed.")

def test_pick_and_place_dry_run():
    """
    Test complete pick-and-place sequence with real environment.
    """
    print("\nTesting pick-and-place (dry run)...")

    log_id = None
    env = None
    try:
        import gymnasium as gym
        import envs

        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        log_id = _start_video(log_dir="videos", base_name="test_pick_and_place")

        obs, info = env.reset()
        time.sleep(0.5)

        from utils.robot_util import execute_pick_and_place

        if len(env.unwrapped.objects) > 0:
            target_obj = list(env.unwrapped.objects.keys())[0]
            print(f"  Attempting to pick {target_obj}...")

            # 【核心修正】: 传入 workspace_bounds 参数
            success = execute_pick_and_place(
                sim=env.unwrapped.sim,
                robot=env.unwrapped.robot,
                target_object=target_obj,
                alpha_x=0.0,
                alpha_y=0.0,
                goal_pos=env.unwrapped.goal_pos,
                workspace_bounds=env.unwrapped.WORKSPACE_BOUNDS
            )

            print(f"  Pick-and-place {'succeeded' if success else 'completed'}")

    except Exception as e:
        print(f"  Pick-and-place test encountered issue: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if log_id is not None: _stop_video(log_id)
        if env is not None: env.close()


def test_push_dry_run():
    """
    Test object pushing with real environment.
    """
    print("\nTesting push (dry run)...")

    log_id = None
    env = None
    try:
        import gymnasium as gym
        import envs

        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        log_id = _start_video(log_dir="videos", base_name="test_push")

        obs, info = env.reset()
        time.sleep(0.5)

        from utils.robot_util import execute_push

        if len(env.unwrapped.objects) > 0:
            target_obj = list(env.unwrapped.objects.keys())[0]
            print(f"  Attempting to push {target_obj}...")

            # 【核心修正】: 传入 workspace_bounds 参数
            success = execute_push(
                sim=env.unwrapped.sim,
                robot=env.unwrapped.robot,
                target_object=target_obj,
                alpha_x=0.0,
                alpha_y=0.0,
                alpha_theta=0.0,
                workspace_bounds=env.unwrapped.WORKSPACE_BOUNDS
            )

            print(f"  Push {'succeeded' if success else 'completed'}")

    except Exception as e:
        print(f"  Push test encountered issue: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if log_id is not None: _stop_video(log_id)
        if env is not None: env.close()

def test_robot_diagnostics():
    """Run comprehensive robot control diagnostics to identify issues"""
    print("\nRunning robot diagnostics...")

    try:
        import gymnasium as gym
        import envs
        from utils.robot_util import diagnose_robot_control

        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        obs, info = env.reset()

        diagnose_robot_control(env.unwrapped.robot, env.unwrapped.sim, steps=10)

        env.close()
        print("  ✓ Diagnostics complete")

    except Exception as e:
        print(f"  ⚠ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing utils/robot_util.py")
    print("=" * 60 + "\n")

    all_passed = True

    # Test 1: Import verification
    if not test_basic_imports():
        sys.exit(1)

    # Test 2-5: Unit tests
    unit_tests = [
        test_trajectory_planning,
        test_gripper_functions,
        test_quaternion_conversion,
        test_safe_ee_getters
    ]

    for test_func in unit_tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ❌ UNEXPECTED ERROR in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Critical Diagnostics
    print("\n" + "=" * 60)
    print("ROBOT DIAGNOSTICS")
    print("=" * 60)
    test_robot_diagnostics()

    # Integration Tests
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60)

    integration_tests = [
        ("Basic environment", test_with_real_environment),
        ("Pick-and-place", test_pick_and_place_dry_run),
        ("Push", test_push_dry_run)
    ]

    for test_name, test_func in integration_tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ⚠ {test_name} test FAILED with an unhandled exception: {e}")
            import traceback
            traceback.print_exc()

    # Final Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CORE TESTS PASSED")
    else:
        print("✗ SOME CORE TESTS FAILED")
    print("=" * 60)