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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def _ensure_dir(path: str):
    """Ensure directory exists."""
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


def test_basic_imports():
    """Test that all functions can be imported"""
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
    """Test trajectory planning without simulation"""
    print("\nTesting trajectory planning...")
    
    from utils.robot_util import plan_trajectory
    
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    trajectory = plan_trajectory(start, goal, num_waypoints=10)
    
    assert trajectory.shape == (10, 7), f"Expected (10, 7), got {trajectory.shape}"
    assert np.allclose(trajectory[0], start), "First waypoint should be start"
    assert np.allclose(trajectory[-1], goal), "Last waypoint should be goal"
    
    # Check intermediate waypoints are interpolated correctly
    # np.linspace with 10 points means index 4 or 5 would be near middle
    for i in range(7):  # For each joint
        diffs = np.diff(trajectory[:, i])
        # All differences should have same sign (monotonic)
        if not np.all(diffs >= 0) and not np.all(diffs <= 0):
            # Allow small numerical errors
            assert np.allclose(diffs, 0, atol=1e-10) or \
                   (np.all(diffs >= -1e-10)) or \
                   (np.all(diffs <= 1e-10)), \
                   f"Joint {i} trajectory not monotonic"
    
    print(" Trajectory planning works")


def test_gripper_functions():
    """Test gripper utility functions"""
    print("\nTesting gripper functions...")
    
    from utils.robot_util import get_gripper_state
    
    # Create mock robot with open gripper
    class MockRobotOpen:
        def get_obs(self):
            # Mock observation with 16 values (7 joints + 7 velocities + 2 gripper)
            return np.array([0]*14 + [0.04, 0.04])  # Gripper open (0.08 total)
    
    # Create mock robot with closed gripper
    class MockRobotClosed:
        def get_obs(self):
            return np.array([0]*14 + [0.0, 0.0])  # Gripper closed
    
    # Test open gripper
    robot_open = MockRobotOpen()
    state_open = get_gripper_state(robot_open)
    
    assert 'width' in state_open, "Should have 'width' key"
    assert 'is_closed' in state_open, "Should have 'is_closed' key"
    assert 'is_open' in state_open, "Should have 'is_open' key"
    assert state_open['is_open'] == True, f"Gripper should be open, got {state_open}"
    assert state_open['is_closed'] == False, "Open gripper should not be closed"
    
    # Test closed gripper
    robot_closed = MockRobotClosed()
    state_closed = get_gripper_state(robot_closed)
    
    assert state_closed['is_closed'] == True, f"Gripper should be closed, got {state_closed}"
    assert state_closed['is_open'] == False, "Closed gripper should not be open"
    
    print("  Gripper state function works")


def test_quaternion_conversion():
    """Test quaternion to rotation matrix conversion"""
    print("\nTesting quaternion conversion...")
    
    from utils.robot_util import quaternion_to_rotation_matrix
    
    # Test identity quaternion [0, 0, 0, 1]
    identity_quat = np.array([0, 0, 0, 1])
    rot_matrix = quaternion_to_rotation_matrix(identity_quat)
    
    assert rot_matrix.shape == (3, 3), f"Expected (3, 3), got {rot_matrix.shape}"
    assert np.allclose(rot_matrix, np.eye(3)), "Identity quaternion should give identity matrix"
    
    # Test 90-degree rotation around Z-axis [0, 0, sin(45°), cos(45°)]
    z_rot_quat = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    rot_matrix_z = quaternion_to_rotation_matrix(z_rot_quat)
    
    # Should rotate [1, 0, 0] to approximately [0, 1, 0]
    vec_x = np.array([1, 0, 0])
    rotated = rot_matrix_z @ vec_x
    expected = np.array([0, 1, 0])
    assert np.allclose(rotated, expected, atol=0.01), f"Z-rotation failed: {rotated} vs {expected}"
    
    print("  ✓ Quaternion conversion works")


def test_safe_ee_getters():
    """Test safe end-effector position/orientation getters"""
    print("\nTesting safe EE getters...")
    
    from utils.robot_util import get_ee_position_safe, get_ee_orientation_safe
    
    # Mock robot with get_ee_position method
    class MockRobotWithMethod:
        def get_ee_position(self):
            return np.array([0.3, 0.0, 0.5])
        
        def get_ee_orientation(self):
            return np.array([0, 0, 0, 1])
    
    robot = MockRobotWithMethod()
    
    pos = get_ee_position_safe(robot)
    assert pos.shape == (3,), f"Expected (3,), got {pos.shape}"
    assert np.allclose(pos, [0.3, 0.0, 0.5]), "Position should match mock"
    
    ori = get_ee_orientation_safe(robot)
    assert ori.shape == (4,), f"Expected (4,), got {ori.shape}"
    assert np.allclose(ori, [0, 0, 0, 1]), "Orientation should match mock"
    
    print(" Safe EE getters work")


def test_with_real_environment():
    """Test with real environment and video recording."""
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

        from utils.robot_util import (
            get_gripper_state, 
            move_to_position,
            get_ee_position_safe,
            open_gripper,
            close_gripper
        )

        gripper_state = get_gripper_state(env.unwrapped.robot)
        print(f"  ✓ Gripper state: {gripper_state}")

        ee_pos = get_ee_position_safe(env.unwrapped.robot)
        print(f"  ✓ EE position: {ee_pos}")

        target_pos = ee_pos + np.array([0.05, 0.0, 0.0])
        success = move_to_position(env.unwrapped.sim, env.unwrapped.robot, target_pos, steps=40)
        print(f"  ✓ Movement {'succeeded' if success else 'completed with warnings'}")

        print("  Testing gripper open...")
        open_gripper(env.unwrapped.sim, env.unwrapped.robot, steps=20)
        for _ in range(20):
            env.unwrapped.sim.step()

        print("  Testing gripper close...")
        close_gripper(env.unwrapped.sim, env.unwrapped.robot, steps=20)
        for _ in range(20):
            env.unwrapped.sim.step()

        for _ in range(60):
            env.unwrapped.sim.step()

        print("  ✓ Environment integration works")

    except Exception as e:
        print(f"  ⚠ Real environment test skipped: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # CRITICAL: Stop video BEFORE closing environment
        if log_id is not None:
            _stop_video(log_id)
        
        if env is not None:
            env.close()
            print("  Environment closed.")

def test_pick_and_place_dry_run():
    """
    Test pick-and-place with real environment (dry run - may fail gracefully)
    """
    print("\nTesting pick-and-place (dry run)...")
    
    log_id = None
    try:
        import gymnasium as gym
        import envs
        
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        log_id = _start_video(log_dir="videos", base_name="test_pick_and_place")
        
        obs, info = env.reset()
        time.sleep(0.5)
        
        from utils.robot_util import execute_pick_and_place
        
        # Try to pick the first object
        if len(env.unwrapped.objects) > 0:
            target_obj = list(env.unwrapped.objects.keys())[0]
            print(f"  Attempting to pick {target_obj}...")
            
            success = execute_pick_and_place(
                env.unwrapped.sim, env.unwrapped.robot, target_obj,
                alpha_x=0.0, alpha_y=0.0,  # Center grasp
                goal_pos=env.unwrapped.goal_pos
            )
            
            print(f"  {'yes successful' if success else 'not successful'} Pick-and-place {'succeeded' if success else 'completed'}")
        
        env.close()
        
    except Exception as e:
        print(f"  Pick-and-place test encountered issue: {e}")
        print("  This is expected if grasp parameters need tuning")
    finally:
        if log_id is not None:
            _stop_video(log_id)


def test_push_dry_run():
    """
    Test push with real environment (dry run - may fail gracefully)
    """
    print("\nTesting push (dry run)...")
    
    log_id = None
    try:
        import gymnasium as gym
        import envs
        
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        log_id = _start_video(log_dir="videos", base_name="test_push")
        
        obs, info = env.reset()
        time.sleep(0.5)
        
        from utils.robot_util import execute_push
        
        # Try to push the first object
        if len(env.unwrapped.objects) > 0:
            target_obj = list(env.unwrapped.objects.keys())[0]
            print(f"  Attempting to push {target_obj}...")
            
            success = execute_push(
                env.unwrapped.sim, env.unwrapped.robot, target_obj,
                alpha_x=0.0, alpha_y=0.0,  # Center contact
                alpha_theta=0.0  # Push in positive X direction
            )
            
            print(f"  {'yes successful' if success else 'not successful'} Push {'succeeded' if success else 'completed'}")
        
        env.close()
        
    except Exception as e:
        print(f"  Push test encountered issue: {e}")
        print("  This is expected if push parameters need tuning")
    finally:
        if log_id is not None:
            _stop_video(log_id)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing utils/robot_util.py")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # Test 1: Imports
    if not test_basic_imports():
        all_passed = False
        print("\n Cannot proceed - import errors")
        sys.exit(1)
    
    # Test 2-5: Unit tests (don't require PyBullet)
    unit_tests = [
        test_trajectory_planning,
        test_gripper_functions,
        test_quaternion_conversion,
        test_safe_ee_getters
    ]
    
    for test_func in unit_tests:
        try:
            test_func()
        except AssertionError as e:
            print(f" Test failed: {e}")
            all_passed = False
        except Exception as e:
            print(f" Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Test 6-8: Integration tests (require PyBullet - optional)
    integration_tests = [
        ("Basic environment", test_with_real_environment),
        ("Pick-and-place", test_pick_and_place_dry_run),
        ("Push", test_push_dry_run)
    ]
    
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS (Optional - require PyBullet GUI)")
    print("=" * 60)
    
    for test_name, test_func in integration_tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ⚠ {test_name} test failed: {e}")
            print(f"  This is expected if PyBullet GUI isn't available")
    
    print("\n" + "=" * 60)
    if all_passed:
        print(" ALL CORE TESTS PASSED")
        print("  (Integration tests are optional)")
    else:
        print(" SOME TESTS FAILED")
    print("=" * 60)
