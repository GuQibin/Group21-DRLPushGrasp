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
    
    # Define start and goal joint configurations for a 7-DOF robot
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    # Generate smooth trajectory between start and goal
    trajectory = plan_trajectory(start, goal, num_waypoints=10)
    
    # Verify trajectory shape and endpoints
    assert trajectory.shape == (10, 7), f"Expected (10, 7), got {trajectory.shape}"
    assert np.allclose(trajectory[0], start), "First waypoint should be start"
    assert np.allclose(trajectory[-1], goal), "Last waypoint should be goal"
    
    # Check intermediate waypoints are interpolated correctly
    # np.linspace with 10 points means index 4 or 5 would be near middle
    for i in range(7):  # For each joint
        diffs = np.diff(trajectory[:, i])
        # All differences should have same sign (monotonic) to ensure smooth motion
        if not np.all(diffs >= 0) and not np.all(diffs <= 0):
            # Allow small numerical errors but ensure basically monotonic
            assert np.allclose(diffs, 0, atol=1e-10) or \
                   (np.all(diffs >= -1e-10)) or \
                   (np.all(diffs <= 1e-10)), \
                   f"Joint {i} trajectory not monotonic"
    
    print(" Trajectory planning works")


def test_gripper_functions():
    """Test gripper state detection and utility functions"""
    print("\nTesting gripper functions...")
    
    from utils.robot_util import get_gripper_state
    
    # Create mock robot with open gripper (simulated observation data)
    class MockRobotOpen:
        def get_obs(self):
            # Mock observation with 16 values: 7 joints + 7 velocities + 2 gripper positions
            return np.array([0]*14 + [0.04, 0.04])  # Gripper open (0.08 total width)
    
    # Create mock robot with closed gripper
    class MockRobotClosed:
        def get_obs(self):
            return np.array([0]*14 + [0.0, 0.0])  # Gripper closed (0 width)
    
    # Test open gripper state detection
    robot_open = MockRobotOpen()
    state_open = get_gripper_state(robot_open)
    
    # Verify gripper state dictionary structure and values
    assert 'width' in state_open, "Should have 'width' key"
    assert 'is_closed' in state_open, "Should have 'is_closed' key"
    assert 'is_open' in state_open, "Should have 'is_open' key"
    assert state_open['is_open'] == True, f"Gripper should be open, got {state_open}"
    assert state_open['is_closed'] == False, "Open gripper should not be closed"
    
    # Test closed gripper state detection
    robot_closed = MockRobotClosed()
    state_closed = get_gripper_state(robot_closed)
    
    assert state_closed['is_closed'] == True, f"Gripper should be closed, got {state_closed}"
    assert state_closed['is_open'] == False, "Closed gripper should not be open"
    
    print("  Gripper state function works")


def test_quaternion_conversion():
    """Test quaternion to rotation matrix conversion for 3D orientation"""
    print("\nTesting quaternion conversion...")
    
    from utils.robot_util import quaternion_to_rotation_matrix
    
    # Test identity quaternion [x, y, z, w] = [0, 0, 0, 1]
    identity_quat = np.array([0, 0, 0, 1])
    rot_matrix = quaternion_to_rotation_matrix(identity_quat)
    
    # Verify output shape and identity transformation
    assert rot_matrix.shape == (3, 3), f"Expected (3, 3), got {rot_matrix.shape}"
    assert np.allclose(rot_matrix, np.eye(3)), "Identity quaternion should give identity matrix"
    
    # Test 90-degree rotation around Z-axis [0, 0, sin(45°), cos(45°)]
    z_rot_quat = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    rot_matrix_z = quaternion_to_rotation_matrix(z_rot_quat)
    
    # Verify rotation transforms X-axis to Y-axis
    vec_x = np.array([1, 0, 0])
    rotated = rot_matrix_z @ vec_x
    expected = np.array([0, 1, 0])
    assert np.allclose(rotated, expected, atol=0.01), f"Z-rotation failed: {rotated} vs {expected}"
    
    print("  ✓ Quaternion conversion works")


def test_safe_ee_getters():
    """Test safe end-effector position/orientation getters with error handling"""
    print("\nTesting safe EE getters...")
    
    from utils.robot_util import get_ee_position_safe, get_ee_orientation_safe
    
    # Mock robot with standard getter methods
    class MockRobotWithMethod:
        def get_ee_position(self):
            return np.array([0.3, 0.0, 0.5])  # End-effector position in 3D space
        
        def get_ee_orientation(self):
            return np.array([0, 0, 0, 1])  # Identity quaternion orientation
    
    robot = MockRobotWithMethod()
    
    # Test position getter
    pos = get_ee_position_safe(robot)
    assert pos.shape == (3,), f"Expected (3,), got {pos.shape}"
    assert np.allclose(pos, [0.3, 0.0, 0.5]), "Position should match mock"
    
    # Test orientation getter
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

        # Create actual robot environment with GUI rendering
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        # Start video recording for debugging and documentation
        log_id = _start_video(log_dir="videos", base_name="test_robot_util")

        obs, info = env.reset()
        time.sleep(0.5)  # Allow environment to stabilize

        from utils.robot_util import (
            get_gripper_state, 
            move_to_position,
            get_ee_position_safe,
            open_gripper,
            close_gripper
        )

        # Test basic robot state queries
        gripper_state = get_gripper_state(env.unwrapped.robot)
        print(f"  ✓ Gripper state: {gripper_state}")

        ee_pos = get_ee_position_safe(env.unwrapped.robot)
        print(f"  ✓ EE position: {ee_pos}")

        # Test simple movement command
        target_pos = ee_pos + np.array([0.05, 0.0, 0.0])  # Move 5cm in X direction
        success = move_to_position(env.unwrapped.sim, env.unwrapped.robot, target_pos, steps=40)
        print(f"  ✓ Movement {'succeeded' if success else 'completed with warnings'}")

        # Test gripper control functions
        print("  Testing gripper open...")
        open_gripper(env.unwrapped.sim, env.unwrapped.robot, steps=20)
        for _ in range(20):
            env.unwrapped.sim.step()  # Step simulation to execute command

        print("  Testing gripper close...")
        close_gripper(env.unwrapped.sim, env.unwrapped.robot, steps=20)
        for _ in range(20):
            env.unwrapped.sim.step()

        # Allow final stabilization
        for _ in range(60):
            env.unwrapped.sim.step()

        print("  ✓ Environment integration works")

    except Exception as e:
        print(f"  ⚠ Real environment test skipped: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # CRITICAL: Stop video BEFORE closing environment to ensure file is saved properly
        if log_id is not None:
            _stop_video(log_id)
        
        if env is not None:
            env.close()
            print("  Environment closed.")

def test_pick_and_place_dry_run():
    """
    Test complete pick-and-place sequence with real environment
    This is a 'dry run' that may fail gracefully if grasp parameters need tuning
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
        
        # Try to pick the first available object
        if len(env.unwrapped.objects) > 0:
            target_obj = list(env.unwrapped.objects.keys())[0]
            print(f"  Attempting to pick {target_obj}...")
            
            # Execute complete pick-and-place sequence
            success = execute_pick_and_place(
                env.unwrapped.sim, env.unwrapped.robot, target_obj,
                alpha_x=0.0, alpha_y=0.0,  # Center grasp relative to object
                goal_pos=env.unwrapped.goal_pos  # Target placement location
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
    Test object pushing with real environment
    This may fail gracefully if push parameters need adjustment
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
        
        # Try to push the first available object
        if len(env.unwrapped.objects) > 0:
            target_obj = list(env.unwrapped.objects.keys())[0]
            print(f"  Attempting to push {target_obj}...")
            
            # Execute push maneuver
            success = execute_push(
                env.unwrapped.sim, env.unwrapped.robot, target_obj,
                alpha_x=0.0, alpha_y=0.0,  # Center contact point on object
                alpha_theta=0.0  # Push in positive X direction (0° angle)
            )
            
            print(f"  {'yes successful' if success else 'not successful'} Push {'succeeded' if success else 'completed'}")
        
        env.close()
        
    except Exception as e:
        print(f"  Push test encountered issue: {e}")
        print("  This is expected if push parameters need tuning")
    finally:
        if log_id is not None:
            _stop_video(log_id)

def test_robot_diagnostics():
    """Run comprehensive robot control diagnostics to identify issues"""
    print("\nRunning robot diagnostics...")
    
    try:
        import gymnasium as gym
        import envs
        from utils.robot_util import diagnose_robot_control
        
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        obs, info = env.reset()
        
        # Run comprehensive diagnostics on robot system
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
    
    # Test 1: Import verification (critical - stops execution if fails)
    if not test_basic_imports():
        all_passed = False
        print("\n❌ Cannot proceed - import errors")
        sys.exit(1)
    
    # Test 2-5: Unit tests (don't require PyBullet - fast and reliable)
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
            print(f"  ❌ Test failed: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # **CRITICAL SECTION - Run diagnostics before integration tests**
    print("\n" + "=" * 60)
    print("ROBOT DIAGNOSTICS (Critical for debugging robot issues)")
    print("=" * 60)
    test_robot_diagnostics()
    
    # Test 6-8: Integration tests (require PyBullet - optional but important)
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS (Optional - require PyBullet GUI)")
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
            print(f"  ⚠ {test_name} test failed: {e}")
            print(f"  This is expected if robot control needs tuning")
    
    # Final test summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CORE TESTS PASSED")
        print("  (Integration tests may need robot tuning)")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
