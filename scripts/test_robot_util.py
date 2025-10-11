"""
Test script to verify robot_utils.py works correctly
Run this ONLY after ensuring PyBullet connection works
"""

import numpy as np
import sys
from pathlib import Path

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
    Test with actual PyBullet environment.
    Only run this if PyBullet GUI works!
    """
    print("\nTesting with real environment...")
    print("  (This requires PyBullet to be working)")
    
    try:
        import gymnasium as gym
        import envs
        
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        obs, info = env.reset()
        
        from utils.robot_util import get_gripper_state
        
        # Test gripper state
        gripper_state = get_gripper_state(env.robot)
        print(f"  ✓ Gripper state: {gripper_state}")
        
        # Test basic movement
        target_pos = env.robot.get_ee_position() + np.array([0.05, 0.0, 0.0])
        
        from utils.robot_util import move_to_position
        success = move_to_position(env.sim, env.robot, target_pos, steps=20)
        
        if success:
            print("  ✓ Basic movement works")
        else:
            print("  ⚠ Movement didn't reach target (this is okay for testing)")
        
        env.close()
        print("  ✓ Environment integration works")
        
    except Exception as e:
        print(f"  ⚠ Real environment test skipped: {e}")
        print("  This is okay if PyBullet GUI isn't set up yet")


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
