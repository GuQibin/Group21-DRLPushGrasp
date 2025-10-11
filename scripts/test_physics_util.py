"""
Test script to verify physics_utils.py works correctly
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test all imports work"""
    print("Testing imports...")
    
    try:
        from utils.physics_utils import (
            check_workspace_violation,
            check_collision_with_table,
            check_object_collision,
            get_contact_force,
            is_object_stable,
            get_object_bounding_box,
            compute_object_volume,
            set_object_color,
            apply_force_to_object,
            reset_object_pose,
            set_gravity,
            get_all_collisions,
            check_multiple_workspace_violations,
            get_object_velocity_magnitude
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_workspace_violation_logic():
    """Test workspace violation logic with mock data"""
    print("\nTesting workspace violation logic...")
    
    from utils.physics_utils import check_workspace_violation
    
    # Mock sim
    class MockSim:
        def __init__(self):
            self.positions = {
                'obj_inside': [0.0, 0.0, 0.05],
                'obj_outside_x': [0.5, 0.0, 0.05],
                'obj_outside_y': [0.0, 0.5, 0.05],
                'obj_fell': [0.0, 0.0, -0.1]
            }
        
        def get_base_position(self, name):
            return self.positions.get(name, [0, 0, 0])
    
    sim = MockSim()
    bounds = (-0.3, 0.3, -0.3, 0.3)  # x_min, x_max, y_min, y_max
    
    # Test cases
    assert check_workspace_violation(sim, 'obj_inside', bounds) == False, "Inside object shouldn't violate"
    assert check_workspace_violation(sim, 'obj_outside_x', bounds) == True, "Outside X should violate"
    assert check_workspace_violation(sim, 'obj_outside_y', bounds) == True, "Outside Y should violate"
    assert check_workspace_violation(sim, 'obj_fell', bounds) == True, "Fallen object should violate"
    assert check_workspace_violation(sim, 'nonexistent', bounds) == True, "Missing object should violate"
    
    print("  ✓ Workspace violation logic correct")


def test_stability_check():
    """Test object stability checking"""
    print("\nTesting stability check...")
    
    from utils.physics_utils import is_object_stable
    
    # Mock sim
    class MockSim:
        def __init__(self):
            self.velocities = {
                'stable': ([0.001, 0.001, 0.0], [0.01, 0.01, 0.0]),
                'moving': ([0.5, 0.0, 0.0], [0.0, 0.0, 0.0]),
                'rotating': ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            }
        
        def get_base_velocity(self, name):
            return self.velocities.get(name, ([0, 0, 0], [0, 0, 0]))[0]
        
        def get_base_angular_velocity(self, name):
            return self.velocities.get(name, ([0, 0, 0], [0, 0, 0]))[1]
    
    sim = MockSim()
    
    assert is_object_stable(sim, 'stable') == True, "Stable object should be stable"
    assert is_object_stable(sim, 'moving') == False, "Moving object should not be stable"
    assert is_object_stable(sim, 'rotating') == False, "Rotating object should not be stable"
    
    print("  ✓ Stability check correct")


def test_volume_computation():
    """Test volume computation"""
    print("\nTesting volume computation...")
    
    from utils.physics_utils import compute_object_volume
    
    # Mock sim with bounding box
    class MockSim:
        def __init__(self):
            self._bodies_idx = {'cube': 1}
            self.physics_client = self
        
        def getAABB(self, body_id):
            # Returns min and max corners for a 0.04m cube
            return ([0.0, 0.0, 0.0], [0.04, 0.04, 0.04])
    
    sim = MockSim()
    volume = compute_object_volume(sim, 'cube')
    
    expected_volume = 0.04 * 0.04 * 0.04
    assert abs(volume - expected_volume) < 1e-6, f"Expected {expected_volume}, got {volume}"
    
    print("  ✓ Volume computation correct")


def test_velocity_magnitude():
    """Test velocity magnitude calculation"""
    print("\nTesting velocity magnitude...")
    
    from utils.physics_utils import get_object_velocity_magnitude
    
    # Mock sim
    class MockSim:
        def get_base_velocity(self, name):
            return np.array([0.3, 0.4, 0.0])  # magnitude = 0.5
        
        def get_base_angular_velocity(self, name):
            return np.array([1.0, 0.0, 0.0])  # magnitude = 1.0
    
    sim = MockSim()
    lin_speed, ang_speed = get_object_velocity_magnitude(sim, 'test_obj')
    
    assert abs(lin_speed - 0.5) < 1e-6, f"Expected linear speed 0.5, got {lin_speed}"
    assert abs(ang_speed - 1.0) < 1e-6, f"Expected angular speed 1.0, got {ang_speed}"
    
    print("  ✓ Velocity magnitude correct")


def test_multiple_violations():
    """Test checking multiple objects"""
    print("\nTesting multiple workspace violations...")
    
    from utils.physics_utils import check_multiple_workspace_violations
    
    # Mock sim
    class MockSim:
        def get_base_position(self, name):
            positions = {
                'obj1': [0.0, 0.0, 0.05],    # Inside
                'obj2': [0.5, 0.0, 0.05],    # Outside
                'obj3': [0.0, 0.5, 0.05],    # Outside
                'obj4': [0.1, 0.1, 0.05]     # Inside
            }
            return positions.get(name, [0, 0, 0])
    
    sim = MockSim()
    bounds = (-0.3, 0.3, -0.3, 0.3)
    
    violated = check_multiple_workspace_violations(
        sim, 
        ['obj1', 'obj2', 'obj3', 'obj4'],
        bounds
    )
    
    assert len(violated) == 2, f"Expected 2 violations, got {len(violated)}"
    assert 'obj2' in violated, "obj2 should be in violations"
    assert 'obj3' in violated, "obj3 should be in violations"
    
    print("  ✓ Multiple violations check correct")


def test_with_real_environment():
    """Test with actual environment (optional)"""
    print("\nTesting with real environment...")
    print("  (This requires PyBullet to be working)")
    
    try:
        import gymnasium as gym
        import envs
        
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        obs, info = env.reset()
        
        from utils.physics_utils import (
            is_object_stable,
            get_object_velocity_magnitude,
            check_workspace_violation
        )
        
        # Get first object
        if len(env.objects) > 0:
            obj_name = list(env.objects.keys())[0]
            
            # Test stability
            stable = is_object_stable(env.sim, obj_name)
            print(f"  ✓ Object {obj_name} stable: {stable}")
            
            # Test velocity
            lin_speed, ang_speed = get_object_velocity_magnitude(env.sim, obj_name)
            print(f"  ✓ Velocity: lin={lin_speed:.4f}, ang={ang_speed:.4f}")
            
            # Test workspace
            bounds = (-0.3, 0.3, -0.3, 0.3)
            violated = check_workspace_violation(env.sim, obj_name, bounds)
            print(f"  ✓ Workspace violation: {violated}")
        
        env.close()
        print("  ✓ Real environment integration works")
        
    except Exception as e:
        print(f"  ⚠ Real environment test skipped: {e}")
        print("  This is okay if PyBullet GUI isn't set up yet")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing utils/physics_utils.py")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
        print("\n Cannot proceed - import errors")
        sys.exit(1)
    
    # Test 2-6: Unit tests
    tests = [
        test_workspace_violation_logic,
        test_stability_check,
        test_volume_computation,
        test_velocity_magnitude,
        test_multiple_violations
    ]
    
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"Test failed: {e}")
            all_passed = False
        except Exception as e:
            print(f"Unexpected error: {e}")
            all_passed = False
    
    # Test 7: Real environment (optional)
    try:
        test_with_real_environment()
    except Exception as e:
        print(f"  ⚠ Real environment test failed: {e}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CORE TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
