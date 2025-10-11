"""
Test script to verify object_utils.py works correctly
Run this after adding object_utils.py to your project
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_shape_descriptors():
    """Test shape descriptor computation"""
    print("Testing compute_shape_descriptors...")
    
    from utils.object_util import compute_shape_descriptors
    
    # Test cube
    cube_desc = compute_shape_descriptors("cube", half_extents=np.array([0.02, 0.02, 0.02]))
    assert cube_desc.shape == (8,), f"Expected shape (8,), got {cube_desc.shape}"
    assert cube_desc[0] == 1.0, "Cube flag should be 1.0"
    assert cube_desc[1] == 0.0, "Sphere flag should be 0.0"
    print("  ✓ Cube descriptor works")
    
    # Test sphere
    sphere_desc = compute_shape_descriptors("sphere", radius=0.02)
    assert sphere_desc.shape == (8,), f"Expected shape (8,), got {sphere_desc.shape}"
    assert sphere_desc[0] == 0.0, "Cube flag should be 0.0"
    assert sphere_desc[1] == 1.0, "Sphere flag should be 1.0"
    print("  ✓ Sphere descriptor works")
    
    # Test irregular
    irregular_desc = compute_shape_descriptors("irregular")
    assert irregular_desc[2] == 1.0, "Irregular flag should be 1.0"
    print("  ✓ Irregular descriptor works")
    
    print("Shape descriptors test passed!\n")


def test_with_mock_sim():
    """Test functions with mock simulation"""
    print("Testing with mock simulation...")
    
    from utils.object_util import (
        compute_pairwise_distance_matrix,
        compute_occlusion_masks,
        select_target_heuristic,
        check_object_in_goal
    )
    
    # Create mock sim
    class MockSim:
        def __init__(self):
            self.positions = {
                'object_0': np.array([0.0, 0.0, 0.02]),
                'object_1': np.array([0.1, 0.0, 0.02]),
                'object_2': np.array([0.0, 0.1, 0.02])
            }
        
        def get_base_position(self, name):
            return self.positions.get(name, np.array([0, 0, 0]))
    
    sim = MockSim()
    
    objects = {
        'object_0': {'type': 'cube', 'is_occluded': False},
        'object_1': {'type': 'cube', 'is_occluded': False},
        'object_2': {'type': 'sphere', 'is_occluded': False}
    }
    
    # Test distance matrix
    D = compute_pairwise_distance_matrix(sim, objects)
    assert D.shape == (3, 3), f"Expected (3,3), got {D.shape}"
    assert np.all(np.diag(D) == 0), "Diagonal should be zero"
    assert np.allclose(D, D.T), "Matrix should be symmetric"
    print("  ✓ Distance matrix works")
    
    # Test occlusion mask
    O = compute_occlusion_masks(sim, objects, threshold=0.05)
    assert O.shape == (3,), f"Expected (3,), got {O.shape}"
    assert O.dtype in [np.int32, np.int64], f"Expected int type, got {O.dtype}"
    print("  ✓ Occlusion mask works")
    
    # Test target selection
    goal_pos = np.array([0.0, 0.0])
    target = select_target_heuristic(sim, objects, goal_pos)
    assert target in objects.keys(), f"Target {target} not in objects"
    print(f"  ✓ Target selection works (selected: {target})")
    
    # Test goal checking
    in_goal = check_object_in_goal(
        np.array([0.0, 0.0, 0.02]), 
        goal_pos, 
        goal_size=0.1
    )
    assert isinstance(in_goal, (bool, np.bool_)), "Should return boolean"
    print("  ✓ Goal checking works")
    
    print("Mock simulation tests passed!\n")


def test_empty_objects():
    """Test edge case with no objects"""
    print("Testing edge cases...")
    
    from utils.object_util import (
        compute_pairwise_distance_matrix,
        compute_occlusion_masks
    )
    
    class MockSim:
        def get_base_position(self, name):
            return np.array([0, 0, 0])
    
    sim = MockSim()
    objects = {}
    
    # Empty objects
    D = compute_pairwise_distance_matrix(sim, objects)
    assert D.shape == (0, 0) or D.size == 0, f"Expected empty array, got {D.shape}"
    print("  ✓ Empty distance matrix works")
    
    O = compute_occlusion_masks(sim, objects)
    assert O.shape == (0,) or O.size == 0, f"Expected empty array, got {O.shape}"
    print("  ✓ Empty occlusion mask works")
    
    print("Edge case tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing utils/object_utils.py")
    print("=" * 60 + "\n")
    
    try:
        test_shape_descriptors()
        test_with_mock_sim()
        test_empty_objects()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nMake sure:")
        print("1. utils/object_utils.py exists")
        print("2. utils/__init__.py exists")
        print("3. You're running from the correct directory")
        
    except AssertionError as e:
        print(f"Test Failed: {e}")
        
    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
