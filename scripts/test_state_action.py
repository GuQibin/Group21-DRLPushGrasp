"""
Complete test suite for state space and action space verification.
Tests that the implementation matches the ME5418 proposal exactly.
"""

import numpy as np
import gymnasium as gym
import envs  # 这一行非常重要，它会执行__init__.py中的注册代码

def test_environment_creation():
    """Test 1: Environment can be created successfully."""
    print("\n" + "=" * 70)
    print("TEST 1: Environment Creation")
    print("=" * 70)
    
    try:
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        print("Environment created successfully")
        
        print(f"\nEnvironment Info:")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space shape: {env.observation_space.shape}")
        print(f"  - Max episode steps: {env.max_episode_steps}")
        
        env.close()
        return True
    except Exception as e:
        print(f"Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_space():
    """Test 2: Action space matches proposal (4D continuous, [-1, 1])."""
    print("\n" + "=" * 70)
    print("TEST 2: Action Space Verification")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    
    # Check shape
    assert env.action_space.shape == (4,), \
        f"Expected shape (4,), got {env.action_space.shape}"
    print("Action space shape: (4,)")
    
    # Check bounds
    assert np.all(env.action_space.low == -1.0), \
        f"Expected low=-1.0, got {env.action_space.low}"
    assert np.all(env.action_space.high == 1.0), \
        f"Expected high=1.0, got {env.action_space.high}"
    print("Action space bounds: [-1.0, 1.0]")
    
    # Check dtype
    assert env.action_space.dtype == np.float32, \
        f"Expected dtype=float32, got {env.action_space.dtype}"
    print("Action space dtype: float32")
    
    # Test random sampling
    print("\nTesting random action sampling:")
    for i in range(5):
        action = env.action_space.sample()
        print(f"  Sample {i+1}: {action}")
        assert action.shape == (4,), "Sample shape incorrect"
        assert np.all(action >= -1.0) and np.all(action <= 1.0), "Sample out of bounds"
    print("Random sampling works correctly")
    
    env.close()
    return True


def test_state_space_structure():
    """Test 3: State space structure matches proposal."""
    print("\n" + "=" * 70)
    print("TEST 3: State Space Structure")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    # Get number of objects (last element of observation)
    N = int(obs[-1])
    print(f"Number of objects: N = {N}")
    
    # Check observation dimensions
    expected_dim = 28 + 22*N + N**2
    actual_dim = obs.shape[0]
    
    print(f"\nDimension Formula: 28 + 22N + N²")
    print(f"  28 (robot + env info)")
    print(f"  + 22×{N} = {22*N} (object states)")
    print(f"  + {N}² = {N**2} (distance matrix)")
    print(f"  = {expected_dim} dimensions")
    print(f"\nActual observation: {actual_dim} dimensions")
    
    assert actual_dim == expected_dim, \
        f"Dimension mismatch! Expected {expected_dim}, got {actual_dim}"
    print("State space dimensions match formula!")
    
    # Verify observation is not all zeros
    assert not np.all(obs == 0), "Observation is all zeros!"
    print("Observation contains non-zero values")
    
    # Verify observation has no NaN or Inf
    assert not np.any(np.isnan(obs)), "Observation contains NaN!"
    assert not np.any(np.isinf(obs)), "Observation contains Inf!"
    print("Observation is valid (no NaN/Inf)")
    
    env.close()
    return True


def test_state_space_components():
    """Test 4: Individual state space components."""
    print("\n" + "=" * 70)
    print("TEST 4: State Space Components")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    # Component 1: Robot state (22D)
    robot_state = env._get_robot_state()
    print(f"\n1. Robot State (22D):")
    print(f"   - Joint positions: {robot_state['joint_positions'].shape}")
    assert robot_state['joint_positions'].shape == (7,), "Joint positions wrong shape"
    print(f"   - Joint velocities: {robot_state['joint_velocities'].shape}")
    assert robot_state['joint_velocities'].shape == (7,), "Joint velocities wrong shape"
    print(f"   - EE position: {robot_state['ee_position'].shape}")
    assert robot_state['ee_position'].shape == (3,), "EE position wrong shape"
    print(f"   - EE orientation: {robot_state['ee_orientation'].shape}")
    assert robot_state['ee_orientation'].shape == (4,), "EE orientation wrong shape"
    print(f"   - Gripper width: {robot_state['gripper_width'].shape}")
    assert robot_state['gripper_width'].shape == (1,), "Gripper width wrong shape"
    print("Robot state components correct")
    
    # Component 2: Object states (N×21D)
    object_states = env._get_object_states()
    N = len(env.objects)
    print(f"\n2. Object States ({N}×21D):")
    print(f"   - Positions: {object_states['positions'].shape}")
    assert object_states['positions'].shape == (N, 3), "Positions wrong shape"
    print(f"   - Orientations: {object_states['orientations'].shape}")
    assert object_states['orientations'].shape == (N, 4), "Orientations wrong shape"
    print(f"   - Velocities: {object_states['velocities'].shape}")
    assert object_states['velocities'].shape == (N, 3), "Velocities wrong shape"
    print(f"   - Angular velocities: {object_states['angular_velocities'].shape}")
    assert object_states['angular_velocities'].shape == (N, 3), "Angular velocities wrong shape"
    print(f"   - Shape descriptors: {object_states['shape_descriptors'].shape}")
    assert object_states['shape_descriptors'].shape == (N, 8), "Shape descriptors wrong shape"
    print("Object state components correct")
    
    # Component 3: Spatial relationships
    spatial = env._get_spatial_relationships()
    print(f"\n3. Spatial Relationships:")
    print(f"   - Distance matrix: {spatial['distance_matrix'].shape}")
    assert spatial['distance_matrix'].shape == (N, N), "Distance matrix wrong shape"
    print(f"   - Occlusion mask: {spatial['occlusion_mask'].shape}")
    assert spatial['occlusion_mask'].shape == (N,), "Occlusion mask wrong shape"
    print(" Spatial relationship components correct")
    
    # Check distance matrix is symmetric
    D = spatial['distance_matrix']
    assert np.allclose(D, D.T), "Distance matrix not symmetric!"
    print("Distance matrix is symmetric")
    
    # Check diagonal is zero
    assert np.allclose(np.diag(D), 0), "Distance matrix diagonal not zero!"
    print("Distance matrix diagonal is zero")
    
    env.close()
    return True


def test_action_execution():
    """Test 5: Action execution (grasp vs push)."""
    print("\n" + "=" * 70)
    print("TEST 5: Action Execution")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    print(f"Initial objects: {len(env.objects)}")
    print(f"Current target: {env.current_target}")
    
    # Test grasp action (α_skill > 0)
    print("\n--- Testing GRASP action (α_skill > 0) ---")
    action_grasp = np.array([0.8, 0.0, 0.0, 0.0])  # α_skill = 0.8
    obs, reward, terminated, truncated, info = env.step(action_grasp)
    
    assert info['action_type'] == 'grasp', "Action type should be 'grasp'"
    print(f" α_skill=0.8 triggered GRASP")
    print(f"   Action successful: {env.action_was_successful}")
    print(f"   Reward breakdown: {info['reward_breakdown']}")
    
    # Test push action (α_skill ≤ 0)
    print("\n--- Testing PUSH action (α_skill ≤ 0) ---")
    action_push = np.array([-0.5, 0.2, -0.3, 0.7])  # α_skill = -0.5
    obs, reward, terminated, truncated, info = env.step(action_push)
    
    assert info['action_type'] == 'push', "Action type should be 'push'"
    print(f"α_skill=-0.5 triggered PUSH")
    print(f"   Action successful: {env.action_was_successful}")
    print(f"   Reward breakdown: {info['reward_breakdown']}")
    
    # Test boundary case (α_skill = 0)
    print("\n--- Testing boundary (α_skill = 0) ---")
    action_boundary = np.array([0.0, 0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action_boundary)
    
    assert info['action_type'] == 'push', "α_skill=0 should trigger 'push'"
    print(f"α_skill=0.0 triggered PUSH (correct boundary behavior)")
    
    env.close()
    return True


def test_reward_components():
    """Test 6: All 8 reward components are computed."""
    print("\n" + "=" * 70)
    print("TEST 6: Reward Components")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    # Take one action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check all 8 components exist
    expected_components = [
        'placement', 'completion', 'push_success', 'failure',
        'workspace_violation', 'collision', 'step', 'trajectory'
    ]
    
    reward_breakdown = info['reward_breakdown']
    print("\nReward Components:")
    for i, component in enumerate(expected_components, 1):
        assert component in reward_breakdown, f"Missing component: {component}"
        value = reward_breakdown[component]
        print(f"  {i}. {component:20s}: {value:+7.3f}")
    
    print(" All 8 reward components present")
    
    # Verify total equals sum
    total_from_breakdown = sum(reward_breakdown.values())
    assert np.isclose(reward, total_from_breakdown, atol=1e-5), \
        f"Reward mismatch! {reward} != {total_from_breakdown}"
    print(f" Total reward matches sum: {reward:.3f}")
    
    env.close()
    return True


def test_episode_flow():
    """Test 7: Complete episode flow."""
    print("\n" + "=" * 70)
    print("TEST 7: Episode Flow")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    print(f"Initial state:")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Objects: {len(env.objects)}")
    print(f"  - Collected: {len(env.collected_objects)}")
    
    # Run 10 steps
    print(f"\nRunning 10 steps...")
    total_reward = 0
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 3 == 0:
            print(f"  Step {step+1}: reward={reward:+6.2f}, collected={info['collected']}/{info['total']}")
        
        if terminated:
            print(f"\n Episode completed at step {step+1}!")
            break
        
        if truncated:
            print(f"\n Episode truncated at step {step+1}")
            break
    
    print(f"\nEpisode Summary:")
    print(f"  - Total steps: {step+1}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Objects collected: {info['collected']}/{info['total']}")
    print(f"  - Success: {info['is_success']}")
    print("Episode flow working correctly")
    
    env.close()
    return True


def test_reset_consistency():
    """Test 8: Reset produces consistent structure."""
    print("\n" + "=" * 70)
    print("TEST 8: Reset Consistency")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    
    # Reset multiple times
    obs_shapes = []
    num_objects = []
    
    for i in range(5):
        obs, info = env.reset()
        N = int(obs[-1])
        expected_dim = 28 + 22*N + N**2
        
        obs_shapes.append(obs.shape[0])
        num_objects.append(N)
        
        assert obs.shape[0] == expected_dim, \
            f"Reset {i+1}: dimension mismatch {obs.shape[0]} != {expected_dim}"
    
    print(f"Reset results:")
    for i, (shape, N) in enumerate(zip(obs_shapes, num_objects), 1):
        expected = 28 + 22*N + N**2
        print(f"  Reset {i}: N={N}, obs_dim={shape}, expected={expected} ✓")
    
    print("Reset produces consistent structure")
    
    env.close()
    return True


def test_state_bounds():
    """Test 9: State values are within reasonable bounds."""
    print("\n" + "=" * 70)
    print("TEST 9: State Value Bounds")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    # Check for extreme values
    max_val = np.max(np.abs(obs))
    print(f"Maximum absolute value in observation: {max_val:.2f}")
    
    # Reasonable bounds for robot workspace
    assert max_val < 100, f"Observation contains very large values: {max_val}"
    print("No extreme values detected")
    
    # Check specific components
    robot_state = env._get_robot_state()
    
    # Joint angles should be in reasonable range (typically -π to π)
    joint_range = np.max(np.abs(robot_state['joint_positions']))
    print(f"Joint angle range: ±{joint_range:.2f} rad")
    assert joint_range < 10, "Joint angles out of reasonable range"
    print("Joint angles in reasonable range")
    
    # EE position should be near robot base
    ee_pos = robot_state['ee_position']
    ee_dist = np.linalg.norm(ee_pos)
    print(f"EE distance from origin: {ee_dist:.2f}m")
    assert ee_dist < 2.0, "EE too far from robot base"
    print("EE position reasonable")
    
    env.close()
    return True


def test_shape_descriptors():
    """Test 10: Shape descriptors are correctly computed."""
    print("\n" + "=" * 70)
    print("TEST 10: Shape Descriptors")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    print(f"\nObjects in scene:")
    for name, props in env.objects.items():
        desc = props['shape_descriptor']
        obj_type = props['type']
        
        print(f"\n  {name} ({obj_type}):")
        print(f"    Shape descriptor: {desc}")
        print(f"    [is_cube, is_sphere, is_irregular, dim1, dim2, dim3, volume, grasp_score]")
        
        # Verify one-hot encoding
        type_flags = desc[:3]
        assert np.sum(type_flags) == 1.0, f"Type flags should sum to 1: {type_flags}"
        
        if obj_type == "cube":
            assert desc[0] == 1.0, "Cube flag should be 1"
            assert desc[1] == 0.0, "Sphere flag should be 0"
        elif obj_type == "sphere":
            assert desc[0] == 0.0, "Cube flag should be 0"
            assert desc[1] == 1.0, "Sphere flag should be 1"
        
        # Check dimensions are positive
        dims = desc[3:6]
        assert np.all(dims > 0), f"Dimensions should be positive: {dims}"
        
        # Check volume is positive
        volume = desc[6]
        assert volume > 0, f"Volume should be positive: {volume}"
        
        # Check graspability score in [0, 1]
        grasp_score = desc[7]
        assert 0 <= grasp_score <= 1.0, f"Grasp score should be in [0,1]: {grasp_score}"
    
    print("\n Shape descriptors correctly computed")
    
    env.close()
    return True


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "-" * 35)
    print("COMPLETE STATE & ACTION SPACE TEST SUITE")
    print("-" * 35)
    
    tests = [
        ("Environment Creation", test_environment_creation),
        ("Action Space", test_action_space),
        ("State Space Structure", test_state_space_structure),
        ("State Space Components", test_state_space_components),
        ("Action Execution", test_action_execution),
        ("Reward Components", test_reward_components),
        ("Episode Flow", test_episode_flow),
        ("Reset Consistency", test_reset_consistency),
        ("State Value Bounds", test_state_bounds),
        ("Shape Descriptors", test_shape_descriptors)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:8s} - {test_name}")
    
    print("=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n ALL TESTS PASSED! 🎉")
        print("State space and action space are correctly implemented!")
        return True
    else:
        print(f"\n {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
