"""
COMPREHENSIVE TEST SUITE FOR STRATEGIC PUSH-GRASP ENVIRONMENT

This test suite validates all critical components of the robotic manipulation environment:

TESTS 1-4: FOUNDATIONAL VERIFICATION
1. Environment Creation - Basic instantiation and registration
2. Action Space - 4D continuous space [-1,1] validation  
3. State Space Structure - Observation dimension formula compliance
4. State Components - Individual state element shapes and properties

TESTS 5-7: CORE FUNCTIONALITY
5. Action Execution - Grasp vs push decision logic (α_skill threshold)
6. Reward Components - All 8 reward terms computation and summation
7. Episode Flow - Complete lifecycle from reset to termination

TESTS 8-10: ROBUSTNESS & CONSISTENCY  
8. Reset Consistency - Stable observation structure across resets
9. State Bounds - Physically realistic value ranges
10. Shape Descriptors - Object geometry encoding and properties

USAGE: Run this file directly to execute full test suite.
        All tests must pass for environment to be considered stable.
"""
"""

import numpy as np
import gymnasium as gym
import envs  # This line is CRITICAL - it executes registration code in __init__.py

def test_environment_creation():
    """Test 1: Environment can be created successfully with correct initialization."""
    print("\n" + "=" * 70)
    print("TEST 1: Environment Creation")
    print("=" * 70)
    
    try:
        # Create the environment instance with GUI rendering enabled
        env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
        print("Environment created successfully")
        
        # Display key environment properties for verification
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
    """Test 2: Action space matches proposal specification (4D continuous, [-1, 1])."""
    print("\n" + "=" * 70)
    print("TEST 2: Action Space Verification")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    
    # Verify action space has correct dimensionality (4 continuous parameters)
    assert env.action_space.shape == (4,), \
        f"Expected shape (4,), got {env.action_space.shape}"
    print("Action space shape: (4,)")
    
    # Verify action bounds match proposal specification [-1, 1]
    assert np.all(env.action_space.low == -1.0), \
        f"Expected low=-1.0, got {env.action_space.low}"
    assert np.all(env.action_space.high == 1.0), \
        f"Expected high=1.0, got {env.action_space.high}"
    print("Action space bounds: [-1.0, 1.0]")
    
    # Verify data type is 32-bit float for efficient neural network processing
    assert env.action_space.dtype == np.float32, \
        f"Expected dtype=float32, got {env.action_space.dtype}"
    print("Action space dtype: float32")
    
    # Test random action sampling to ensure proper distribution
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
    """Test 3: State space structure and dimensionality matches proposal exactly."""
    print("\n" + "=" * 70)
    print("TEST 3: State Space Structure")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    # Use unwrapped environment to access internal attributes
    env_unwrapped = env.unwrapped
    MAX_OBJECTS = env_unwrapped.MAX_OBJECTS
    
    # Get ACTUAL number of objects in current scene
    N_actual = len(env_unwrapped.objects)
    print(f"Actual objects in scene: N = {N_actual}")
    print(f"MAX_OBJECTS (padded): {MAX_OBJECTS}")
    
    # Calculate expected dimension with PADDING for fixed-size observation
    # Formula: 28 (robot + env) + 21×MAX_OBJECTS (object states) + MAX_OBJECTS² (distances) + MAX_OBJECTS (occlusion)
    expected_dim = 28 + MAX_OBJECTS * 21 + MAX_OBJECTS**2 + MAX_OBJECTS
    actual_dim = obs.shape[0]
    
    # Display detailed dimension breakdown for verification
    print(f"\nDimension Formula (with padding):")
    print(f"  28 (robot + env info)")
    print(f"  + 21×{MAX_OBJECTS} = {21*MAX_OBJECTS} (padded object states)")
    print(f"  + {MAX_OBJECTS}² = {MAX_OBJECTS**2} (padded distance matrix)")
    print(f"  + {MAX_OBJECTS} = {MAX_OBJECTS} (padded occlusion mask)")
    print(f"  = {expected_dim} dimensions")
    print(f"\nActual observation: {actual_dim} dimensions")
    
    # Critical assertion: verify total dimension matches expected
    assert actual_dim == expected_dim, \
        f"Dimension mismatch! Expected {expected_dim}, got {actual_dim}"
    print("✓ State space dimensions match formula!")
    
    env.close()
    return True

def test_state_space_components():
    """Test 4: Verify individual state space components have correct shapes and properties."""
    print("\n" + "=" * 70)
    print("TEST 4: State Space Components")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    env_unwrapped = env.unwrapped
    
    # Component 1: Robot state (22D total)
    # Breakdown: 7 joints + 7 velocities + 3 position + 4 orientation + 1 gripper = 22
    robot_state = env_unwrapped._get_robot_state()
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
    print("✓ Robot state components correct")
    
    # Component 2: Object states (N×21D per object)
    # Breakdown per object: 3 position + 4 orientation + 3 velocity + 3 angular velocity + 8 shape = 21
    object_states = env_unwrapped._get_object_states()
    N = len(env_unwrapped.objects)
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
    
    # Component 3: Spatial relationships between objects
    spatial = env_unwrapped._get_spatial_relationships()
    print(f"\n3. Spatial Relationships:")
    print(f"   - Distance matrix: {spatial['distance_matrix'].shape}")
    assert spatial['distance_matrix'].shape == (N, N), "Distance matrix wrong shape"
    print(f"   - Occlusion mask: {spatial['occlusion_mask'].shape}")
    assert spatial['occlusion_mask'].shape == (N,), "Occlusion mask wrong shape"
    print(" Spatial relationship components correct")
    
    # Verify mathematical properties of distance matrix
    D = spatial['distance_matrix']
    
    # Distance matrix must be symmetric (distance A→B = distance B→A)
    assert np.allclose(D, D.T), "Distance matrix not symmetric!"
    print("Distance matrix is symmetric")
    
    # Diagonal must be zero (distance from object to itself)
    assert np.allclose(np.diag(D), 0), "Distance matrix diagonal not zero!"
    print("Distance matrix diagonal is zero")
    
    env.close()
    return True

def test_action_execution():
    """Test 5: Action execution and skill selection logic (grasp vs push decision making)."""
    print("\n" + "=" * 70)
    print("TEST 5: Action Execution")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    print(f"Initial objects: {len(env.objects)}")
    print(f"Current target: {env.current_target}")
    
    # Test grasp action (α_skill > 0) - skill parameter determines action type
    print("\n--- Testing GRASP action (α_skill > 0) ---")
    action_grasp = np.array([0.8, 0.0, 0.0, 0.0])  # α_skill = 0.8 triggers grasping
    obs, reward, terminated, truncated, info = env.step(action_grasp)
    
    # Verify the environment correctly identifies this as a grasp action
    assert info['action_type'] == 'grasp', "Action type should be 'grasp'"
    print(f" α_skill=0.8 triggered GRASP")
    print(f"   Action successful: {env.action_was_successful}")
    print(f"   Reward breakdown: {info['reward_breakdown']}")
    
    # Test push action (α_skill ≤ 0) - negative or zero skill parameter triggers pushing
    print("\n--- Testing PUSH action (α_skill ≤ 0) ---")
    action_push = np.array([-0.5, 0.2, -0.3, 0.7])  # α_skill = -0.5 triggers pushing
    obs, reward, terminated, truncated, info = env.step(action_push)
    
    assert info['action_type'] == 'push', "Action type should be 'push'"
    print(f"α_skill=-0.5 triggered PUSH")
    print(f"   Action successful: {env.action_was_successful}")
    print(f"   Reward breakdown: {info['reward_breakdown']}")
    
    # Test boundary case (α_skill = 0) - verify edge case behavior
    print("\n--- Testing boundary (α_skill = 0) ---")
    action_boundary = np.array([0.0, 0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action_boundary)
    
    assert info['action_type'] == 'push', "α_skill=0 should trigger 'push'"
    print(f"α_skill=0.0 triggered PUSH (correct boundary behavior)")
    
    env.close()
    return True


def test_reward_components():
    """Test 6: Verify all 8 reward components are computed and sum correctly."""
    print("\n" + "=" * 70)
    print("TEST 6: Reward Components")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    # Take one random action to generate reward signal
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Define the expected 8 reward components from the proposal
    expected_components = [
        'placement',      # Reward for placing objects in goal
        'completion',     # Reward for task completion
        'push_success',   # Reward for successful pushes
        'failure',        # Penalty for failures
        'workspace_violation',  # Penalty for leaving workspace
        'collision',      # Penalty for collisions
        'step',           # Small penalty per step (encourages efficiency)
        'trajectory'      # Reward for smooth trajectories
    ]
    
    # Extract reward breakdown from info dictionary
    reward_breakdown = info['reward_breakdown']
    print("\nReward Components:")
    
    # Verify each component exists and display its value
    for i, component in enumerate(expected_components, 1):
        assert component in reward_breakdown, f"Missing component: {component}"
        value = reward_breakdown[component]
        print(f"  {i}. {component:20s}: {value:+7.3f}")  # Formatted output
    
    print(" All 8 reward components present")
    
    # Critical validation: ensure total reward equals sum of components
    total_from_breakdown = sum(reward_breakdown.values())
    assert np.isclose(reward, total_from_breakdown, atol=1e-5), \
        f"Reward mismatch! {reward} != {total_from_breakdown}"
    print(f" Total reward matches sum: {reward:.3f}")
    
    env.close()
    return True


def test_episode_flow():
    """Test 7: Complete episode lifecycle from start to termination."""
    print("\n" + "=" * 70)
    print("TEST 7: Episode Flow")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    # Display initial environment state
    print(f"Initial state:")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Objects: {len(env.objects)}")
    print(f"  - Collected: {len(env.collected_objects)}")
    
    # Run a complete episode (or 10 steps maximum for testing)
    print(f"\nRunning 10 steps...")
    total_reward = 0
    
    for step in range(10):
        # Sample random action and step the environment
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print progress every 3 steps for monitoring
        if step % 3 == 0:
            print(f"  Step {step+1}: reward={reward:+6.2f}, collected={info['collected']}/{info['total']}")
        
        # Check for normal episode completion (success/failure)
        if terminated:
            print(f"\n Episode completed at step {step+1}!")
            break
        
        # Check for episode truncation (time limit exceeded)
        if truncated:
            print(f"\n Episode truncated at step {step+1}")
            break
    
    # Display comprehensive episode summary
    print(f"\nEpisode Summary:")
    print(f"  - Total steps: {step+1}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Objects collected: {info['collected']}/{info['total']}")
    print(f"  - Success: {info['is_success']}")
    print("Episode flow working correctly")
    
    env.close()
    return True

def test_reset_consistency():
    """Test 8: Verify environment reset produces consistent observation structure across multiple resets."""
    print("\n" + "=" * 70)
    print("TEST 8: Reset Consistency")
    print("=" * 70)

    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    env_u = env.unwrapped  # Access the underlying environment implementation
    MAX_OBJECTS = env_u.MAX_OBJECTS  # Maximum object capacity for padding

    obs_shapes = []   # Track observation dimensions across resets
    n_actual_list = []  # Track actual object counts across resets

    # Perform 5 consecutive reset operations to test consistency
    for i in range(5):
        obs, info = env.reset()
        N_actual = len(env_u.objects)  # Actual number of objects in current scene
        # Calculate expected observation dimension using the formula:
        # 28 (robot+env) + 21×MAX_OBJECTS (object states) + MAX_OBJECTS² (distances) + MAX_OBJECTS (occlusion)
        expected_dim = 28 + 21*MAX_OBJECTS + MAX_OBJECTS**2 + MAX_OBJECTS
        
        print(f"[Reset {i+1}] obs.shape={obs.shape}, "
              f"N_actual={N_actual}, MAX_OBJECTS={MAX_OBJECTS}, "
              f"expected_dim={expected_dim}")

        # Store metrics for analysis
        obs_shapes.append(obs.shape[0])
        n_actual_list.append(N_actual)

        # Critical assertion: observation dimension must match expected formula
        assert obs.shape[0] == expected_dim, \
            f"Reset {i+1}: dimension mismatch {obs.shape[0]} != {expected_dim}"

    print("Reset produces consistent structure")
    env.close()
    return True


def test_state_bounds():
    """Test 9: Verify state values remain within physically reasonable bounds."""
    print("\n" + "=" * 70)
    print("TEST 9: State Value Bounds")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    # Check for extreme values that might indicate numerical instability
    max_val = np.max(np.abs(obs))
    print(f"Maximum absolute value in observation: {max_val:.2f}")
    
    # Assert no extremely large values (indicative of bugs or numerical issues)
    assert max_val < 100, f"Observation contains very large values: {max_val}"
    print("No extreme values detected")
    
    # Check specific physical components for realism
    robot_state = env.unwrapped._get_robot_state()
    
    # Joint angles should be in reasonable range (typical robotic arms: -π to π radians)
    joint_range = np.max(np.abs(robot_state['joint_positions']))
    print(f"Joint angle range: ±{joint_range:.2f} rad")
    assert joint_range < 10, "Joint angles out of reasonable range"
    print("Joint angles in reasonable range")
    
    # End-effector position should be near robot base (reasonable workspace)
    ee_pos = robot_state['ee_position']
    ee_dist = np.linalg.norm(ee_pos)  # Distance from world origin
    print(f"EE distance from origin: {ee_dist:.2f}m")
    assert ee_dist < 2.0, "EE too far from robot base"
    print("EE position reasonable")
    
    env.close()
    return True


def test_shape_descriptors():
    """Test 10: Verify shape descriptors are correctly computed with proper one-hot encoding and physical properties."""
    print("\n" + "=" * 70)
    print("TEST 10: Shape Descriptors")
    print("=" * 70)
    
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    obs, info = env.reset()
    
    print(f"\nObjects in scene:")
    # Iterate through all objects and validate their shape descriptors
    for name, props in env.objects.items():
        desc = props['shape_descriptor']  # 8-dimensional shape descriptor
        obj_type = props['type']  # Object type: 'cube', 'sphere', or 'irregular'
        
        print(f"\n  {name} ({obj_type}):")
        print(f"    Shape descriptor: {desc}")
        print(f"    [is_cube, is_sphere, is_irregular, dim1, dim2, dim3, volume, grasp_score]")
        
        # Verify one-hot encoding for object type (exactly one type flag should be 1)
        type_flags = desc[:3]  # First 3 elements: [cube_flag, sphere_flag, irregular_flag]
        assert np.sum(type_flags) == 1.0, f"Type flags should sum to 1: {type_flags}"
        
        # Validate specific type flags based on actual object type
        if obj_type == "cube":
            assert desc[0] == 1.0, "Cube flag should be 1"
            assert desc[1] == 0.0, "Sphere flag should be 0"
        elif obj_type == "sphere":
            assert desc[0] == 0.0, "Cube flag should be 0"
            assert desc[1] == 1.0, "Sphere flag should be 1"
        
        # Check physical dimensions are positive (non-zero size)
        dims = desc[3:6]  # Dimensions [length, width, height] or [radius, radius, radius]
        assert np.all(dims > 0), f"Dimensions should be positive: {dims}"
        
        # Check volume is positive (physically meaningful)
        volume = desc[6]
        assert volume > 0, f"Volume should be positive: {volume}"
        
        # Check graspability score is normalized between 0 and 1
        grasp_score = desc[7]
        assert 0 <= grasp_score <= 1.0, f"Grasp score should be in [0,1]: {grasp_score}"
    
    print("\n Shape descriptors correctly computed")
    
    env.close()
    return True


def run_all_tests():
    """Execute complete test suite and provide comprehensive results summary."""
    print("\n" + "-" * 35)
    print("COMPLETE STATE & ACTION SPACE TEST SUITE")
    print("-" * 35)
    
    # Define all test cases in execution order
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
    
    results = []  # Store test results for final summary
    
    # Execute each test with proper error handling
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()  # Provide detailed error context
            results.append((test_name, False))
    
    # Print comprehensive test summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    # Display individual test results
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:8s} - {test_name}")
    
    print("=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 70)
    
    # Final verdict with appropriate messaging
    if passed == total:
        print("\n ALL TESTS PASSED! 🎉")
        print("State space and action space are correctly implemented!")
        return True
    else:
        print(f"\n {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    # Execute test suite and return appropriate exit code
    success = run_all_tests()
    exit(0 if success else 1)  # 0 = success, 1 = failure 
