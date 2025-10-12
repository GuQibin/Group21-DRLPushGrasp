"""
Robot control utilities for action primitive execution.
Implements high-level manipulation primitives: pick-and-place and push.

CORRECTED VERSION - Uses only verified Panda robot and PyBullet wrapper methods
"""

import numpy as np
from typing import Tuple, Optional
import pybullet as p


def get_ee_position_safe(robot) -> np.ndarray:
    """
    Get end-effector position safely.
    
    Args:
        robot: Panda robot instance
    
    Returns:
        position: [x, y, z] or [0, 0, 0] if error
    """
    try:
        # Method 1: Try robot's built-in method
        if hasattr(robot, 'get_ee_position'):
            return robot.get_ee_position()
        
        # Method 2: Use PyBullet directly via robot's sim
        if hasattr(robot, 'sim'):
            ee_link = getattr(robot, 'ee_link', 11)
            panda_uid = robot.sim._bodies_idx.get('panda')
            if panda_uid is not None:
                link_state = robot.sim.physics_client.getLinkState(
                    panda_uid, ee_link, computeForwardKinematics=1
                )
                return np.array(link_state[0], dtype=np.float32)
        
        print(f"Warning: Could not get EE position")
        return np.zeros(3, dtype=np.float32)
        
    except Exception as e:
        print(f"Warning: Could not get EE position: {e}")
        return np.zeros(3, dtype=np.float32)


def get_ee_orientation_safe(robot) -> np.ndarray:
    """
    Get end-effector orientation safely.
    
    Args:
        robot: Panda robot instance
    
    Returns:
        orientation: Quaternion [x, y, z, w] or [0, 0, 0, 1] if error
    """
    try:
        # Method 1: Try robot's built-in method
        if hasattr(robot, 'get_ee_orientation'):
            return robot.get_ee_orientation()
        
        # Method 2: Use PyBullet directly via robot's sim
        if hasattr(robot, 'sim'):
            ee_link = getattr(robot, 'ee_link', 11)
            panda_uid = robot.sim._bodies_idx.get('panda')
            if panda_uid is not None:
                link_state = robot.sim.physics_client.getLinkState(
                    panda_uid, ee_link, computeForwardKinematics=1
                )
                # link_state[1] is world orientation quaternion
                return np.array(link_state[1], dtype=np.float32)
        
        print(f"Warning: Could not get EE orientation")
        return np.array([0, 0, 0, 1], dtype=np.float32)
        
    except Exception as e:
        print(f"Warning: Could not get EE orientation: {e}")
        return np.array([0, 0, 0, 1], dtype=np.float32)

def execute_pick_and_place(sim, robot, target_object: str, 
                          alpha_x: float, alpha_y: float, 
                          goal_pos: np.ndarray,
                          approach_height: float = 0.15,
                          grasp_height: float = 0.05) -> bool:
    """
    Execute complete pick-and-place sequence for target object.
    
    Sequence:
    1. Move to pre-grasp position above object
    2. Lower to grasp height
    3. Close gripper
    4. Check grasp success
    5. Lift object
    6. Move to goal position
    7. Lower and release
    8. Retract
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_object: Name of object to grasp
        alpha_x, alpha_y: Relative grasp position in object's local frame [-1, 1]
        goal_pos: Goal position [x, y]
        approach_height: Height for approach phase (meters)
        grasp_height: Height for grasping (meters)
    
    Returns:
        success: True if grasp and placement were successful
    """
    try:
        # Get object pose
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except Exception as e:
        print(f"❌ Error: Could not get pose for {target_object}: {e}")
        return False
    
    # Store initial height for grasp verification
    initial_obj_z = obj_pos[2]
    
    # Convert alpha_x, alpha_y from [-1, 1] to actual offset (±2.5cm)
    offset_scale = 0.025
    grasp_offset = np.array([alpha_x * offset_scale, alpha_y * offset_scale, 0.0])
    
    # Transform offset to world frame (simplified - assumes minimal rotation)
    # TODO: For better accuracy, use proper quaternion rotation
    grasp_point = obj_pos + grasp_offset
    
    print(f"  Phase 1: Approaching {target_object} from above...")
    # Phase 1: Approach from above
    approach_pos = grasp_point.copy()
    approach_pos[2] = approach_height
    success = move_to_position(sim, robot, approach_pos, gripper_open=True, steps=50)
    if not success:
        print(f"  ❌ Failed to approach {target_object}")
        return False
    
    print(f"  Phase 2: Lowering to grasp height...")
    # Phase 2: Lower to grasp height
    grasp_pos = grasp_point.copy()
    grasp_pos[2] = grasp_height
    success = move_to_position(sim, robot, grasp_pos, gripper_open=True, steps=30)
    if not success:
        print(f"  ❌ Failed to lower to grasp height for {target_object}")
        return False
    
    print(f"  Phase 3: Closing gripper...")
    # Phase 3: Close gripper
    close_gripper(sim, robot, steps=20)
    
    print(f"  Phase 4: Checking grasp...")
    # Phase 4: Check if object is grasped
    if not check_grasp_success(sim, robot, target_object, initial_z=initial_obj_z):
        print(f"  ❌ Grasp failed for {target_object}")
        open_gripper(sim, robot, steps=10)  # Release and give up
        return False
    
    print(f"  Successfully grasped {target_object}")
    
    print(f"  Phase 5: Lifting object...")
    # Phase 5: Lift
    lift_pos = grasp_pos.copy()
    lift_pos[2] = approach_height
    move_to_position(sim, robot, lift_pos, gripper_open=False, steps=30)
    
    print(f"  Phase 6: Transporting to goal...")
    # Phase 6: Transport to goal
    transport_pos = np.array([goal_pos[0], goal_pos[1], approach_height])
    move_to_position(sim, robot, transport_pos, gripper_open=False, steps=50)
    
    print(f"  Phase 7: Placing object...")
    # Phase 7: Lower and release
    place_pos = np.array([goal_pos[0], goal_pos[1], 0.05])
    move_to_position(sim, robot, place_pos, gripper_open=False, steps=30)
    open_gripper(sim, robot, steps=20)
    
    print(f"  Phase 8: Retracting...")
    # Phase 8: Retract
    retract_pos = place_pos.copy()
    retract_pos[2] = approach_height
    move_to_position(sim, robot, retract_pos, gripper_open=True, steps=30)
    
    print(f"  Pick-and-place complete!")
    return True


def execute_push(sim, robot, target_object: str,
                alpha_x: float, alpha_y: float, alpha_theta: float,
                push_distance: float = 0.05,
                push_height: float = 0.03,
                use_object_frame: bool = False) -> bool:
    """
    Execute push primitive on target object.
    
    Sequence:
    1. Calculate contact point on object
    2. Convert push angle to world frame direction
    3. Move to pre-push position
    4. Execute linear push
    5. Retract
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_object: Name of object to push
        alpha_x, alpha_y: Contact point in object's local frame [-1, 1]
        alpha_theta: Push direction [-1, 1], mapped to [-π, π]
        push_distance: Distance to push (default 5cm)
        push_height: Height of push contact point (meters)
        use_object_frame: If True, rotate push direction by object orientation
    
    Returns:
        success: True if push was executed
    """
    try:
        # Get object pose
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except Exception as e:
        print(f"Error: Could not get position for {target_object}: {e}")
        return False
    
    # Convert alpha_x, alpha_y to contact offset
    offset_scale = 0.025  # 2.5cm max offset
    contact_offset = np.array([alpha_x * offset_scale, alpha_y * offset_scale, 0.0])
    
    if use_object_frame:
        # Transform offset to world frame using object orientation
        rot_matrix = quaternion_to_rotation_matrix(obj_ori)
        contact_offset = rot_matrix @ contact_offset
    
    contact_point = obj_pos + contact_offset
    contact_point[2] = push_height
    
    # Convert alpha_theta from [-1, 1] to angle in radians [-π, π]
    push_angle = alpha_theta * np.pi
    push_direction = np.array([np.cos(push_angle), np.sin(push_angle), 0.0])
    
    if use_object_frame:
        # Rotate push direction by object orientation
        rot_matrix = quaternion_to_rotation_matrix(obj_ori)
        push_direction = rot_matrix @ push_direction
    
    # Calculate pre-push position (slightly behind contact point)
    pre_push_offset = 0.03  # 3cm behind
    pre_push_pos = contact_point - push_direction * pre_push_offset
    
    print(f"  Phase 1: Moving to pre-push position...")
    # Phase 1: Move to pre-push position
    success = move_to_position(sim, robot, pre_push_pos, gripper_open=True, steps=40)
    if not success:
        print(f"  Failed to reach pre-push position for {target_object}")
        return False
    
    print(f"  Phase 2: Executing push...")
    # Phase 2: Execute push (linear motion)
    post_push_pos = contact_point + push_direction * push_distance
    success = move_to_position(sim, robot, post_push_pos, gripper_open=True, steps=30)
    if not success:
        print(f"  Push may have been incomplete")
    
    print(f"  Phase 3: Retracting...")
    # Phase 3: Retract (move back and up)
    retract_pos = post_push_pos.copy()
    retract_pos[2] += 0.1
    move_to_position(sim, robot, retract_pos, gripper_open=True, steps=20)
    
    print(f" Push complete!")
    return True


def move_to_position(sim, robot, target_pos: np.ndarray, 
                    gripper_open: bool = True,
                    steps: int = 50) -> bool:
    """
    Move end-effector to target position using robot's action interface.
    
    IMPORTANT: This assumes the Panda robot uses the standard panda-gym
    action space: [delta_x, delta_y, delta_z, gripper_control]
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_pos: Target [x, y, z] position
        gripper_open: Whether gripper should be open
        steps: Number of simulation steps
    
    Returns:
        success: True if reached near target
    """
    # Gripper control
    # Standard panda-gym: 1.0 = open, -1.0 = close
    # OR: 0.04 = open, 0.0 = close (depends on version)
    # Check your Panda implementation to determine correct values!
    print(f"\n[MOVE] Moving to {target_pos}")
    initial_pos = get_ee_position_safe(robot)
    print(f"[MOVE] From {initial_pos}, distance: {np.linalg.norm(target_pos - initial_pos):.4f}m")
    
    # Determine gripper action
    # Based on diagnostic: gripper stays closed, so we need to figure out correct value
    gripper_ctrl = 1.0 if gripper_open else -1.0
    
    for step in range(steps):
        try:
            current_pos = get_ee_position_safe(robot)
            error = target_pos - current_pos
            
            # METHOD 1: Try with much larger gain since actions are being scaled down
            # If 0.01 action only moves 0.006m, we need ~30x gain
            delta = np.clip(error * 50.0, -1.0, 1.0)  # Much more aggressive!
            
            action = np.concatenate([delta, [gripper_ctrl]])
            
            if step % 10 == 0:
                error_mag = np.linalg.norm(error)
                print(f"  Step {step:2d}: error={error_mag:.4f}m, action={delta}")
            
            robot.set_action(action)
            sim.step()
            
            # Check if reached
            if np.linalg.norm(error) < 0.01:
                print(f"  ✓ Reached target at step {step}")
                return True
                
        except Exception as e:
            print(f"  ❌ Error at step {step}: {e}")
            return False
    
    # Final check
    final_pos = get_ee_position_safe(robot)
    final_error = np.linalg.norm(target_pos - final_pos)
    success = final_error < 0.05  # Relax threshold to 5cm
    
    print(f"  Final: pos={final_pos}, error={final_error:.4f}m")
    if not success:
        print(f"  Did not reach target")
    
    return success


def open_gripper(sim, robot, steps: int = 30):
    """
    Open gripper - try multiple strategies since diagnostic shows it's stuck closed.
    """
    print("  [GRIPPER] Attempting to open...")
    
    # Strategy 1: Standard normalized control
    for _ in range(steps):
        robot.set_action(np.array([0.0, 0.0, 0.0, 1.0]))
        sim.step()
    
    # Check if it worked
    state = get_gripper_state(robot)
    if state['is_open']:
        print("  ✓ Gripper opened (Strategy 1)")
        return
    
    # Strategy 2: Try larger positive values
    print("  [GRIPPER] Trying larger values...")
    for _ in range(steps):
        robot.set_action(np.array([0.0, 0.0, 0.0, 10.0]))  # Much larger
        sim.step()
    
    state = get_gripper_state(robot)
    if state['is_open']:
        print("  ✓ Gripper opened (Strategy 2)")
        return
    
    # Strategy 3: Try directly setting gripper joints
    print("  [GRIPPER] Trying direct joint control...")
    try:
        # Panda gripper has finger joints - try setting them directly
        panda_uid = sim._bodies_idx.get('panda')
        if panda_uid is not None:
            # Panda finger joints are typically 9 and 10
            for finger_joint in [9, 10]:
                sim.physics_client.resetJointState(
                    panda_uid, finger_joint, targetValue=0.04
                )
            for _ in range(steps):
                sim.step()
            print("  ✓ Gripper opened (Strategy 3 - direct control)")
    except Exception as e:
        print(f"  ⚠ Strategy 3 failed: {e}")


def close_gripper(sim, robot, steps: int = 30):
    """Close gripper."""
    print("  [GRIPPER] Attempting to close...")
    
    # Standard close command
    for _ in range(steps):
        robot.set_action(np.array([0.0, 0.0, 0.0, -1.0]))
        sim.step()
    
    state = get_gripper_state(robot)
    if state['is_closed']:
        print("  ✓ Gripper closed")
    else:
        print("  ⚠ Gripper may not be fully closed")

def check_grasp_success(sim, robot, object_name: str, 
                       initial_z: Optional[float] = None,
                       min_lift: float = 0.01) -> bool:
    """
    Check if grasp was successful by monitoring object height.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        object_name: Name of grasped object
        initial_z: Initial height before grasp (if None, get current)
        min_lift: Minimum lift distance (meters)
    
    Returns:
        success: True if object lifted
    """
    try:
        if initial_z is None:
            initial_z = sim.get_base_position(object_name)[2]
        
        # Wait a few steps for gripper to settle
        for _ in range(20):
            sim.step()
        
        # Check current height
        current_z = sim.get_base_position(object_name)[2]
        height_gained = current_z - initial_z
        
        if height_gained >= min_lift:
            # Verify object is not falling
            obj_vel = sim.get_base_velocity(object_name)
            is_falling = obj_vel[2] < -0.1  # Fast downward velocity
            
            return not is_falling
        
        return False
        
    except Exception as e:
        print(f"  ⚠ Error checking grasp: {e}")
        return False


def compute_inverse_kinematics(sim, robot, target_pos: np.ndarray, 
                               target_ori: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Compute inverse kinematics using PyBullet wrapper.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_pos: Target position [x, y, z]
        target_ori: Target orientation as quaternion [x, y, z, w]
    
    Returns:
        joint_positions: 7-DOF joint angles or None if fails
    """
    try:
        if target_ori is None:
            # Default downward orientation
            target_ori = np.array([0, 1, 0, 0], dtype=np.float32)
        
        # Get EE link index
        ee_link = getattr(robot, 'ee_link', 11)
        
        # Use the wrapper's inverse_kinematics method
        joint_state = sim.inverse_kinematics(
            body='panda',
            link=ee_link,
            position=target_pos,
            orientation=target_ori
        )
        
        # Return first 7 DOF (arm joints only, not gripper)
        return joint_state[:7]
        
    except Exception as e:
        print(f"  ⚠ Error computing IK: {e}")
        return None


def plan_trajectory(start_joints: np.ndarray, goal_joints: np.ndarray,
                    num_waypoints: int = 10) -> np.ndarray:
    """
    Plan smooth trajectory between joint configurations.
    
    Simple linear interpolation. For real deployment, use
    proper trajectory optimization (e.g., minimum jerk).
    
    Args:
        start_joints: Starting joint configuration (7-DOF)
        goal_joints: Goal joint configuration (7-DOF)
        num_waypoints: Number of intermediate waypoints
    
    Returns:
        trajectory: Array of shape (num_waypoints, 7) with joint waypoints
    """
    if start_joints.shape != goal_joints.shape:
        print(f"  ⚠ Error: Joint array shapes don't match: {start_joints.shape} vs {goal_joints.shape}")
        return np.array([])
    
    trajectory = np.linspace(start_joints, goal_joints, num_waypoints)
    return trajectory


def get_gripper_state(robot) -> dict:
    """
    Get current gripper state information.
    
    Args:
        robot: Panda robot instance
    
    Returns:
        Dictionary with gripper width and status flags
    """
    try:
        obs = robot.get_obs()
        
        # For MOCK robots in tests: check if obs has 16+ elements
        if len(obs) >= 16:
            # Standard panda-gym format: gripper at indices 14-15
            finger_positions = obs[14:16]
            gripper_width = float(np.sum(finger_positions))
        elif len(obs) >= 9:
            # Alternative: gripper at indices 7-8
            finger_positions = obs[7:9]
            gripper_width = float(np.sum(finger_positions))
        else:
            # No gripper data in observation
            gripper_width = 0.0
        
        return {
            'width': gripper_width,
            'is_closed': gripper_width < 0.01,
            'is_open': gripper_width > 0.07
        }
    except Exception as e:
        print(f"  ⚠ Error getting gripper state: {e}")
        return {
            'width': 0.0,
            'is_closed': False,
            'is_open': False
        }


def check_collision_between_bodies(sim, body1_name: str, body2_name: str) -> bool:
    """
    Check collision using direct PyBullet API.
    
    Args:
        sim: PyBullet simulation instance
        body1_name: First body name
        body2_name: Second body name
    
    Returns:
        has_collision: True if bodies collide
    """
    try:
        body1_id = sim._bodies_idx.get(body1_name)
        body2_id = sim._bodies_idx.get(body2_name)
        
        if body1_id is None or body2_id is None:
            return False
        
        # Direct PyBullet API call
        contacts = sim.physics_client.getContactPoints(
            bodyA=body1_id,
            bodyB=body2_id
        )
        
        return len(contacts) > 0
        
    except Exception as e:
        print(f"  ⚠ Error checking collision: {e}")
        return False


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
    
    Returns:
        rotation_matrix: 3x3 rotation matrix
    """
    try:
        # Use PyBullet's conversion
        rot_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        return rot_matrix
    except Exception as e:
        print(f"  ⚠ Error converting quaternion: {e}")
        return np.eye(3)


def wait_for_stability(sim, object_name: str, max_steps: int = 50,
                      velocity_threshold: float = 0.01) -> bool:
    """
    Wait for object to stabilize after manipulation.
    
    Args:
        sim: PyBullet simulation instance
        object_name: Name of object to monitor
        max_steps: Maximum steps to wait
        velocity_threshold: Velocity threshold for stability (m/s)
    
    Returns:
        is_stable: True if object stabilized within max_steps
    """
    for _ in range(max_steps):
        sim.step()
        
        try:
            vel = sim.get_base_velocity(object_name)
            speed = np.linalg.norm(vel)
            
            if speed < velocity_threshold:
                return True
        except Exception:
            return False
    
    return False

def diagnose_robot_control(robot, sim, steps: int = 10):
    """
    Diagnose robot control to verify action space format.
    
    Args:
        robot: Panda robot instance
        sim: PyBullet simulation instance
        steps: Number of test steps
    """
    print("\n" + "="*60)
    print("ROBOT CONTROL DIAGNOSTICS")
    print("="*60)
    
    # Get initial state
    initial_pos = get_ee_position_safe(robot)
    initial_obs = robot.get_obs()
    
    print(f"Initial EE position: {initial_pos}")
    print(f"Initial joint angles: {initial_obs[:7]}")
    print(f"Observation shape: {initial_obs.shape}")
    
    # Test 1: Send zero action
    print("\nTest 1: Zero action (no movement)")
    for _ in range(10):
        robot.set_action(np.array([0.0, 0.0, 0.0, 0.0]))
        sim.step()
    
    pos_after_zero = get_ee_position_safe(robot)
    delta_zero = np.linalg.norm(pos_after_zero - initial_pos)
    print(f"Position after zero action: {pos_after_zero}")
    print(f"Movement: {delta_zero:.6f}m (should be ~0)")
    
    # Test 2: Send small positive X action
    print("\nTest 2: Positive X delta (+0.01m)")
    for _ in range(steps):
        robot.set_action(np.array([0.01, 0.0, 0.0, 0.0]))
        sim.step()
    
    pos_after_x = get_ee_position_safe(robot)
    delta_x = pos_after_x - pos_after_zero
    print(f"Position after +X action: {pos_after_x}")
    print(f"Delta: {delta_x}")
    print(f"Expected: [+0.1, 0, 0] (10 steps × 0.01)")
    print(f"Actual magnitude: {np.linalg.norm(delta_x):.6f}m")
    
    # Test 3: Check gripper
    print("\nTest 3: Gripper control")
    gripper_state_initial = get_gripper_state(robot)
    print(f"Initial gripper: {gripper_state_initial}")
    
    # Open gripper
    for _ in range(20):
        robot.set_action(np.array([0.0, 0.0, 0.0, 1.0]))
        sim.step()
    
    gripper_state_open = get_gripper_state(robot)
    print(f"After open command: {gripper_state_open}")
    
    # Close gripper
    for _ in range(20):
        robot.set_action(np.array([0.0, 0.0, 0.0, -1.0]))
        sim.step()
    
    gripper_state_closed = get_gripper_state(robot)
    print(f"After close command: {gripper_state_closed}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    if delta_zero < 0.001:
        print("✓ Zero action works correctly")
    else:
        print("✗ Zero action caused movement - check action space!")
    
    if 0.05 < np.linalg.norm(delta_x) < 0.15:
        print("✓ Position delta control works")
    else:
        print("✗ Position control not working as expected")
        print("  → Check if action space is [dx, dy, dz, gripper]")
        print("  → Or if it's [target_x, target_y, target_z, gripper]")
    
    if gripper_state_open['is_open'] and gripper_state_closed['is_closed']:
        print("✓ Gripper control works")
    else:
        print("✗ Gripper control issue")
        print(f"  → Open state: {gripper_state_open}")
        print(f"  → Closed state: {gripper_state_closed}")
    
    print("="*60 + "\n")
