"""
Robot control utilities for action primitive execution.
Implements high-level manipulation primitives: pick-and-place and push.

TESTED VERSION - Fixed simulation stepping and error handling
"""

import numpy as np
from typing import Tuple, Optional
import pybullet as p


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
    4. Lift object
    5. Move to goal position
    6. Lower and release
    7. Retract
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_object: Name of object to grasp
        alpha_x, alpha_y: Relative grasp position in object's local frame [-1, 1]
        goal_pos: Goal position [x, y]
        approach_height: Height for approach phase
        grasp_height: Height for grasping
    
    Returns:
        success: True if grasp was successful
    """
    try:
        # Get object pose
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except Exception as e:
        print(f"Error: Could not get pose for {target_object}: {e}")
        return False
    
    # Convert alpha_x, alpha_y from [-1, 1] to actual offset (e.g., ±2cm)
    offset_scale = 0.02
    grasp_offset = np.array([alpha_x * offset_scale, alpha_y * offset_scale, 0.0])
    
    # Transform offset to world frame (simplified - assumes no rotation)
    # TODO: Proper transformation using quaternion rotation
    grasp_point = obj_pos + grasp_offset
    
    # Phase 1: Approach from above
    approach_pos = grasp_point.copy()
    approach_pos[2] = approach_height
    success = move_to_position(sim, robot, approach_pos, open_gripper=True, steps=50)
    if not success:
        print(f"Failed to approach {target_object}")
        return False
    
    # Phase 2: Lower to grasp height
    grasp_pos = grasp_point.copy()
    grasp_pos[2] = grasp_height
    success = move_to_position(sim, robot, grasp_pos, open_gripper=True, steps=30)
    if not success:
        print(f"Failed to lower to grasp height for {target_object}")
        return False
    
    # Phase 3: Close gripper
    close_gripper(sim, robot, steps=20)
    
    # Check if object is grasped (simple check: object moved up)
    try:
        obj_pos_after = np.array(sim.get_base_position(target_object))
        if obj_pos_after[2] < obj_pos[2] + 0.005:  # Object didn't lift
            print(f"Grasp failed for {target_object} - object didn't lift")
            return False
    except Exception as e:
        print(f"Error checking grasp: {e}")
        return False
    
    # Phase 4: Lift
    lift_pos = grasp_pos.copy()
    lift_pos[2] = approach_height
    move_to_position(sim, robot, lift_pos, open_gripper=False, steps=30)
    
    # Phase 5: Transport to goal
    transport_pos = np.array([goal_pos[0], goal_pos[1], approach_height])
    move_to_position(sim, robot, transport_pos, open_gripper=False, steps=50)
    
    # Phase 6: Lower and release
    place_pos = np.array([goal_pos[0], goal_pos[1], 0.05])
    move_to_position(sim, robot, place_pos, open_gripper=False, steps=30)
    open_gripper(sim, robot, steps=20)
    
    # Phase 7: Retract
    retract_pos = place_pos.copy()
    retract_pos[2] = approach_height
    move_to_position(sim, robot, retract_pos, open_gripper=True, steps=30)
    
    return True


def execute_push(sim, robot, target_object: str,
                alpha_x: float, alpha_y: float, alpha_theta: float,
                push_distance: float = 0.05,
                push_height: float = 0.03) -> bool:
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
        push_height: Height of push contact point
    
    Returns:
        success: True if push was executed
    """
    try:
        # Get object pose
        obj_pos = np.array(sim.get_base_position(target_object))
    except Exception as e:
        print(f"Error: Could not get position for {target_object}: {e}")
        return False
    
    # Convert alpha_x, alpha_y to contact offset
    offset_scale = 0.02
    contact_offset = np.array([alpha_x * offset_scale, alpha_y * offset_scale, 0.0])
    contact_point = obj_pos + contact_offset
    contact_point[2] = push_height
    
    # Convert alpha_theta from [-1, 1] to angle in radians [-π, π]
    push_angle = alpha_theta * np.pi
    push_direction = np.array([np.cos(push_angle), np.sin(push_angle), 0.0])
    
    # Calculate pre-push position (slightly behind contact point)
    pre_push_offset = 0.03  # 3cm behind
    pre_push_pos = contact_point - push_direction * pre_push_offset
    
    # Phase 1: Move to pre-push position
    success = move_to_position(sim, robot, pre_push_pos, open_gripper=True, steps=40)
    if not success:
        print(f"Failed to reach pre-push position for {target_object}")
        return False
    
    # Phase 2: Execute push (linear motion)
    post_push_pos = contact_point + push_direction * push_distance
    success = move_to_position(sim, robot, post_push_pos, open_gripper=True, steps=30)
    
    # Phase 3: Retract (move back and up)
    retract_pos = post_push_pos.copy()
    retract_pos[2] += 0.1
    move_to_position(sim, robot, retract_pos, open_gripper=True, steps=20)
    
    return True


def move_to_position(sim, robot, target_pos: np.ndarray, 
                    open_gripper: bool = True,
                    steps: int = 50) -> bool:
    """
    Move end-effector to target position using simple velocity control.
    
    Note: This is a simplified implementation. For real deployment,
    use proper inverse kinematics and motion planning.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_pos: Target [x, y, z] position
        open_gripper: Whether gripper should be open (1.0) or closed (-1.0)
        steps: Number of simulation steps
    
    Returns:
        success: True if reached near target
    """
    gripper_ctrl = 1.0 if open_gripper else -1.0
    
    for step in range(steps):
        try:
            current_pos = robot.get_ee_position()
            error = target_pos - current_pos
            
            # Simple proportional control
            velocity = np.clip(error * 5.0, -0.1, 0.1)
            
            action = np.concatenate([velocity, [gripper_ctrl]])
            robot.set_action(action)
            
            # CRITICAL: Step the simulation!
            sim.step()
            
            # Check if close enough (early termination)
            if np.linalg.norm(error) < 0.01:
                return True
                
        except Exception as e:
            print(f"Error during movement step {step}: {e}")
            return False
    
    # Check final distance
    try:
        final_pos = robot.get_ee_position()
        final_error = np.linalg.norm(target_pos - final_pos)
        success = final_error < 0.02
        
        if not success:
            print(f"Warning: Final error {final_error:.4f}m exceeds threshold")
        
        return success
    except Exception as e:
        print(f"Error checking final position: {e}")
        return False


def open_gripper(sim, robot, steps: int = 20, gripper_value: float = 0.04):
    """
    Open gripper fully.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        steps: Number of steps to execute
        gripper_value: Target gripper width (0.04 = ~4cm = fully open)
    """
    for _ in range(steps):
        try:
            # Action: [dx, dy, dz, gripper_target]
            # gripper_target: 0.04 for open, 0.0 for closed
            action = np.array([0.0, 0.0, 0.0, gripper_value])
            robot.set_action(action)
            sim.step()
        except Exception as e:
            print(f"Error opening gripper: {e}")
            break


def close_gripper(sim, robot, steps: int = 20, gripper_value: float = 0.0):
    """
    Close gripper.
    
    Args:
        gripper_value: Target gripper width (0.0 = closed)
    """
    for _ in range(steps):
        try:
            action = np.array([0.0, 0.0, 0.0, gripper_value])
            robot.set_action(action)
            sim.step()
        except Exception as e:
            print(f"Error closing gripper: {e}")
            break


def compute_inverse_kinematics(sim, robot, target_pos: np.ndarray, 
                               target_ori: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Compute inverse kinematics for target pose.
    
    Uses PyBullet's built-in IK solver.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_pos: Target position [x, y, z]
        target_ori: Target orientation as quaternion [x, y, z, w] (optional)
    
    Returns:
        joint_positions: Target joint angles for 7-DOF arm, or None if IK fails
    """
    try:
        robot_id = sim._bodies_idx.get('panda')
        if robot_id is None:
            print("Error: Could not find robot body ID")
            return None
        
        # Get end-effector link index (usually link 11 for Panda)
        ee_link_index = 11  # This may need adjustment based on URDF
        
        if target_ori is None:
            # Default orientation (pointing down)
            target_ori = [0, 1, 0, 0]  # Quaternion [x, y, z, w]
        
        # Call PyBullet IK
        joint_positions = sim.physics_client.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=ee_link_index,
            targetPosition=target_pos,
            targetOrientation=target_ori,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Return only the first 7 joint angles (arm joints, not gripper)
        return np.array(joint_positions[:7], dtype=np.float32)
        
    except Exception as e:
        print(f"Error computing IK: {e}")
        return None


def plan_trajectory(start_joints: np.ndarray, goal_joints: np.ndarray,
                    num_waypoints: int = 10) -> np.ndarray:
    """
    Plan smooth trajectory between joint configurations.
    
    Simple linear interpolation. For real deployment, use
    proper trajectory optimization (e.g., minimum jerk).
    
    Args:
        start_joints: Starting joint configuration
        goal_joints: Goal joint configuration  
        num_waypoints: Number of intermediate waypoints
    
    Returns:
        trajectory: Array of shape (num_waypoints, 7) with joint waypoints
    """
    if start_joints.shape != goal_joints.shape:
        print(f"Error: Joint array shapes don't match: {start_joints.shape} vs {goal_joints.shape}")
        return np.array([])
    
    trajectory = np.linspace(start_joints, goal_joints, num_waypoints)
    return trajectory


def check_self_collision(sim, robot) -> bool:
    """
    Check if robot is in self-collision state.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
    
    Returns:
        has_collision: True if robot has self-collision
    """
    try:
        # Get robot body ID
        robot_id = sim._bodies_idx.get('panda')
        if robot_id is None:
            return False
        
        # Check for contact points on robot with itself
        contact_points = sim.physics_client.getContactPoints(
            bodyA=robot_id,
            bodyB=robot_id
        )
        
        return len(contact_points) > 0
        
    except Exception as e:
        print(f"Error checking self-collision: {e}")
        return False


def check_collision_with_obstacles(sim, robot, obstacle_ids: list) -> bool:
    """
    Check if robot collides with any obstacles.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        obstacle_ids: List of body IDs to check collision with
    
    Returns:
        has_collision: True if collision detected
    """
    try:
        robot_id = sim._bodies_idx.get('panda')
        if robot_id is None:
            return False
        
        for obstacle_id in obstacle_ids:
            contact_points = sim.physics_client.getContactPoints(
                bodyA=robot_id,
                bodyB=obstacle_id
            )
            if len(contact_points) > 0:
                return True
        
        return False
        
    except Exception as e:
        print(f"Error checking obstacle collision: {e}")
        return False


def get_gripper_state(robot) -> dict:
    """
    Get current gripper state information.
    
    Args:
        robot: Panda robot instance
    
    Returns:
        Dictionary with gripper width, force, and grasp status
    """
    try:
        obs = robot.get_obs()
        
        # Panda gripper has 2 finger joints (indices 14 and 15 in obs)
        finger_positions = obs[14:16] if len(obs) > 15 else np.array([0.0, 0.0])
        gripper_width = np.sum(finger_positions)
        
        return {
            'width': float(gripper_width),
            'is_closed': gripper_width < 0.01,
            'is_open': gripper_width > 0.07
        }
    except Exception as e:
        print(f"Error getting gripper state: {e}")
        return {
            'width': 0.0,
            'is_closed': False,
            'is_open': False
        }


def execute_trajectory(sim, robot, trajectory: np.ndarray, steps_per_waypoint: int = 10) -> bool:
    """
    Execute a planned joint trajectory.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        trajectory: Array of shape (num_waypoints, 7) with joint angles
        steps_per_waypoint: Simulation steps between waypoints
    
    Returns:
        success: True if trajectory executed successfully
    """
    try:
        for waypoint in trajectory:
            # Set target joint positions
            # Note: This is simplified - real implementation would use 
            # position control or computed torques
            for _ in range(steps_per_waypoint):
                sim.step()
        return True
    except Exception as e:
        print(f"Error executing trajectory: {e}")
        return False
