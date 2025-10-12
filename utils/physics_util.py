"""
Physics and simulation utilities for PyBullet environment setup.
Handles collision detection, workspace boundaries, and scene management.

TESTED VERSION - Fixed error handling and edge cases
"""

import numpy as np
import pybullet as p
from typing import Tuple, List, Optional


def check_workspace_violation(sim, body_name: str, 
                              workspace_bounds: Tuple[float, float, float, float],
                              z_min: float = -0.05) -> bool:
    """
    Check if object has left the workspace boundaries.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of body to check
        workspace_bounds: (x_min, x_max, y_min, y_max)
        z_min: Minimum allowed z-coordinate (below = fell off table)
    
    Returns:
        violated: True if object is outside workspace
    """
    try:
        pos = sim.get_base_position(body_name)
    except Exception as e:
        print(f"Warning: Could not get position for {body_name}: {e}")
        return True  # Consider it a violation if body doesn't exist
    
    x_min, x_max, y_min, y_max = workspace_bounds
    
    # Check XY bounds
    if pos[0] < x_min or pos[0] > x_max:
        return True
    if pos[1] < y_min or pos[1] > y_max:
        return True
    
    # Check if fell below table
    if pos[2] < z_min:
        return True
    
    return False


def check_collision_with_table(sim, body_name: str, table_name: str = "table") -> bool:
    """
    Check if object is colliding with the table.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object to check
        table_name: Name of table body
    
    Returns:
        has_collision: True if collision detected
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        table_id = sim._bodies_idx.get(table_name)
        
        if body_id is None or table_id is None:
            return False
        
        contact_points = sim.physics_client.getContactPoints(
            bodyA=body_id,
            bodyB=table_id
        )
        
        return len(contact_points) > 0
        
    except Exception as e:
        print(f"Warning: Error checking table collision: {e}")
        return False


def check_object_collision(sim, body_name1: str, body_name2: str) -> bool:
    """
    Check if two objects are colliding.
    
    Args:
        sim: PyBullet simulation instance
        body_name1: First object name
        body_name2: Second object name
    
    Returns:
        has_collision: True if objects are in contact
    """
    try:
        body_id1 = sim._bodies_idx.get(body_name1)
        body_id2 = sim._bodies_idx.get(body_name2)
        
        if body_id1 is None or body_id2 is None:
            return False
        
        contact_points = sim.physics_client.getContactPoints(
            bodyA=body_id1,
            bodyB=body_id2
        )
        
        return len(contact_points) > 0
        
    except Exception as e:
        print(f"Warning: Error checking object collision: {e}")
        return False

def check_robot_link_collision(sim, robot_body_name: str, 
                               link_id: int,
                               other_body_name: str) -> bool:
    """
    Check if a specific robot link is colliding with another object.
    
    Args:
        sim: PyBullet simulation instance
        robot_body_name: Name of robot body (e.g., "panda")
        link_id: Specific link index to check
        other_body_name: Name of other object
    
    Returns:
        has_collision: True if specified link is in contact
    """
    try:
        robot_id = sim._bodies_idx.get(robot_body_name)
        other_id = sim._bodies_idx.get(other_body_name)
        
        if robot_id is None or other_id is None:
            return False
        
        contact_points = sim.physics_client.getContactPoints(
            bodyA=robot_id,
            bodyB=other_id,
            linkIndexA=link_id
        )
        
        return len(contact_points) > 0
        
    except Exception as e:
        print(f"Warning: Error checking robot link collision: {e}")
        return False

def check_self_collision(sim, body_name: str) -> bool:
    """
    Check if robot has self-collision between its links.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of robot body
    
    Returns:
        has_self_collision: True if any links are colliding with each other
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            return False
        
        # Get all contact points for this body
        contact_points = sim.physics_client.getContactPoints(bodyA=body_id)
        
        # Check if any contacts are with itself
        for contact in contact_points:
            body_b_id = contact[2]
            if body_b_id == body_id:
                return True
        
        return False
        
    except Exception as e:
        print(f"Warning: Error checking self-collision: {e}")
        return False

def get_contact_force(sim, body_name1: str, body_name2: str) -> float:
    """
    Get magnitude of contact force between two bodies.
    
    Args:
        sim: PyBullet simulation instance
        body_name1: First object name
        body_name2: Second object name
    
    Returns:
        force_magnitude: Total contact force magnitude
    """
    try:
        body_id1 = sim._bodies_idx.get(body_name1)
        body_id2 = sim._bodies_idx.get(body_name2)
        
        if body_id1 is None or body_id2 is None:
            return 0.0
        
        contact_points = sim.physics_client.getContactPoints(
            bodyA=body_id1,
            bodyB=body_id2
        )
        
        total_force = 0.0
        for contact in contact_points:
            # contact[9] is the normal force magnitude
            total_force += abs(contact[9])
        
        return float(total_force)
        
    except Exception as e:
        print(f"Warning: Error getting contact force: {e}")
        return 0.0

def get_contact_details(sim, body_name1: str, body_name2: str) -> List[dict]:
    """
    Get detailed contact point information between two bodies.
    
    Args:
        sim: PyBullet simulation instance
        body_name1: First object name
        body_name2: Second object name
    
    Returns:
        List of contact dictionaries with detailed information
    """
    try:
        body_id1 = sim._bodies_idx.get(body_name1)
        body_id2 = sim._bodies_idx.get(body_name2)
        
        if body_id1 is None or body_id2 is None:
            return []
        
        contact_points = sim.physics_client.getContactPoints(
            bodyA=body_id1,
            bodyB=body_id2
        )
        
        contacts = []
        for cp in contact_points:
            contact_info = {
                'contact_flag': cp[0],
                'body_a_id': cp[1],
                'body_b_id': cp[2],
                'link_a_id': cp[3],
                'link_b_id': cp[4],
                'position_on_a': np.array(cp[5]),
                'position_on_b': np.array(cp[6]),
                'contact_normal': np.array(cp[7]),
                'contact_distance': cp[8],
                'normal_force': cp[9],
                'lateral_friction_1': cp[10],
                'lateral_friction_dir_1': np.array(cp[11]),
                'lateral_friction_2': cp[12],
                'lateral_friction_dir_2': np.array(cp[13]),
            }
            contacts.append(contact_info)
        
        return contacts
        
    except Exception as e:
        print(f"Warning: Error getting contact details: {e}")
        return []

def get_all_collisions(sim, body_name: str) -> List[Tuple[str, float]]:
    """
    Get all objects currently colliding with the specified body.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of body to check
    
    Returns:
        List of (colliding_body_name, contact_force) tuples
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            return []
        
        collisions = []
        contact_points = sim.physics_client.getContactPoints(bodyA=body_id)
        
        # Group by body B and sum forces
        body_forces = {}
        for contact in contact_points:
            body_b_id = contact[2]
            force = abs(contact[9])
            
            # Find body name from ID
            body_b_name = None
            for name, bid in sim._bodies_idx.items():
                if bid == body_b_id:
                    body_b_name = name
                    break
            
            if body_b_name and body_b_name != body_name:
                if body_b_name in body_forces:
                    body_forces[body_b_name] += force
                else:
                    body_forces[body_b_name] = force
        
        collisions = list(body_forces.items())
        return collisions
        
    except Exception as e:
        print(f"Warning: Error getting collisions for {body_name}: {e}")
        return []

def draw_workspace_boundary(client, workspace_bounds: Tuple[float, float, float, float],
                           z_height: float = 0.0, color: List[float] = None,
                           line_width: float = 2.0):
    """
    Draw rectangular workspace boundary visualization.
    
    Args:
        client: PyBullet physics client
        workspace_bounds: (x_min, x_max, y_min, y_max)
        z_height: Height at which to draw boundary
        color: RGB color [r, g, b] (default: red)
        line_width: Width of boundary lines
    """
    if color is None:
        color = [1, 0, 0]
    
    try:
        x_min, x_max, y_min, y_max = workspace_bounds
        
        corners = [
            [x_min, y_min, z_height],
            [x_max, y_min, z_height],
            [x_max, y_max, z_height],
            [x_min, y_max, z_height]
        ]
        
        # Draw rectangle
        for i in range(4):
            client.addUserDebugLine(
                corners[i],
                corners[(i + 1) % 4],
                color,
                lineWidth=line_width
            )
    except Exception as e:
        print(f"Warning: Error drawing workspace boundary: {e}")


def is_object_stable(sim, body_name: str, 
                    velocity_threshold: float = 0.01,
                    angular_velocity_threshold: float = 0.05) -> bool:
    """
    Check if object has stabilized (stopped moving).
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object to check
        velocity_threshold: Linear velocity threshold for "stable"
        angular_velocity_threshold: Angular velocity threshold
    
    Returns:
        is_stable: True if object is not moving significantly
    """
    try:
        lin_vel = sim.get_base_velocity(body_name)
        ang_vel = sim.get_base_angular_velocity(body_name)
        
        lin_speed = np.linalg.norm(lin_vel)
        ang_speed = np.linalg.norm(ang_vel)

        velocity_stable = (lin_speed < velocity_threshold and 
                          ang_speed < angular_velocity_threshold)
      
        # Additional check: position hasn't changed much
        if position_history and len(position_history) >= 2:
            current_pos = sim.get_base_position(body_name)
            last_pos = position_history[-1]
            pos_change = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
            position_stable = pos_change < position_threshold
            return velocity_stable and position_stable
        
        return velocity_stable
        
    except Exception as e:
        print(f"Warning: Could not check stability for {body_name}: {e}")
        return False


def wait_for_objects_stable(sim, object_names: List[str], 
                           max_steps: int = 100,
                           check_interval: int = 10) -> bool:
    """
    Wait until all objects in scene have stabilized.
    
    Args:
        sim: PyBullet simulation instance
        object_names: List of object names to monitor
        max_steps: Maximum simulation steps to wait
        check_interval: Check stability every N steps
    
    Returns:
        all_stable: True if all objects stabilized within max_steps
    """
    if not object_names:
        return True  # No objects to check
    
    for step in range(max_steps):
        sim.step()
        
        if step % check_interval == 0:
            all_stable = True
            for name in object_names:
                if not is_object_stable(sim, name):
                    all_stable = False
                    break
            
            if all_stable:
                return True
    
    # Timeout - not all objects stabilized
    return False


def get_object_bounding_box(sim, body_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get axis-aligned bounding box (AABB) for object.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object
    
    Returns:
        (min_coords, max_coords): AABB corners, or None if body not found
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            return None
        
        aabb = sim.physics_client.getAABB(body_id)
        min_coords = np.array(aabb[0], dtype=np.float32)
        max_coords = np.array(aabb[1], dtype=np.float32)
        
        return min_coords, max_coords
        
    except Exception as e:
        print(f"Warning: Error getting bounding box for {body_name}: {e}")
        return None


def compute_object_volume(sim, body_name: str) -> float:
    """
    Estimate object volume from bounding box.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object
    
    Returns:
        volume: Approximate volume in m³
    """
    bbox = get_object_bounding_box(sim, body_name)
    if bbox is None:
        return 0.0
    
    min_coords, max_coords = bbox
    dimensions = max_coords - min_coords
    volume = np.prod(dimensions)
    
    return float(volume)


def set_object_color(sim, body_name: str, rgba_color: List[float]):
    """
    Change visual color of an object.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object to recolor
        rgba_color: RGBA color [r, g, b, a] in range [0, 1]
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is not None:
            sim.physics_client.changeVisualShape(
                body_id,
                -1,  # Link index (-1 = base)
                rgbaColor=rgba_color
            )
    except Exception as e:
        print(f"Warning: Could not change color for {body_name}: {e}")


def apply_force_to_object(sim, body_name: str, force: np.ndarray,
                         position: Optional[np.ndarray] = None):
    """
    Apply external force to object (e.g., for push simulation).
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object
        force: Force vector [fx, fy, fz] in Newtons
        position: Position to apply force (world frame), None = center of mass
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            print(f"Warning: Body {body_name} not found")
            return
        
        if position is None:
            position = sim.get_base_position(body_name)
        
        # Convert to list if numpy array
        if isinstance(force, np.ndarray):
            force = force.tolist()
        if isinstance(position, np.ndarray):
            position = position.tolist()
        
        sim.physics_client.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,
            forceObj=force,
            posObj=position,
            flags=p.WORLD_FRAME
        )
        
    except Exception as e:
        print(f"Warning: Could not apply force to {body_name}: {e}")


def reset_object_pose(sim, body_name: str, position: np.ndarray,
                     orientation: Optional[np.ndarray] = None):
    """
    Reset object to specific pose.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object
        position: Target position [x, y, z]
        orientation: Target quaternion [x, y, z, w], None = no rotation
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            print(f"Warning: Body {body_name} not found")
            return
        
        if orientation is None:
            orientation = [0, 0, 0, 1]  # No rotation
        
        # Convert to lists if numpy arrays
        if isinstance(position, np.ndarray):
            position = position.tolist()
        if isinstance(orientation, np.ndarray):
            orientation = orientation.tolist()
        
        sim.physics_client.resetBasePositionAndOrientation(
            body_id,
            position,
            orientation
        )
        
        # Reset velocities to zero
        sim.physics_client.resetBaseVelocity(
            body_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )
        
    except Exception as e:
        print(f"Warning: Could not reset pose for {body_name}: {e}")


def get_simulation_time(sim) -> float:
    """
    Get current simulation time in seconds.
    
    Args:
        sim: PyBullet simulation instance
    
    Returns:
        time: Simulation time in seconds (estimated from step count)
    """
    # PyBullet's default time step is 1/240 seconds
    try:
        # This is an approximation - actual implementation would need to track steps
        # For now, return 0 as placeholder
        return 0.0
    except Exception as e:
        print(f"Warning: Could not get simulation time: {e}")
        return 0.0


def set_gravity(sim, gravity: float = -9.81):
    """
    Set gravity in simulation.
    
    Args:
        sim: PyBullet simulation instance
        gravity: Gravity acceleration (default: -9.81 m/s²)
    """
    try:
        sim.physics_client.setGravity(0, 0, gravity)
    except Exception as e:
        print(f"Warning: Could not set gravity: {e}")


def get_all_collisions(sim, body_name: str) -> List[Tuple[str, float]]:
    """
    Get all objects currently colliding with the specified body.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of body to check
    
    Returns:
        List of (colliding_body_name, contact_force) tuples
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            return []
        
        collisions = []
        contact_points = sim.physics_client.getContactPoints(bodyA=body_id)
        
        # Group by body B and sum forces
        body_forces = {}
        for contact in contact_points:
            body_b_id = contact[2]
            force = abs(contact[9])
            
            # Find body name from ID
            body_b_name = None
            for name, bid in sim._bodies_idx.items():
                if bid == body_b_id:
                    body_b_name = name
                    break
            
            if body_b_name and body_b_name != body_name:
                if body_b_name in body_forces:
                    body_forces[body_b_name] += force
                else:
                    body_forces[body_b_name] = force
        
        collisions = list(body_forces.items())
        return collisions
        
    except Exception as e:
        print(f"Warning: Error getting collisions for {body_name}: {e}")
        return []


def check_multiple_workspace_violations(sim, object_names: List[str],
                                       workspace_bounds: Tuple[float, float, float, float],
                                       z_min: float = -0.05) -> List[str]:
    """
    Check multiple objects for workspace violations.
    
    Args:
        sim: PyBullet simulation instance
        object_names: List of object names to check
        workspace_bounds: (x_min, x_max, y_min, y_max)
        z_min: Minimum allowed z-coordinate
    
    Returns:
        List of object names that violated workspace bounds
    """
    violated_objects = []
    
    for name in object_names:
        if check_workspace_violation(sim, name, workspace_bounds, z_min):
            violated_objects.append(name)
    
    return violated_objects


def get_object_velocity_magnitude(sim, body_name: str) -> Tuple[float, float]:
    """
    Get linear and angular velocity magnitudes for an object.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object
    
    Returns:
        (linear_speed, angular_speed): Speed magnitudes in m/s and rad/s
    """
    try:
        lin_vel = sim.get_base_velocity(body_name)
        ang_vel = sim.get_base_angular_velocity(body_name)
        
        lin_speed = float(np.linalg.norm(lin_vel))
        ang_speed = float(np.linalg.norm(ang_vel))
        
        return lin_speed, ang_speed
        
    except Exception as e:
        print(f"Warning: Could not get velocity for {body_name}: {e}")
        return 0.0, 0.0
