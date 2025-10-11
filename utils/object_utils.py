"""
Object-related utility functions for the Strategic Push-Grasp Environment.
Handles object state tracking, spatial relationships, and scene analysis.

TESTED VERSION - Fixed edge cases and error handling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_shape_descriptors(object_type: str, half_extents: np.ndarray = None, 
                              radius: float = None) -> np.ndarray:
    """
    Convert object geometric properties into a shape descriptor vector.
    
    Args:
        object_type: "cube", "sphere", or "irregular"
        half_extents: For cubes, the half extents [x, y, z]
        radius: For spheres, the radius
    
    Returns:
        Shape descriptor vector of size 8:
        [is_cube, is_sphere, is_irregular, dim1, dim2, dim3, volume, graspability_score]
    """
    descriptor = np.zeros(8, dtype=np.float32)
    
    if object_type == "cube":
        descriptor[0] = 1.0  # is_cube
        if half_extents is not None:
            descriptor[3:6] = half_extents * 2  # Convert to full dimensions
            descriptor[6] = np.prod(half_extents * 2)  # Volume
        descriptor[7] = 0.8  # Graspability score (cubes are easy to grasp)
        
    elif object_type == "sphere":
        descriptor[1] = 1.0  # is_sphere
        if radius is not None:
            descriptor[3] = radius * 2  # Diameter
            descriptor[4] = radius * 2
            descriptor[5] = radius * 2
            descriptor[6] = (4/3) * np.pi * (radius ** 3)  # Volume
        descriptor[7] = 0.3  # Graspability score (spheres are harder to grasp)
        
    elif object_type == "irregular":
        descriptor[2] = 1.0  # is_irregular
        descriptor[7] = 0.5  # Medium graspability
    
    return descriptor


def compute_pairwise_distance_matrix(sim, objects: Dict) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all objects.
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and their properties
    
    Returns:
        Distance matrix D ∈ ℝ^(N×N) where D[i,j] is distance between object i and j
    """
    object_names = sorted(objects.keys())
    N = len(object_names)
    
    # Handle empty case
    if N == 0:
        return np.array([], dtype=np.float32).reshape(0, 0)
    
    distance_matrix = np.zeros((N, N), dtype=np.float32)
    
    # Get all positions first (error handling)
    positions = []
    for name in object_names:
        try:
            pos = np.array(sim.get_base_position(name)[:2])  # Only x, y
            positions.append(pos)
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            positions.append(np.array([0.0, 0.0], dtype=np.float32))
    
    # Compute distances
    for i in range(N):
        for j in range(i + 1, N):  # Only upper triangle, then mirror
            dist = np.linalg.norm(positions[i] - positions[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric
    
    return distance_matrix


def compute_occlusion_masks(sim, objects: Dict, threshold: float = 0.05) -> np.ndarray:
    """
    Compute binary occlusion mask for all objects.
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and their properties
        threshold: Distance threshold for occlusion detection
    
    Returns:
        Occlusion mask O ∈ {0,1}^N where O[i]=1 if object i is occluded
    """
    object_names = sorted(objects.keys())
    N = len(object_names)
    
    # Handle empty case
    if N == 0:
        return np.array([], dtype=np.int32)
    
    occlusion_mask = np.zeros(N, dtype=np.int32)
    
    # Get all positions first (error handling)
    positions = []
    for name in object_names:
        try:
            pos = np.array(sim.get_base_position(name)[:2])
            positions.append(pos)
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            positions.append(np.array([0.0, 0.0], dtype=np.float32))
    
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(positions[i] - positions[j])
            
            if distance < threshold:
                occlusion_mask[i] = 1
                occlusion_mask[j] = 1
    
    return occlusion_mask


def analyze_scene_occlusions(sim, objects: Dict, threshold: float = 0.05) -> None:
    """
    Analyze and update occlusion status for all objects in the scene.
    Modifies the objects dictionary in-place.
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and their properties
        threshold: Distance threshold for occlusion detection
    """
    object_names = list(objects.keys())
    
    # Reset all occlusion flags
    for name in object_names:
        objects[name]["is_occluded"] = False
    
    # Handle empty case
    if len(object_names) == 0:
        return
    
    # Get all positions with error handling
    positions = {}
    for name in object_names:
        try:
            positions[name] = np.array(sim.get_base_position(name)[:2])
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            positions[name] = np.array([0.0, 0.0], dtype=np.float32)
    
    # Check pairwise occlusions
    for i in range(len(object_names)):
        for j in range(i + 1, len(object_names)):
            name1, name2 = object_names[i], object_names[j]
            distance = np.linalg.norm(positions[name1] - positions[name2])
            
            if distance < threshold:
                objects[name1]["is_occluded"] = True
                objects[name2]["is_occluded"] = True


def update_object_colors(sim, objects: Dict, 
                        color_green: List[float], 
                        color_yellow: List[float], 
                        color_red: List[float]) -> None:
    """
    Update object colors based on type and occlusion state.
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and their properties
        color_green: RGBA color for cubes [r, g, b, a]
        color_yellow: RGBA color for spheres
        color_red: RGBA color for occluded objects
    """
    for name, properties in objects.items():
        color_to_set = None
        
        # Occlusion takes priority
        if properties.get("is_occluded", False):
            color_to_set = color_red
        elif properties.get("type") == "cube":
            color_to_set = color_green
        elif properties.get("type") == "sphere":
            color_to_set = color_yellow
        
        if color_to_set is not None:
            try:
                body_id = sim._bodies_idx.get(name)
                if body_id is not None:
                    sim.physics_client.changeVisualShape(
                        body_id, -1, rgbaColor=color_to_set
                    )
            except Exception as e:
                print(f"Warning: Could not change color for {name}: {e}")


def select_target_heuristic(sim, objects: Dict, goal_pos: np.ndarray, 
                           collected_objects: set = None) -> Optional[str]:
    """
    Heuristic target selection: Select the nearest uncollected object to the goal.
    This is NOT learned - it's a simple rule to reduce the action space.
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and their properties
        goal_pos: Goal position [x, y]
        collected_objects: Set of object names already collected
    
    Returns:
        Name of the selected target object, or None if no valid target
    """
    if collected_objects is None:
        collected_objects = set()
    
    uncollected = [name for name in objects.keys() if name not in collected_objects]
    
    if not uncollected:
        return None
    
    # Find object closest to goal
    min_distance = float('inf')
    target = None
    
    for name in uncollected:
        try:
            pos = np.array(sim.get_base_position(name)[:2])
            distance = np.linalg.norm(pos - goal_pos)
            
            if distance < min_distance:
                min_distance = distance
                target = name
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            continue
    
    return target


def get_safe_spawn_position(sim, existing_objects: List[str], 
                           goal_pos: np.ndarray, goal_size: float,
                           min_separation: float = 0.08,
                           workspace_bounds: Tuple[float, float] = (-0.3, 0.3),
                           max_attempts: int = 100) -> np.ndarray:
    """
    Generate a safe spawn position that avoids goal area and other objects.
    
    Args:
        sim: PyBullet simulation instance
        existing_objects: List of existing object names
        goal_pos: Goal position [x, y]
        goal_size: Size of goal square
        min_separation: Minimum distance between objects
        workspace_bounds: (min, max) bounds for x and y
        max_attempts: Maximum number of random sampling attempts
    
    Returns:
        Safe spawn position [x, y, z]
    """
    half_goal = goal_size / 2
    
    for attempt in range(max_attempts):
        # Random position in workspace
        pos_xy = np.random.uniform(
            low=workspace_bounds[0], 
            high=workspace_bounds[1], 
            size=2
        )
        
        # Check if in goal area
        in_goal = (
            (goal_pos[0] - half_goal < pos_xy[0] < goal_pos[0] + half_goal) and
            (goal_pos[1] - half_goal < pos_xy[1] < goal_pos[1] + half_goal)
        )
        
        if in_goal:
            continue
        
        # Check distance to existing objects
        is_valid = True
        for obj_name in existing_objects:
            try:
                obj_pos = np.array(sim.get_base_position(obj_name)[:2])
                distance = np.linalg.norm(pos_xy - obj_pos)
                
                if distance < min_separation:
                    is_valid = False
                    break
            except Exception as e:
                print(f"Warning: Could not check distance for {obj_name}: {e}")
                continue
        
        if is_valid:
            return np.array([pos_xy[0], pos_xy[1], 0.02], dtype=np.float32)
    
    # Fallback: return a position even if not ideal
    print(f"Warning: Could not find ideal spawn position after {max_attempts} attempts")
    fallback_pos = np.array([
        np.random.uniform(workspace_bounds[0], workspace_bounds[1]),
        np.random.uniform(workspace_bounds[0], workspace_bounds[1]),
        0.02
    ], dtype=np.float32)
    return fallback_pos


def check_object_in_goal(object_pos: np.ndarray, goal_pos: np.ndarray, 
                        goal_size: float) -> bool:
    """
    Check if an object is within the goal square.
    
    Args:
        object_pos: Object position [x, y, z] or [x, y]
        goal_pos: Goal center position [x, y]
        goal_size: Side length of goal square
    
    Returns:
        True if object is in goal area
    """
    half_size = goal_size / 2
    obj_xy = object_pos[:2]
    
    return bool(
        (goal_pos[0] - half_size < obj_xy[0] < goal_pos[0] + half_size) and
        (goal_pos[1] - half_size < obj_xy[1] < goal_pos[1] + half_size)
    )
