"""
Object-related utility functions for the Strategic Push-Grasp Environment.
Handles object state tracking, spatial relationships, and scene analysis.

This module provides all object-related computations needed for the RL environment:
- Shape descriptor encoding for neural network input
- Spatial relationship computation (distances, occlusions)
- Scene analysis for visual feedback
- Heuristic-based target selection
- Safe object spawning logic

MODIFIED: Retains only 'cube' type logic for shape descriptors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_shape_descriptors(object_type: str, half_extents: np.ndarray = None,
                              radius: float = None) -> np.ndarray:
    """
    MODIFIED: Convert object geometric properties into a fixed-size shape descriptor vector,
    retaining only CUBE logic.

    Shape Descriptor Structure (8D):
    [0]=is_cube, [1]=is_sphere, [2]=is_irregular, [3]=dim1, [4]=dim2,
    [5]=dim3, [6]=volume, [7]=graspability_score
    """
    descriptor = np.zeros(8, dtype=np.float32)

    # ========================================================================
    # CUBE ENCODING
    # ========================================================================
    if object_type == "cube":
        descriptor[0] = 1.0 # One-hot: is_cube
        if half_extents is not None:
            # Convert half-extents to full dimensions
            full_dims = half_extents * 2
            descriptor[3:6] = full_dims # [width, depth, height]

            # Compute volume: V = width × depth × height
            descriptor[6] = np.prod(full_dims)
        else:
            print("Warning: cube created without half_extents")

        # Graspability: 0.8 (cubes are easy to grasp)
        descriptor[7] = 0.8

    # ========================================================================
    # FALLBACK: Removed SPHERE and IRREGULAR ENCODING
    # ========================================================================
    else:
        print(f"Warning: Unexpected object_type '{object_type}' encountered. Returning zero descriptor.")

    return descriptor

def compute_pairwise_distance_matrix(sim, objects: Dict) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all objects (vectorized).
    (Logic remains unchanged)
    """
    object_names = sorted(objects.keys())  # Consistent ordering
    N = len(object_names)

    if N == 0:
        return np.array([], dtype=np.float32).reshape(0, 0)

    positions = np.zeros((N, 2), dtype=np.float32)

    for i, name in enumerate(object_names):
        try:
            positions[i] = sim.get_base_position(name)[:2]  # [x, y]
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            positions[i] = [0.0, 0.0]

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    distance_matrix = np.linalg.norm(diff, axis=2).astype(np.float32)

    return distance_matrix

def compute_occlusion_masks(sim, objects: Dict, threshold: float = 0.05) -> np.ndarray:
    """
    Compute binary occlusion mask for all objects.
    (Logic remains unchanged)
    """
    object_names = sorted(objects.keys())
    N = len(object_names)

    if N == 0:
        return np.array([], dtype=np.int32)

    occlusion_mask = np.zeros(N, dtype=np.int32)
    positions = []

    for name in object_names:
        try:
            pos = np.array(sim.get_base_position(name)[:2])
            positions.append(pos)
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            positions.append(np.array([0.0, 0.0], dtype=np.float32))

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
    (Logic remains unchanged)
    """
    object_names = list(objects.keys())

    for name in object_names:
        objects[name]["is_occluded"] = False

    if len(object_names) == 0:
        return

    positions = {}
    for name in object_names:
        try:
            positions[name] = np.array(sim.get_base_position(name)[:2])
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            positions[name] = np.array([0.0, 0.0], dtype=np.float32)

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
    (Logic remains unchanged)
    """
    for name, properties in objects.items():
        color_to_set = None

        if properties.get("is_occluded", False):
            color_to_set = color_red

        elif properties.get("type") == "cube":
            color_to_set = color_green
        elif properties.get("type") == "sphere":
            # Sphere logic is technically kept for robustness, but should not be called
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
    (Logic remains unchanged)
    """
    if collected_objects is None:
        collected_objects = set()

    uncollected = [name for name in objects.keys() if name not in collected_objects]

    if not uncollected:
        return None

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
                           workspace_bounds: Tuple[float, float, float, float] = (-0.3, 0.3, -0.3, 0.3),
                           max_attempts: int = 100) -> np.ndarray:
    """
    Generate a safe spawn position that avoids goal area and other objects.
    (Logic remains unchanged)
    """
    x_min, x_max, y_min, y_max = workspace_bounds
    half_goal = goal_size / 2

    for attempt in range(max_attempts):
        pos_xy = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ], dtype=np.float32)

        in_goal = (
            (goal_pos[0] - half_goal < pos_xy[0] < goal_pos[0] + half_goal) and
            (goal_pos[1] - half_goal < pos_xy[1] < goal_pos[1] + half_goal)
        )

        if in_goal:
            continue

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

    print(f"Warning: Could not find ideal spawn position after {max_attempts} attempts")
    fallback_pos = np.array([
        np.random.uniform(x_min, x_max),
        np.random.uniform(y_min, y_max),
        0.02
    ], dtype=np.float32)
    return fallback_pos

def check_object_in_goal(object_pos: np.ndarray, goal_pos: np.ndarray,
                        goal_size: float) -> bool:
    """
    Check if an object is within the goal square.
    (Logic remains unchanged)
    """
    half_size = goal_size / 2
    obj_xy = object_pos[:2]

    return bool(
        (goal_pos[0] - half_size < obj_xy[0] < goal_pos[0] + half_size) and
        (goal_pos[1] - half_size < obj_xy[1] < goal_pos[1] + half_size)
    )