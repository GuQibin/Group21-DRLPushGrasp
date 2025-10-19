"""
Object-related utility functions for the Strategic Push-Grasp Environment.
Handles object state tracking, spatial relationships, and scene analysis.

This module provides all object-related computations needed for the RL environment:
- Shape descriptor encoding for neural network input
- Spatial relationship computation (distances, occlusions)
- Scene analysis for visual feedback
- Heuristic-based target selection
- Safe object spawning logic

TESTED VERSION - Fixed edge cases and error handling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_shape_descriptors(object_type: str, half_extents: np.ndarray = None, 
                              radius: float = None) -> np.ndarray:
    """
    Convert object geometric properties into a fixed-size shape descriptor vector.
    
    This encoding allows the neural network to reason about object geometry without
    needing separate encoders for different object types. The descriptor captures:
    - Object type (one-hot encoding)
    - Physical dimensions
    - Volume (useful for mass/physics reasoning)
    - Graspability heuristic (guides initial exploration)
    
    Shape Descriptor Structure (8D):
    ┌──────────────────────────────────────────────────────────────┐
    │ Index │ Field              │ Description                     │
    ├───────┼────────────────────┼─────────────────────────────────┤
    │ [0]   │ is_cube            │ 1.0 if cube, else 0.0           │
    │ [1]   │ is_sphere          │ 1.0 if sphere, else 0.0         │
    │ [2]   │ is_irregular       │ 1.0 if irregular, else 0.0      │
    │ [3]   │ dim1               │ Width/diameter (m)              │
    │ [4]   │ dim2               │ Depth/diameter (m)              │
    │ [5]   │ dim3               │ Height/diameter (m)             │
    │ [6]   │ volume             │ Object volume (m³)              │
    │ [7]   │ graspability_score │ 0.0-1.0 (higher = easier grasp) │
    └──────────────────────────────────────────────────────────────┘
    
    Args:
        object_type: "cube", "sphere", or "irregular"
        half_extents: Half-sizes [x, y, z] for cubes (full size = 2 × half_extents)
        radius: Radius for spheres (meters)
    
    Returns:
        Shape descriptor vector (8D) as float32 numpy array
    
    Graspability Heuristics:
    - Cubes: 0.8 (high) - flat surfaces, stable grasps
    - Spheres: 0.3 (low) - rolling tendency, harder to grasp
    - Irregular: 0.5 (medium) - case-by-case
    
    Design Note:
        This is a simple hand-crafted encoding. For complex shapes, could be replaced
        with learned embeddings from a point cloud encoder or shape autoencoder.
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
            # Validation warning - cube should always have dimensions
            print("Warning: cube created without half_extents")

        # Graspability: 0.8 (cubes are easy to grasp)
        # Rationale: Flat surfaces, stable, predictable contact
        descriptor[7] = 0.8
        
    # ========================================================================
    # SPHERE ENCODING
    # ========================================================================
    elif object_type == "sphere":
        descriptor[1] = 1.0  # One-hot: is_sphere
        
        if radius is not None:
            diameter = radius * 2
            # Spheres have same dimension in all axes
            descriptor[3:6] = diameter
            
            # Compute volume: V = (4/3)πr³
            descriptor[6] = (4/3) * np.pi * (radius ** 3)
        else:
            # Validation warning - sphere should always have radius
            print("Warning: sphere created without radius")
        
        # Graspability: 0.3 (spheres are hard to grasp)
        # Rationale: Rolling tendency, point contact, slip easily
        # → Agent should learn to prefer pushing spheres
        descriptor[7] = 0.3
    
    # ========================================================================
    # IRREGULAR OBJECT ENCODING
    # ========================================================================
    elif object_type == "irregular":
        descriptor[2] = 1.0  # One-hot: is_irregular
        
        # Medium graspability (depends on specific shape)
        descriptor[7] = 0.5
    
    # ========================================================================
    # VALIDATION - Unknown object type
    # ========================================================================
    else:
        print(f"Warning: Unknown object_type '{object_type}'")
        # Returns zero vector by default
    
    return descriptor

def compute_pairwise_distance_matrix(sim, objects: Dict) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between all objects (vectorized).
    
    Uses NumPy broadcasting for efficient computation instead of nested loops.
    This matrix is used for:
    - Occlusion detection (objects too close may block each other)
    - Spatial reasoning in the neural network
    - Collision avoidance planning
    
    Algorithm:
    1. Extract all object positions into (N, 2) array
    2. Use broadcasting to compute all pairwise differences: (N, 1, 2) - (1, N, 2)
    3. Compute norms along axis 2 to get (N, N) distance matrix
    
    Time Complexity: O(N²) space, O(N²) time (unavoidable for all-pairs distances)
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and properties
    
    Returns:
        Distance matrix D ∈ ℝ^(N×N) where D[i,j] = ||pos_i - pos_j||₂
        - Diagonal entries are zero (distance to self)
        - Matrix is symmetric: D[i,j] = D[j,i]
        - Returns empty (0, 0) array if no objects
    
    Example:
        For 3 objects at positions [(0,0), (1,0), (0,1)]:
        D = [[0.0, 1.0, 1.0],
             [1.0, 0.0, 1.414],
             [1.0, 1.414, 0.0]]
    """
    object_names = sorted(objects.keys())  # Consistent ordering
    N = len(object_names)
    
    # Handle empty scene
    if N == 0:
        return np.array([], dtype=np.float32).reshape(0, 0)
    
    # ========================================================================
    # EXTRACT ALL POSITIONS (XY only for tabletop)
    # ========================================================================
    # We only use XY coordinates since all objects are on the table surface
    # Z-coordinate is not relevant for proximity/occlusion
    positions = np.zeros((N, 2), dtype=np.float32)
    
    for i, name in enumerate(object_names):
        try:
            positions[i] = sim.get_base_position(name)[:2]  # [x, y]
        except Exception as e:
            # Graceful fallback: place at origin if position query fails
            print(f"Warning: Could not get position for {name}: {e}")
            positions[i] = [0.0, 0.0]
    
    # ========================================================================
    # VECTORIZED DISTANCE COMPUTATION (FAST)
    # ========================================================================
    # Broadcasting magic:
    # positions[:, np.newaxis, :]  → shape (N, 1, 2)
    # positions[np.newaxis, :, :]  → shape (1, N, 2)
    # Subtraction broadcasts to (N, N, 2) with all pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    
    # Compute Euclidean norm along last axis (axis=2)
    # Result: (N, N) matrix of distances
    distance_matrix = np.linalg.norm(diff, axis=2).astype(np.float32)
    
    return distance_matrix

def compute_occlusion_masks(sim, objects: Dict, threshold: float = 0.05) -> np.ndarray:
    """
    Compute binary occlusion mask for all objects.
    
    Occlusion Definition:
    An object is "occluded" if any other object is within the threshold distance.
    This is a simplified geometric occlusion model suitable for tabletop scenarios.
    
    In real-world terms:
    - Objects closer than 5cm may block each other during grasping
    - Robot should push occluded objects apart before attempting grasp
    - Visual feedback: occluded objects turn RED
    
    Limitation:
    This is a proximity-based heuristic, not true line-of-sight occlusion.
    For vision-based systems, would need ray-casting or view-frustum checks.
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and their properties
        threshold: Distance threshold for occlusion detection (default: 5cm)
    
    Returns:
        Occlusion mask O ∈ {0,1}^N where:
        - O[i] = 1 if object i is occluded (has neighbor within threshold)
        - O[i] = 0 if object i is free (no nearby obstacles)
        - Returns empty array if no objects
    
    Example:
        For 3 objects where obj0 and obj1 are 3cm apart, obj2 is isolated:
        → occlusion_mask = [1, 1, 0]
    """
    object_names = sorted(objects.keys())
    N = len(object_names)
    
    if N == 0:
        return np.array([], dtype=np.int32)

    occlusion_mask = np.zeros(N, dtype=np.int32)
    
    # ========================================================================
    # GET ALL POSITIONS (XY only)
    # ========================================================================
    positions = []
    for name in object_names:
        try:
            pos = np.array(sim.get_base_position(name)[:2])
            positions.append(pos)
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            positions.append(np.array([0.0, 0.0], dtype=np.float32))
    
    # ========================================================================
    # CHECK PAIRWISE DISTANCES
    # ========================================================================
    # If any pair of objects is closer than threshold, BOTH are marked occluded
    # This is conservative: assumes occlusion is bidirectional
    for i in range(N):
        for j in range(i + 1, N):  # Only check upper triangle (avoid duplicates)
            distance = np.linalg.norm(positions[i] - positions[j])
            
            if distance < threshold:
                # Mark BOTH objects as occluded
                occlusion_mask[i] = 1
                occlusion_mask[j] = 1
    
    return occlusion_mask

def analyze_scene_occlusions(sim, objects: Dict, threshold: float = 0.05) -> None:
    """
    Analyze and update occlusion status for all objects in the scene.
    Modifies the objects dictionary in-place.
    
    This is the high-level function called every step to update scene state.
    It updates the "is_occluded" field in each object's metadata dictionary.
    
    Key Difference from compute_occlusion_masks():
    - This function modifies objects dict IN-PLACE (side effect)
    - compute_occlusion_masks() returns a mask without modifying anything
    
    Use Cases:
    - Called in reset() to analyze initial scene
    - Called in step() after each action to track changes
    - Used before update_object_colors() to determine visual feedback
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and properties (MODIFIED IN-PLACE)
        threshold: Distance threshold for occlusion detection (default: 5cm)
    
    Returns:
        None (modifies objects dictionary directly)
    
    Side Effect:
        Sets objects[name]["is_occluded"] = True/False for all objects
    """
    object_names = list(objects.keys())
    
    # ========================================================================
    # RESET ALL OCCLUSION FLAGS
    # ========================================================================
    # Important: Clear previous state before recomputing
    # Otherwise, false positives would accumulate across steps
    for name in object_names:
        objects[name]["is_occluded"] = False
    
    # ========================================================================
    # HANDLE EMPTY SCENE
    # ========================================================================
    if len(object_names) == 0:
        return
    
    # ========================================================================
    # GET ALL POSITIONS WITH ERROR HANDLING
    # ========================================================================
    positions = {}
    for name in object_names:
        try:
            positions[name] = np.array(sim.get_base_position(name)[:2])
        except Exception as e:
            print(f"Warning: Could not get position for {name}: {e}")
            # Fallback: place at origin (prevents crash)
            positions[name] = np.array([0.0, 0.0], dtype=np.float32)
    
    # ========================================================================
    # CHECK PAIRWISE OCCLUSIONS
    # ========================================================================
    # Same logic as compute_occlusion_masks(), but modifies objects dict
    for i in range(len(object_names)):
        for j in range(i + 1, len(object_names)):
            name1, name2 = object_names[i], object_names[j]
            distance = np.linalg.norm(positions[name1] - positions[name2])
            
            if distance < threshold:
                # Mark BOTH objects as occluded
                objects[name1]["is_occluded"] = True
                objects[name2]["is_occluded"] = True

def update_object_colors(sim, objects: Dict, 
                        color_green: List[float], 
                        color_yellow: List[float], 
                        color_red: List[float]) -> None:
    """
    Update object colors based on type and occlusion state.
    
    Provides visual feedback about object state for debugging and human supervision:
    - GREEN: Non-occluded cubes (easy grasp targets)
    - YELLOW: Non-occluded spheres (prefer push)
    - RED: Occluded objects (need clearing first)
    
    Color Priority:
    1. Occlusion status (highest priority) → RED
    2. Object type (cubes vs spheres) → GREEN or YELLOW
    
    This visual encoding helps humans understand what the agent "sees" and
    can aid in debugging learned policies.
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and properties
        color_green: RGBA color for cubes [r, g, b, a] in range [0, 1]
        color_yellow: RGBA color for spheres
        color_red: RGBA color for occluded objects
    
    Returns:
        None (modifies PyBullet visual properties directly)
    
    Implementation Note:
        Uses PyBullet's changeVisualShape() API with link index -1 (base link)
        to update the object's color in the simulation.
    """
    for name, properties in objects.items():
        color_to_set = None
        
        # ====================================================================
        # PRIORITY 1: OCCLUSION (overrides type-based color)
        # ====================================================================
        if properties.get("is_occluded", False):
            color_to_set = color_red
        
        # ====================================================================
        # PRIORITY 2: OBJECT TYPE
        # ====================================================================
        elif properties.get("type") == "cube":
            color_to_set = color_green
        elif properties.get("type") == "sphere":
            color_to_set = color_yellow
        
        # ====================================================================
        # APPLY COLOR CHANGE
        # ====================================================================
        if color_to_set is not None:
            try:
                body_id = sim._bodies_idx.get(name)
                if body_id is not None:
                    # -1 = base link (for simple objects with no joints)
                    sim.physics_client.changeVisualShape(
                        body_id, -1, rgbaColor=color_to_set
                    )
            except Exception as e:
                # Non-critical: visual update failure doesn't break training
                print(f"Warning: Could not change color for {name}: {e}")

def select_target_heuristic(sim, objects: Dict, goal_pos: np.ndarray, 
                           collected_objects: set = None) -> Optional[str]:
    """
    Heuristic target selection: Select the nearest uncollected object to the goal.
    This is NOT learned - it's a simple rule to reduce the action space.
    
    Design Rationale:
    By pre-selecting the target object, we simplify the RL problem from:
    - "Which object to manipulate AND how to manipulate it" (harder)
    To:
    - "How to manipulate the given target object" (easier)
    
    This allows the agent to focus learning on the push-vs-grasp decision
    and action parameterization, rather than also learning object prioritization.
    
    Heuristic Logic:
    1. Filter out already-collected objects
    2. Among remaining objects, choose the one closest to goal
    3. Assumption: Closer objects are easier to place → greedy strategy
    
    Potential Improvements:
    - Could consider occlusion status (prefer non-occluded)
    - Could consider object type (prefer cubes over spheres)
    - Could learn this selection as part of hierarchical RL
    
    Args:
        sim: PyBullet simulation instance
        objects: Dictionary of object names and properties
        goal_pos: Goal position [x, y] in meters
        collected_objects: Set of object names already in goal zone
    
    Returns:
        Name of the selected target object, or None if:
        - All objects are collected
        - No valid objects exist
        - Position queries fail for all objects
    
    Example:
        Objects: {obj0: (0.1, 0.1), obj1: (0.5, 0.5), obj2: (0.2, 0.3)}
        Goal: (-0.2, -0.2)
        Collected: {obj2}
        → Returns "obj0" (closest uncollected to goal)
    """
    if collected_objects is None:
        collected_objects = set()
    
    # ========================================================================
    # FILTER UNCOLLECTED OBJECTS
    # ========================================================================
    uncollected = [name for name in objects.keys() if name not in collected_objects]
    
    if not uncollected:
        # All objects collected or no objects exist
        return None
    
    # ========================================================================
    # FIND OBJECT CLOSEST TO GOAL (GREEDY)
    # ========================================================================
    min_distance = float('inf')
    target = None
    
    for name in uncollected:
        try:
            # Get object position (XY only)
            pos = np.array(sim.get_base_position(name)[:2])
            
            # Compute distance to goal
            distance = np.linalg.norm(pos - goal_pos)
            
            # Track minimum
            if distance < min_distance:
                min_distance = distance
                target = name
        
        except Exception as e:
            # Skip this object if position query fails
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
    
    Uses rejection sampling to find a valid position that satisfies:
    1. Inside workspace bounds
    2. Outside goal zone (prevents trivial task)
    3. Minimum separation from existing objects (prevents collisions)
    
    Rejection Sampling Algorithm:
    1. Sample random position uniformly in workspace
    2. Check constraints (goal overlap, object proximity)
    3. If valid, return position
    4. Else, retry up to max_attempts
    5. Fallback: return random position even if invalid (prevents deadlock)
    
    Why This Matters:
    - Prevents objects from spawning inside goal (would get +5 immediately)
    - Prevents overlapping objects (physics explosions in PyBullet)
    - Ensures reasonable curriculum (not too cluttered initially)
    
    Args:
        sim: PyBullet simulation instance
        existing_objects: List of existing object names to avoid
        goal_pos: Goal position [x, y]
        goal_size: Side length of goal square (meters)
        min_separation: Minimum distance between objects (default: 8cm)
        workspace_bounds: (x_min, x_max, y_min, y_max) in meters
        max_attempts: Maximum rejection sampling iterations
    
    Returns:
        Safe spawn position [x, y, z] where z=0.02 (on table surface)
    
    Example:
        Workspace: [-0.3, 0.3] × [-0.3, 0.3]
        Goal: [-0.2, -0.2] with size 0.1
        Existing objects: [obj0 at (0.1, 0.1)]
        → Samples position, checks:
          - Not in [-0.25, -0.15] × [-0.25, -0.15] (goal)
          - Not within 8cm of (0.1, 0.1) (obj0)
    """
    x_min, x_max, y_min, y_max = workspace_bounds
    half_goal = goal_size / 2
    
    # ========================================================================
    # REJECTION SAMPLING LOOP
    # ========================================================================
    for attempt in range(max_attempts):
        # ====================================================================
        # SAMPLE RANDOM POSITION
        # ====================================================================
        pos_xy = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ], dtype=np.float32)
        
        # ====================================================================
        # CONSTRAINT 1: NOT IN GOAL AREA
        # ====================================================================
        # Check if position is inside goal square
        in_goal = (
            (goal_pos[0] - half_goal < pos_xy[0] < goal_pos[0] + half_goal) and
            (goal_pos[1] - half_goal < pos_xy[1] < goal_pos[1] + half_goal)
        )
        
        if in_goal:
            continue  # Reject and retry
        
        # ====================================================================
        # CONSTRAINT 2: MINIMUM SEPARATION FROM EXISTING OBJECTS
        # ====================================================================
        is_valid = True
        for obj_name in existing_objects:
            try:
                obj_pos = np.array(sim.get_base_position(obj_name)[:2])
                distance = np.linalg.norm(pos_xy - obj_pos)
                
                # Too close to existing object?
                if distance < min_separation:
                    is_valid = False
                    break  # Reject this position
            
            except Exception as e:
                # If position query fails, skip this object
                print(f"Warning: Could not check distance for {obj_name}: {e}")
                continue
        
        # ====================================================================
        # SUCCESS - Found valid position
        # ====================================================================
        if is_valid:
            # Return [x, y, z] with z=0.02 (2cm above table for stable spawn)
            return np.array([pos_xy[0], pos_xy[1], 0.02], dtype=np.float32)
    
    # ========================================================================
    # FALLBACK - Max attempts exhausted
    # ========================================================================
    # This should rarely happen unless workspace is extremely crowded
    # Return a random position to prevent deadlock
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
    
    Simple bounding box check for goal completion detection.
    Used in reward computation to determine if object placement succeeded.
    
    Geometry:
    - Goal is a square with center at goal_pos and side length goal_size
    - Object is considered "in goal" if its center is inside the square
    - Uses axis-aligned bounding box (AABB) check
    
    Edge Cases:
    - Object partially overlapping goal: Not counted (center must be inside)
    - Object on exact boundary: Counted (uses < not <=)
    
    Args:
        object_pos: Object position [x, y, z] or [x, y] in meters
        goal_pos: Goal center position [x, y] in meters
        goal_size: Side length of goal square (meters)
    
    Returns:
        True if object center is inside goal square, False otherwise
    
    Example:
        Goal: center=(-0.2, -0.2), size=0.1
        → Goal bounds: x ∈ [-0.25, -0.15], y ∈ [-0.25, -0.15]
        
        Object at (-0.20, -0.20): True (inside)
        Object at (-0.15, -0.20): True (on boundary)
        Object at (-0.10, -0.20): False (outside)
    
    Implementation Note:
        Only uses XY coordinates (ignores Z) since goal is a 2D region
        on the table surface.
    """
    half_size = goal_size / 2
    obj_xy = object_pos[:2]  # Extract [x, y], ignore z if present
    
    # Axis-aligned bounding box check
    return bool(
        (goal_pos[0] - half_size < obj_xy[0] < goal_pos[0] + half_size) and
        (goal_pos[1] - half_size < obj_xy[1] < goal_pos[1] + half_size)
    )
