"""
Physics and simulation utilities for PyBullet environment setup.
Handles collision detection, workspace boundaries, and scene management.

This module provides all physics-related query functions for the RL environment:
- Workspace violation detection (objects falling off table)
- Collision detection (robot-object, object-object, self-collision)
- Contact force measurement (for grasp success detection)
- Visualization utilities (workspace boundaries)

These functions abstract away PyBullet's C-style API into cleaner Python interfaces
with proper error handling and type hints.

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
    
    This is used for the -10 penalty in the reward function (Component 5).
    Detects two types of violations:
    1. Object fell off the table (z < z_min)
    2. Object moved outside horizontal workspace bounds
    
    Why This Matters:
    - Prevents robot from "solving" task by pushing objects off table
    - Ensures learning focuses on manipulation, not exploitation
    - Harsh penalty encourages careful, precise movements
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of body to check (e.g., "object_0")
        workspace_bounds: (x_min, x_max, y_min, y_max) in meters
        z_min: Minimum allowed z-coordinate (default: -5cm below table)
               Objects below this height are considered "fallen"
    
    Returns:
        violated: True if object is outside workspace or fell off table
                 False if object is within safe boundaries
    
    Example:
        workspace_bounds = (-0.3, 0.3, -0.3, 0.3)
        z_min = -0.05
        
        Object at (0.2, 0.1, 0.02): False (within bounds)
        Object at (0.4, 0.1, 0.02): True (x > x_max)
        Object at (0.1, 0.1, -0.06): True (fell below table)
    
    Error Handling:
        If body_name doesn't exist or position query fails, returns True
        (conservative: assume violation to prevent training on ghost objects)
    """
    try:
        pos = sim.get_base_position(body_name)
    except Exception as e:
        print(f"Warning: Could not get position for {body_name}: {e}")
        # Conservative: treat non-existent bodies as violations
        # This prevents training from continuing with invalid state
        return True
    
    x_min, x_max, y_min, y_max = workspace_bounds
    
    # ========================================================================
    # CHECK HORIZONTAL BOUNDARIES (XY plane)
    # ========================================================================
    if pos[0] < x_min or pos[0] > x_max:
        return True  # Violated X bounds
    if pos[1] < y_min or pos[1] > y_max:
        return True  # Violated Y bounds
    
    # ========================================================================
    # CHECK VERTICAL BOUNDARY (Z axis)
    # ========================================================================
    # Objects should rest at z ≈ 0.02 (2cm above table)
    # If z < -0.05, object has fallen 7cm below table → clearly fallen
    if pos[2] < z_min:
        return True  # Fell off table
    
    return False  # All checks passed


def check_collision_with_table(sim, body_name: str, table_name: str = "table") -> bool:
    """
    Check if object is colliding with the table.
    
    Used for the -5 collision penalty in the reward function (Component 6).
    Detects when the robot arm inappropriately contacts the table surface.
    
    Why This Matters:
    - Prevents robot from "slamming" down on table during manipulation
    - Encourages smooth, careful approach trajectories
    - In real robots, table collisions can damage hardware
    
    Implementation Note:
        Uses PyBullet's getContactPoints() API which queries the physics engine's
        narrow-phase collision detection. Contact points are generated during
        simulation steps when bodies interpenetrate.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of body to check (typically "panda" for robot)
        table_name: Name of table body (default: "table")
    
    Returns:
        has_collision: True if any contact points exist between bodies
                      False if no contact or if bodies don't exist
    
    Example Usage:
        if check_collision_with_table(sim, 'panda', 'table'):
            print("Robot hit the table!")
            reward -= 5.0
    
    PyBullet Contact Point Structure:
        Each contact point contains:
        - Contact flag, body IDs, link IDs
        - Contact position, normal, distance
        - Normal force, friction forces
        (See get_contact_details() for full structure)
    """
    try:
        # Get PyBullet body IDs from names
        body_id = sim._bodies_idx.get(body_name)
        table_id = sim._bodies_idx.get(table_name)
        
        # Handle case where body doesn't exist
        if body_id is None or table_id is None:
            return False  # No collision if body doesn't exist
        
        # Query physics engine for contact points
        contact_points = sim.physics_client.getContactPoints(
            bodyA=body_id,
            bodyB=table_id
        )
        
        # Any contact point = collision detected
        return len(contact_points) > 0
        
    except Exception as e:
        print(f"Warning: Error checking table collision: {e}")
        return False  # Fail-safe: assume no collision on error

def check_object_collision(sim, body_name1: str, body_name2: str) -> bool:
    """
    Check if two objects are colliding.
    
    Used for the -5 collision penalty when robot collides with non-target objects.
    
    Why This Matters:
    - Robot should only contact the target object during manipulation
    - Colliding with other objects can cause unintended displacement
    - In cluttered scenes, avoiding collisions requires spatial reasoning
    
    Use Case in Environment:
        During manipulation, check if robot collides with objects OTHER than
        the current target. Target collisions are allowed (necessary for grasp/push),
        but hitting bystander objects should be penalized.
    
    Args:
        sim: PyBullet simulation instance
        body_name1: First object name (e.g., "panda")
        body_name2: Second object name (e.g., "object_3")
    
    Returns:
        has_collision: True if objects are in contact
                      False if no contact or bodies don't exist
    
    Example Usage:
        for obj_name in objects:
            if obj_name != current_target:
                if check_object_collision(sim, 'panda', obj_name):
                    collision_penalty = -5.0
    
    Implementation Note:
        Symmetric function: check_object_collision(A, B) == check_object_collision(B, A)
    """
    try:
        body_id1 = sim._bodies_idx.get(body_name1)
        body_id2 = sim._bodies_idx.get(body_name2)
        
        if body_id1 is None or body_id2 is None:
            return False
        
        # Query contact points between the two bodies
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
    
    More granular than check_object_collision() - allows checking individual links
    rather than the entire robot body. Useful for:
    - Gripper contact detection (check fingertip links)
    - Elbow collision avoidance (check arm links)
    - Debugging collision sources in complex scenes
    
    Panda Robot Link Structure (for reference):
        Link 0-6: Arm joints (shoulder to wrist)
        Link 7-8: Hand base
        Link 9-10: Gripper fingers (left, right)
        Link 11: End-effector frame (no collision geometry)
    
    Args:
        sim: PyBullet simulation instance
        robot_body_name: Name of robot body (e.g., "panda")
        link_id: Specific link index to check (0-based)
        other_body_name: Name of other object to check against
    
    Returns:
        has_collision: True if specified link is in contact with other object
                      False otherwise
    
    Example Usage:
        # Check if gripper finger (link 9) is touching target object
        if check_robot_link_collision(sim, 'panda', 9, 'object_0'):
            print("Gripper finger touching object!")
    
    Use Case:
        Could be used to verify successful grasp:
        - Both finger links (9, 10) should contact object
        - If only one finger contacts, grasp may be unstable
    """
    try:
        robot_id = sim._bodies_idx.get(robot_body_name)
        other_id = sim._bodies_idx.get(other_body_name)
        
        if robot_id is None or other_id is None:
            return False
        
        # Query contacts for SPECIFIC link only
        contact_points = sim.physics_client.getContactPoints(
            bodyA=robot_id,
            bodyB=other_id,
            linkIndexA=link_id  # Filter by specific link
        )
        
        return len(contact_points) > 0
        
    except Exception as e:
        print(f"Warning: Error checking robot link collision: {e}")
        return False


def check_self_collision(sim, body_name: str) -> bool:
    """
    Check if robot has self-collision between its links.
    
    Self-collision occurs when different parts of the robot collide with each other,
    such as:
    - Forearm hitting shoulder
    - Gripper hitting wrist
    - Elbow hitting torso
    
    Why This Matters:
    - Self-collisions indicate kinematically infeasible configurations
    - Can cause physics instability (jittering, exploding joints)
    - In real robots, causes mechanical damage
    - Should ideally never occur if motion planner works correctly
    
    When to Use:
    - Sanity check after motion planning
    - Additional reward penalty if self-collisions occur
    - Debugging motion primitive failures
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of robot body (e.g., "panda")
    
    Returns:
        has_self_collision: True if any links are colliding with each other
                           False if robot is in valid configuration
    
    Example Usage:
        if check_self_collision(sim, 'panda'):
            print("Warning: Robot self-collision detected!")
            # Could add penalty: reward -= 2.0
    
    Implementation Note:
        Checks if bodyA == bodyB in any contact point, indicating
        that the robot is contacting itself (different links).
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            return False
        
        # Get ALL contact points involving this body
        contact_points = sim.physics_client.getContactPoints(bodyA=body_id)
        
        # Check if any contacts are with itself (self-collision)
        for contact in contact_points:
            body_b_id = contact[2]  # Second body in contact
            
            if body_b_id == body_id:
                # Body is colliding with itself!
                return True
        
        return False  # No self-collisions found
        
    except Exception as e:
        print(f"Warning: Error checking self-collision: {e}")
        return False


def get_contact_force(sim, body_name1: str, body_name2: str) -> float:
    """
    Get magnitude of contact force between two bodies.
    
    Measures the total normal force at all contact points between bodies.
    Useful for:
    - Grasp success detection (high force = secure grasp)
    - Push force monitoring (did push apply sufficient force?)
    - Failure diagnosis (zero force = no contact despite command)
    
    Force Interpretation:
    - 0 N: No contact
    - 1-5 N: Light contact (gentle touch, may slip)
    - 5-20 N: Moderate contact (stable grasp, effective push)
    - >20 N: High force (crushing grip, risk of object damage)
    
    Args:
        sim: PyBullet simulation instance
        body_name1: First object name (e.g., "panda")
        body_name2: Second object name (e.g., "object_0")
    
    Returns:
        force_magnitude: Total contact force magnitude in Newtons (N)
                        Sum of normal forces at all contact points
    
    Example Usage:
        # Check grasp strength
        grasp_force = get_contact_force(sim, 'panda', 'object_0')
        if grasp_force > 5.0:
            print(f"Strong grasp: {grasp_force:.1f}N")
        else:
            print("Weak grasp, object may slip!")
    
    PyBullet Contact Force:
        contact[9] = normal force (scalar, always positive)
        Total force = sum of all contact point forces
        
    Limitation:
        Only returns normal force magnitude, not direction or friction forces.
        See get_contact_details() for full force vector information.
    """
    try:
        body_id1 = sim._bodies_idx.get(body_name1)
        body_id2 = sim._bodies_idx.get(body_name2)
        
        if body_id1 is None or body_id2 is None:
            return 0.0  # No force if bodies don't exist
        
        # Query all contact points
        contact_points = sim.physics_client.getContactPoints(
            bodyA=body_id1,
            bodyB=body_id2
        )
        
        # Sum normal forces from all contact points
        total_force = 0.0
        for contact in contact_points:
            # contact[9] is the normal force magnitude (N)
            total_force += abs(contact[9])
        
        return float(total_force)
        
    except Exception as e:
        print(f"Warning: Error getting contact force: {e}")
        return 0.0


def get_contact_details(sim, body_name1: str, body_name2: str) -> List[dict]:
    """
    Get detailed contact point information between two bodies.
    
    Provides comprehensive contact information for advanced manipulation reasoning:
    - Contact positions (where bodies touch)
    - Contact normals (surface orientation)
    - Contact forces (normal + friction)
    - Contact distance (penetration depth)
    
    Use Cases:
    - Grasp quality assessment (contact normal alignment)
    - Slip detection (lateral friction analysis)
    - Contact-rich manipulation (force control)
    - Debugging physics behavior
    
    Args:
        sim: PyBullet simulation instance
        body_name1: First object name
        body_name2: Second object name
    
    Returns:
        List of contact dictionaries, one per contact point.
        Each dictionary contains:
        
        ┌────────────────────────────────────────────────────────────┐
        │ Field                  │ Type        │ Description          │
        ├────────────────────────┼─────────────┼──────────────────────┤
        │ contact_flag           │ int         │ Reserved (unused)    │
        │ body_a_id              │ int         │ First body ID        │
        │ body_b_id              │ int         │ Second body ID       │
        │ link_a_id              │ int         │ Link on body A       │
        │ link_b_id              │ int         │ Link on body B       │
        │ position_on_a          │ np.ndarray  │ Contact point on A   │
        │ position_on_b          │ np.ndarray  │ Contact point on B   │
        │ contact_normal         │ np.ndarray  │ Normal vector (→B)   │
        │ contact_distance       │ float       │ Separation (<0=pen.) │
        │ normal_force           │ float       │ Normal force (N)     │
        │ lateral_friction_1     │ float       │ Friction force 1     │
        │ lateral_friction_dir_1 │ np.ndarray  │ Friction direction 1 │
        │ lateral_friction_2     │ float       │ Friction force 2     │
        │ lateral_friction_dir_2 │ np.ndarray  │ Friction direction 2 │
        └────────────────────────────────────────────────────────────┘
    
    Example Usage:
        contacts = get_contact_details(sim, 'panda', 'object_0')
        for c in contacts:
            print(f"Contact at {c['position_on_a']}")
            print(f"Normal force: {c['normal_force']:.2f}N")
            print(f"Contact normal: {c['contact_normal']}")
    
    Advanced Application:
        # Check if grasp is aligned (contact normals oppose gripper motion)
        for c in contacts:
            if np.dot(c['contact_normal'], gripper_approach_dir) < -0.8:
                print("Good contact alignment!")
    
    PyBullet Contact Point Indices:
        cp[0]: contact flag
        cp[1]: bodyUniqueIdA
        cp[2]: bodyUniqueIdB
        cp[3]: linkIndexA
        cp[4]: linkIndexB
        cp[5]: positionOnA [x,y,z]
        cp[6]: positionOnB [x,y,z]
        cp[7]: contactNormalOnB [x,y,z]
        cp[8]: contactDistance (m)
        cp[9]: normalForce (N)
        cp[10]: lateralFriction1
        cp[11]: lateralFrictionDir1 [x,y,z]
        cp[12]: lateralFriction2
        cp[13]: lateralFrictionDir2 [x,y,z]
    """
    try:
        body_id1 = sim._bodies_idx.get(body_name1)
        body_id2 = sim._bodies_idx.get(body_name2)
        
        if body_id1 is None or body_id2 is None:
            return []  # No contacts if bodies don't exist
        
        # Query contact points from physics engine
        contact_points = sim.physics_client.getContactPoints(
            bodyA=body_id1,
            bodyB=body_id2
        )
        
        # Parse PyBullet's C-style tuples into clean Python dicts
        contacts = []
        for cp in contact_points:
            contact_info = {
                'contact_flag': cp[0],
                'body_a_id': cp[1],
                'body_b_id': cp[2],
                'link_a_id': cp[3],
                'link_b_id': cp[4],
                'position_on_a': np.array(cp[5]),        # [x, y, z]
                'position_on_b': np.array(cp[6]),        # [x, y, z]
                'contact_normal': np.array(cp[7]),       # Unit vector from A to B
                'contact_distance': cp[8],               # <0 = penetration
                'normal_force': cp[9],                   # Newtons
                'lateral_friction_1': cp[10],            # Friction force 1
                'lateral_friction_dir_1': np.array(cp[11]),  # Direction 1
                'lateral_friction_2': cp[12],            # Friction force 2
                'lateral_friction_dir_2': np.array(cp[13]),  # Direction 2
            }
            contacts.append(contact_info)
        
        return contacts
        
    except Exception as e:
        print(f"Warning: Error getting contact details: {e}")
        return []


def get_all_collisions(sim, body_name: str) -> List[Tuple[str, float]]:
    """
    Get all objects currently colliding with the specified body.
    
    Useful for comprehensive collision analysis:
    - "What is the robot touching right now?"
    - Aggregate collision penalty (sum forces from all contacts)
    - Debugging unexpected collisions
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of body to check (e.g., "panda")
    
    Returns:
        List of (colliding_body_name, total_contact_force) tuples
        Sorted by force magnitude (highest first, if needed)
    
    Example Output:
        [
            ('object_0', 15.2),  # Robot touching object_0 with 15.2N
            ('table', 3.5),      # Light table contact with 3.5N
            ('object_3', 1.1)    # Barely touching object_3
        ]
    
    Example Usage:
        collisions = get_all_collisions(sim, 'panda')
        if len(collisions) > 2:
            print(f"Warning: Robot colliding with {len(collisions)} objects!")
        
        for obj_name, force in collisions:
            print(f"  - {obj_name}: {force:.1f}N")
    
    Implementation Note:
        Groups contact points by body B and sums forces.
        Requires reverse lookup from body ID to name (iterates _bodies_idx).
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            return []
        
        collisions = []
        # Get ALL contacts involving this body
        contact_points = sim.physics_client.getContactPoints(bodyA=body_id)
        
        # ====================================================================
        # GROUP CONTACTS BY BODY B AND SUM FORCES
        # ====================================================================
        body_forces = {}  # {body_name: total_force}
        
        for contact in contact_points:
            body_b_id = contact[2]  # Second body in contact
            force = abs(contact[9])  # Normal force magnitude
            
            # ================================================================
            # REVERSE LOOKUP: Find body name from ID
            # ================================================================
            body_b_name = None
            for name, bid in sim._bodies_idx.items():
                if bid == body_b_id:
                    body_b_name = name
                    break
            
            # Accumulate force for this body (if not self-collision)
            if body_b_name and body_b_name != body_name:
                if body_b_name in body_forces:
                    body_forces[body_b_name] += force
                else:
                    body_forces[body_b_name] = force
        
        # Convert to list of tuples
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
    
    Adds debug lines showing the valid manipulation region.
    Useful for:
    - Visual debugging (see where workspace ends)
    - Human supervision (understand task constraints)
    - Video presentations (clearly show boundaries)
    
    Args:
        client: PyBullet physics client (use sim.physics_client)
        workspace_bounds: (x_min, x_max, y_min, y_max) in meters
        z_height: Height at which to draw boundary (default: table surface)
        color: RGB color [r, g, b] in range [0, 1] (default: red)
        line_width: Width of boundary lines in pixels
    
    Returns:
        None (adds debug lines to simulation)
    
    Example Usage:
        # Draw red workspace boundary at table height
        draw_workspace_boundary(
            sim.physics_client,
            workspace_bounds=(-0.3, 0.3, -0.3, 0.3),
            z_height=0.0,
            color=[1, 0, 0],  # Red
            line_width=3.0
        )
    
    Visual Result:
        Creates a red rectangle on the table showing where objects
        are allowed. Objects outside this rectangle trigger -10 penalty.
    
    Implementation Note:
        Uses addUserDebugLine() which persists until simulation reset.
        Lines are purely visual - they don't affect physics.
    """
    if color is None:
        color = [1, 0, 0]  # Default: Red (warning color)
    
    try:
        x_min, x_max, y_min, y_max = workspace_bounds
        
        # Define 4 corners of rectangular boundary
        corners = [
            [x_min, y_min, z_height],  # Bottom-left
            [x_max, y_min, z_height],  # Bottom-right
            [x_max, y_max, z_height],  # Top-right
            [x_min, y_max, z_height]   # Top-left
        ]
        
        # Draw 4 edges of rectangle
        for i in range(4):
            client.addUserDebugLine(
                corners[i],
                corners[(i + 1) % 4],  # Connect to next corner (wrap around)
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
    
    Essential for determining when physics simulation has "settled" after
    manipulation actions. Used to:
    - Wait for objects to stop moving before taking next action
    - Verify successful placement (object at rest in goal)
    - Detect if push was too weak (object didn't move)
    
    Stability Criteria:
    An object is "stable" when BOTH conditions are met:
    1. Linear velocity < 1 cm/s (barely moving)
    2. Angular velocity < 0.05 rad/s (~3 degrees/s, barely rotating)
    
    Why These Thresholds?
    - Too strict (e.g., 0.001): Object never stabilizes due to numerical noise
    - Too loose (e.g., 0.1): Consider still-moving objects as "stable"
    - Current values: Practical balance for tabletop manipulation
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object to check
        velocity_threshold: Linear velocity threshold (m/s) for "stable"
                           Default: 0.01 m/s = 1 cm/s
        angular_velocity_threshold: Angular velocity threshold (rad/s)
                                   Default: 0.05 rad/s ≈ 2.86 deg/s
    
    Returns:
        is_stable: True if object is not moving significantly
                  False if still in motion or if error occurs
    
    Example Usage:
        # After placing object, wait for it to settle
        while not is_object_stable(sim, 'object_0'):
            sim.step()
            time.sleep(0.01)
        print("Object has settled!")
    
    Use in Action Primitives:
        execute_pick_and_place() should wait for object stability
        before considering placement successful.
    """
    try:
        # Get current velocities from physics engine
        lin_vel = sim.get_base_velocity(body_name)
        ang_vel = sim.get_base_angular_velocity(body_name)
        
        # Compute velocity magnitudes (scalars)
        lin_speed = np.linalg.norm(lin_vel)    # √(vx² + vy² + vz²)
        ang_speed = np.linalg.norm(ang_vel)    # √(ωx² + ωy² + ωz²)
        
        # Check both conditions
        return lin_speed < velocity_threshold and ang_speed < angular_velocity_threshold
      
    except Exception as e:
        print(f"Warning: Could not check stability for {body_name}: {e}")
        # Conservative: assume unstable on error (prevents false "success")
        return False


def wait_for_objects_stable(sim, object_names: List[str], 
                           max_steps: int = 100,
                           check_interval: int = 10) -> bool:
    """
    Wait until all objects in scene have stabilized.
    
    Runs physics simulation forward in time until all specified objects
    stop moving, or until timeout is reached. Critical for:
    - Scene initialization (wait for spawned objects to settle)
    - After manipulation (ensure object placement is stable)
    - Before reward computation (avoid transient states)
    
    Algorithm:
    1. Run simulation steps (advance physics)
    2. Every check_interval steps, test all objects for stability
    3. If all stable: return True immediately (early exit)
    4. If timeout (max_steps): return False (some objects still moving)
    
    Timeout Interpretation:
    - False return means scene didn't stabilize (possible issues):
      * Objects rolling off table
      * Unstable stacking
      * Continuous collision/jittering
      * Physics explosion (rare but possible)
    
    Args:
        sim: PyBullet simulation instance
        object_names: List of object names to monitor
        max_steps: Maximum simulation steps to wait (default: 100)
                  At 240 Hz (default), 100 steps ≈ 0.42 seconds
        check_interval: Check stability every N steps (default: 10)
                       Checking every step wastes computation
    
    Returns:
        all_stable: True if all objects stabilized within max_steps
                   False if timeout or no objects to check
    
    Example Usage:
        # After spawning objects, wait for scene to settle
        object_names = ['object_0', 'object_1', 'object_2']
        if wait_for_objects_stable(sim, object_names, max_steps=200):
            print("Scene ready!")
        else:
            print("Warning: Scene didn't stabilize!")
    
    Performance Note:
        With check_interval=10, only checks 10 times over 100 steps.
        This is ~10× faster than checking every step.
    """
    # Handle edge case: no objects to check
    if not object_names:
        return True  # Trivially stable
    
    # Run simulation and check periodically
    for step in range(max_steps):
        sim.step()  # Advance physics by one timestep
        
        # Only check stability every check_interval steps
        if step % check_interval == 0:
            all_stable = True
            
            # Check each object
            for name in object_names:
                if not is_object_stable(sim, name):
                    all_stable = False
                    break  # Early exit if any object is moving
            
            # Success: all objects stopped moving
            if all_stable:
                return True
    
    # Timeout - not all objects stabilized within max_steps
    return False


def get_object_bounding_box(sim, body_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get axis-aligned bounding box (AABB) for object.
    
    Bounding boxes are useful for:
    - Collision avoidance (check if boxes overlap)
    - Grasp planning (determine object size)
    - Volume estimation (bounding box volume ≈ object volume)
    - Visualization (draw boxes around objects)
    
    AABB Definition:
    An axis-aligned bounding box is the smallest box (aligned with world axes)
    that completely contains the object. Defined by two corner points:
    - min_coords: [x_min, y_min, z_min] - bottom-left-front corner
    - max_coords: [x_max, y_max, z_max] - top-right-back corner
    
    Properties:
    - Fast to compute and test
    - Conservative (often larger than object)
    - Rotates with object (recomputed each query)
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object
    
    Returns:
        (min_coords, max_coords): Tuple of 3D corner coordinates,
                                 or None if body not found
        min_coords: np.ndarray [x_min, y_min, z_min]
        max_coords: np.ndarray [x_max, y_max, z_max]
    
    Example Usage:
        bbox = get_object_bounding_box(sim, 'object_0')
        if bbox:
            min_coords, max_coords = bbox
            size = max_coords - min_coords
            print(f"Object dimensions: {size}")
            print(f"Center: {(min_coords + max_coords) / 2}")
    
    Use Case - Collision Prediction:
        # Check if two objects' bounding boxes overlap
        bbox1 = get_object_bounding_box(sim, 'object_0')
        bbox2 = get_object_bounding_box(sim, 'object_1')
        overlap = (bbox1[0] < bbox2[1]).all() and (bbox2[0] < bbox1[1]).all()
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            return None
        
        # Query PyBullet for AABB
        # Returns: ((x_min, y_min, z_min), (x_max, y_max, z_max))
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
    
    Quick approximation of object volume using AABB. Useful for:
    - Physics reasoning (volume ∝ mass for uniform density)
    - Size-based manipulation strategies (large objects harder to grasp)
    - Workspace planning (packing density estimation)
    
    Accuracy Note:
    - Exact for boxes (AABB = actual volume)
    - Overestimate for spheres (AABB volume = 2³ × sphere volume)
    - Overestimate for irregular shapes (depends on concavity)
    - For cubes: ~100% accurate
    - For spheres: ~52% accurate (π/6 ≈ 0.52)
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object
    
    Returns:
        volume: Approximate volume in m³
               Returns 0.0 if bounding box unavailable
    
    Example Usage:
        vol = compute_object_volume(sim, 'object_0')
        if vol > 0.001:  # 1000 cm³
            print("Large object - use two-handed grasp")
        else:
            print("Small object - single grasp OK")
    
    Comparison with True Volume:
        For 4cm cube: AABB volume = 6.4e-5 m³ (exact)
        For 4cm sphere: AABB volume = 6.4e-5 m³ (true: 3.35e-5 m³)
    """
    bbox = get_object_bounding_box(sim, body_name)
    if bbox is None:
        return 0.0
    
    min_coords, max_coords = bbox
    dimensions = max_coords - min_coords  # [width, depth, height]
    volume = np.prod(dimensions)          # width × depth × height
    
    return float(volume)


def set_object_color(sim, body_name: str, rgba_color: List[float]):
    """
    Change visual color of an object.
    
    Updates the object's visual appearance without affecting physics.
    Used for:
    - State visualization (occlusion status, target selection)
    - Debugging (highlight problematic objects)
    - User feedback (show success/failure)
    
    Note: This is a simplified wrapper around update_object_colors()
    in object_util.py. Use that function for batch color updates.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object to recolor
        rgba_color: RGBA color [r, g, b, a] in range [0, 1]
                   Example: [1, 0, 0, 1] = opaque red
    
    Returns:
        None (modifies visual properties in simulation)
    
    Example Usage:
        # Highlight target object in blue
        set_object_color(sim, 'object_0', [0, 0, 1, 1])
        
        # Make object semi-transparent
        set_object_color(sim, 'object_1', [0.5, 0.5, 0.5, 0.3])
    
    Limitation:
        Only changes base link color (link index -1).
        For multi-link objects, would need to iterate over all links.
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is not None:
            sim.physics_client.changeVisualShape(
                body_id,
                -1,  # Link index (-1 = base link for simple objects)
                rgbaColor=rgba_color
            )
    except Exception as e:
        print(f"Warning: Could not change color for {body_name}: {e}")


def apply_force_to_object(sim, body_name: str, force: np.ndarray,
                         position: Optional[np.ndarray] = None):
    """
    Apply external force to object (e.g., for push simulation).
    
    Directly applies force in world frame, bypassing robot kinematics.
    Useful for:
    - Testing push dynamics (how much force moves object?)
    - Simulating disturbances (wind, bumps)
    - Debugging physics behavior
    - Prototyping without full robot control
    
    Physics Note:
        Force is applied for ONE simulation step only.
        For sustained force, call this function every step.
        Impulse = Force × dt (where dt ≈ 1/240 s by default)
    
    Warning:
        In actual RL training, don't use this function!
        The agent should learn to push via robot contact, not magic forces.
        This is for testing/debugging only.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object to push
        force: Force vector [fx, fy, fz] in Newtons
               Example: [10, 0, 0] = 10N push in +X direction
        position: Position to apply force (world frame coordinates)
                 None = apply at center of mass (default)
                 Off-center forces cause rotation (torque)
    
    Returns:
        None (modifies object velocity via physics engine)
    
    Example Usage:
        # Push object with 5N force in +X direction
        apply_force_to_object(sim, 'object_0', np.array([5, 0, 0]))
        sim.step()  # Advance physics to see effect
        
        # Apply off-center force (causes rotation)
        obj_pos = sim.get_base_position('object_0')
        force_pos = obj_pos + [0.02, 0, 0]  # 2cm offset
        apply_force_to_object(sim, 'object_0', [5, 0, 0], force_pos)
    
    Force Magnitude Guide:
        - 1-5 N: Light push (gentle nudge)
        - 5-20 N: Moderate push (typical robot push)
        - 20+ N: Strong push (risk of object flying away)
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            print(f"Warning: Body {body_name} not found")
            return
        
        # Default: apply force at center of mass
        if position is None:
            position = sim.get_base_position(body_name)
        
        # Convert numpy arrays to lists (PyBullet API requirement)
        if isinstance(force, np.ndarray):
            force = force.tolist()
        if isinstance(position, np.ndarray):
            position = position.tolist()
        
        # Apply force in world frame
        sim.physics_client.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,              # Base link
            forceObj=force,            # Force vector [Fx, Fy, Fz]
            posObj=position,           # Application point [x, y, z]
            flags=p.WORLD_FRAME        # Coordinates in world frame
        )
        
    except Exception as e:
        print(f"Warning: Could not apply force to {body_name}: {e}")



def reset_object_pose(sim, body_name: str, position: np.ndarray,
                     orientation: Optional[np.ndarray] = None):
    """
    Reset object to specific pose (teleport, no physics).
    
    Instantly moves object to target pose, bypassing physics simulation.
    Also resets velocities to zero (object starts at rest).
    
    Use Cases:
    - Resetting fallen objects back to workspace
    - Setting up specific test scenarios
    - Correcting physics glitches
    - Implementing "teleport to goal" for debugging
    
    Warning:
        This is a non-physical operation! Object instantly appears at new pose.
        Can cause collision explosions if teleported inside another object.
        Always verify target pose is collision-free before calling.
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object to reset
        position: Target position [x, y, z] in meters (world frame)
        orientation: Target quaternion [x, y, z, w] (world frame)
                    None = no rotation (identity quaternion [0, 0, 0, 1])
    
    Returns:
        None (modifies object state in simulation)
    
    Example Usage:
        # Reset object to table center, upright
        reset_object_pose(
            sim, 'object_0',
            position=np.array([0, 0, 0.02]),
            orientation=np.array([0, 0, 0, 1])  # No rotation
        )
        
        # Reset to goal zone
        reset_object_pose(sim, 'object_0', goal_pos)
    
    Side Effects:
        - Position and orientation set to target values
        - Linear velocity set to [0, 0, 0]
        - Angular velocity set to [0, 0, 0]
        - Contact forces cleared (no residual collisions)
    """
    try:
        body_id = sim._bodies_idx.get(body_name)
        if body_id is None:
            print(f"Warning: Body {body_name} not found")
            return
        
        # Default orientation: no rotation (identity quaternion)
        if orientation is None:
            orientation = [0, 0, 0, 1]  # [qx, qy, qz, qw]
        
        # Convert numpy arrays to lists (PyBullet API requirement)
        if isinstance(position, np.ndarray):
            position = position.tolist()
        if isinstance(orientation, np.ndarray):
            orientation = orientation.tolist()
        
        # Teleport object to target pose
        sim.physics_client.resetBasePositionAndOrientation(
            body_id,
            position,
            orientation
        )
        
        # Reset velocities to zero (object starts at rest)
        # Important: prevents residual momentum from causing immediate movement
        sim.physics_client.resetBaseVelocity(
            body_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )
        
    except Exception as e:
        print(f"Warning: Could not reset pose for {body_name}: {e}")


def set_gravity(sim, gravity: float = -9.81):
    """
    Set gravity in simulation.
    
    Changes the gravitational acceleration applied to all objects.
    
    Use Cases:
    - Standard Earth gravity: -9.81 m/s² (default)
    - Zero gravity testing: 0.0 (floating objects)
    - Moon gravity: -1.62 m/s² (slower falling)
    - Increased gravity: -20.0 (faster dynamics)
    
    Why Change Gravity?
    - Debugging: Zero gravity simplifies testing (no falling)
    - Sim-to-real: Match real robot's environment
    - Curriculum learning: Start with low gravity (easier), increase gradually
    
    Args:
        sim: PyBullet simulation instance
        gravity: Gravity acceleration in m/s² (default: -9.81)
                Negative = downward (-Z direction)
                Positive = upward (floating objects)
    
    Returns:
        None (modifies global physics parameter)
    
    Example Usage:
        # Disable gravity for testing
        set_gravity(sim, 0.0)
        
        # Restore Earth gravity
        set_gravity(sim, -9.81)
        
        # Lunar simulation
        set_gravity(sim, -1.62)
    
    Note:
        Gravity is a global parameter - affects ALL objects in simulation.
        Cannot set per-object gravity in PyBullet.
    """
    try:
        # Set gravity vector: [gx, gy, gz]
        # Standard: [0, 0, -9.81] = downward in Z
        sim.physics_client.setGravity(0, 0, gravity)
    except Exception as e:
        print(f"Warning: Could not set gravity: {e}")


def check_multiple_workspace_violations(sim, object_names: List[str],
                                       workspace_bounds: Tuple[float, float, float, float],
                                       z_min: float = -0.05) -> List[str]:
    """
    Check multiple objects for workspace violations.
    
    Batch version of check_workspace_violation() for efficiency.
    Returns list of ALL objects that violated bounds (not just first one).
    
    Use Cases:
    - End-of-episode cleanup (remove all fallen objects)
    - Batch penalty computation (sum violations)
    - Scene validation (check all objects are valid)
    
    Args:
        sim: PyBullet simulation instance
        object_names: List of object names to check
        workspace_bounds: (x_min, x_max, y_min, y_max) in meters
        z_min: Minimum allowed z-coordinate (default: -5cm below table)
    
    Returns:
        List of object names that violated workspace bounds
        Empty list if all objects are within bounds
    
    Example Usage:
        objects = ['object_0', 'object_1', 'object_2']
        violated = check_multiple_workspace_violations(
            sim, objects, WORKSPACE_BOUNDS, z_min=-0.05
        )
        
        if violated:
            print(f"Objects fell: {violated}")
            for obj in violated:
                # Remove from scene
                del self.objects[obj]
    
    Performance:
        O(N) complexity where N = len(object_names)
        More efficient than checking individually in separate calls
        due to reduced Python/C++ API overhead.
    """
    violated_objects = []
    
    for name in object_names:
        if check_workspace_violation(sim, name, workspace_bounds, z_min):
            violated_objects.append(name)
    
    return violated_objects


def get_object_velocity_magnitude(sim, body_name: str) -> Tuple[float, float]:
    """
    Get linear and angular velocity magnitudes for an object.
    
    Convenience function that returns velocity scalars (speeds) instead of vectors.
    Useful for:
    - Stability checking (is object moving?)
    - Kinetic energy estimation (KE ∝ v²)
    - Debugging motion behavior
    
    Args:
        sim: PyBullet simulation instance
        body_name: Name of object
    
    Returns:
        (linear_speed, angular_speed): Speed magnitudes
        - linear_speed: m/s (0 = stationary, >0.1 = moving fast)
        - angular_speed: rad/s (0 = not rotating, >1 = spinning fast)
        Returns (0.0, 0.0) on error
    
    Example Usage:
        lin_speed, ang_speed = get_object_velocity_magnitude(sim, 'object_0')
        print(f"Object moving at {lin_speed*100:.1f} cm/s")
        print(f"Object rotating at {np.rad2deg(ang_speed):.1f} deg/s")
        
        if lin_speed > 0.05:
            print("Object is moving significantly!")
    
    Comparison with get_base_velocity():
        get_base_velocity() → [vx, vy, vz] (vector)
        This function → ||v|| = √(vx² + vy² + vz²) (scalar)
    """
    try:
        # Get velocity vectors from physics engine
        lin_vel = sim.get_base_velocity(body_name)
        ang_vel = sim.get_base_angular_velocity(body_name)
        
        # Compute magnitudes (Euclidean norms)
        lin_speed = float(np.linalg.norm(lin_vel))  # m/s
        ang_speed = float(np.linalg.norm(ang_vel))  # rad/s
        
        return lin_speed, ang_speed
        
    except Exception as e:
        print(f"Warning: Could not get velocity for {body_name}: {e}")
        return 0.0, 0.0  # Assume stationary on error


def check_line_of_sight(sim, point_a: np.ndarray, point_b: np.ndarray,
                       ignore_bodies: List[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check if there's a clear line of sight between two points.
    
    Uses ray-casting to detect if any object blocks the straight-line path
    between two points. Essential for:
    - Visibility checking (can camera see object?)
    - Grasp planning (is approach path clear?)
    - Occlusion detection (is object blocked by another?)
    - Collision prediction (will motion collide?)
    
    Algorithm:
    1. Cast ray from point_a to point_b
    2. If ray hits nothing: path is clear
    3. If ray hits object: path is blocked (return blocking object name)
    4. Optionally ignore specified bodies (e.g., ignore robot when checking object visibility)
    
    Args:
        sim: PyBullet simulation instance
        point_a: Start point [x, y, z] in world frame
        point_b: End point [x, y, z] in world frame
        ignore_bodies: List of body names to ignore in ray test
                      Example: ['panda', 'table'] to only check object occlusion
    
    Returns:
        (is_clear, blocking_body): Tuple of:
        - is_clear: True if no obstruction, False if blocked
        - blocking_body: Name of blocking object (None if clear)
    
    Example Usage:
        # Check if object is visible from camera
        camera_pos = np.array([0.5, 0, 0.5])
        obj_pos = sim.get_base_position('object_0')
        
        is_clear, blocker = check_line_of_sight(
            sim, camera_pos, obj_pos,
            ignore_bodies=['table']  # Table doesn't block visibility
        )
        
        if is_clear:
            print("Object is visible!")
        else:
            print(f"Object blocked by {blocker}")
    
    Use in Occlusion Detection:
        # Check if object_1 blocks path from object_0 to goal
        obj0_pos = sim.get_base_position('object_0')
        is_clear, blocker = check_line_of_sight(
            sim, obj0_pos, goal_pos,
            ignore_bodies=['object_0', 'table']
        )
        if blocker == 'object_1':
            print("Need to push object_1 aside first!")
    
    PyBullet Ray Test Result:
        result[0] = [objectUniqueId, linkIndex, hit_fraction, hit_position, hit_normal]
        - hit_fraction: 0.0-1.0 (0=hit at start, 1=hit at end, >1=no hit)
        - hit_fraction < 1.0 means something was hit along the ray
    """
    try:
        # Convert numpy arrays to lists (PyBullet API requirement)
        if isinstance(point_a, np.ndarray):
            point_a = point_a.tolist()
        if isinstance(point_b, np.ndarray):
            point_b = point_b.tolist()
        
        # Perform ray test (ray-casting from A to B)
        result = sim.physics_client.rayTest(point_a, point_b)
        
        if not result:
            return True, None  # No hit (clear path)
        
        # Extract hit information
        # result is list: [objectUniqueId, linkIndex, hit_fraction, hit_pos, hit_normal]
        hit_body_id = result[0][0]
        
        # ====================================================================
        # CHECK IF HIT BODY SHOULD BE IGNORED
        # ====================================================================
        if ignore_bodies:
            for name, body_id in sim._bodies_idx.items():
                if body_id == hit_body_id and name in ignore_bodies:
                    return True, None  # Hit ignored body = treat as clear
        
        # ====================================================================
        # FIND NAME OF BLOCKING BODY
        # ====================================================================
        blocking_name = None
        for name, body_id in sim._bodies_idx.items():
            if body_id == hit_body_id:
                blocking_name = name
                break
        
        # ====================================================================
        # CHECK HIT FRACTION
        # ====================================================================
        # hit_fraction < 1.0 means ray hit something before reaching point_b
        # hit_fraction = 0.5 means hit occurred halfway between A and B
        if result[0][2] < 1.0:
            return False, blocking_name  # Path blocked
        
        return True, None  # Ray reached end point (clear)
        
    except Exception as e:
        print(f"Warning: Error in ray test: {e}")
        # Fail-safe: assume clear on error (prevents false positives)
        return True, None

def enable_collision(sim, body_name1: str, body_name2: str, enable: bool = True):
    """
    Enable or disable collision detection between two bodies.
    
    Allows selective collision filtering - useful for:
    - Ignoring robot self-collisions (disable within robot)
    - Allowing gripper to pass through target (disable during approach)
    - Preventing ghost collisions (disable fallen objects)
    - Testing scenarios (disable table collision for debugging)
    
    Warning:
        Disabling collisions can lead to interpenetration (objects overlap).
        Use carefully and only when necessary!
    
    Use Cases in Manipulation:
    1. Grasp planning: Temporarily disable collision between gripper and target
       during approach trajectory (re-enable after grasp)
    2. Multi-robot: Disable collision between robots that won't interact
    3. Debugging: Disable problematic collisions to isolate issues
    
    Args:
        sim: PyBullet simulation instance
        body_name1: First body name (e.g., "panda")
        body_name2: Second body name (e.g., "object_0")
        enable: True to enable collisions (default)
               False to disable collisions (objects pass through each other)
    
    Returns:
        None (modifies collision filtering in physics engine)
    
    Example Usage:
        # Disable collision between robot and target during grasp approach
        enable_collision(sim, 'panda', 'object_0', enable=False)
        # ... execute approach motion ...
        # Re-enable collision for grasp
        enable_collision(sim, 'panda', 'object_0', enable=True)
        
        # Ignore robot self-collision (if problematic)
        enable_collision(sim, 'panda', 'panda', enable=False)
    
    PyBullet Implementation:
        setCollisionFilterPair() controls collision matrix:
        - enableCollision=1: Bodies can collide (default)
        - enableCollision=0: Bodies pass through each other (ghost mode)
    
    Limitation:
        Currently only affects base links (link index -1).
        For full robot collision control, would need to iterate over all link pairs.
    """
    try:
        body_id1 = sim._bodies_idx.get(body_name1)
        body_id2 = sim._bodies_idx.get(body_name2)
        
        if body_id1 is None or body_id2 is None:
            return  # Can't disable collision for non-existent bodies
        
        # Base link index (-1 for simple objects, or specific link IDs)
        link_id = -1
        
        # Set collision filter pair
        # enableCollision: 0=disable, 1=enable
        sim.physics_client.setCollisionFilterPair(
            body_id1, body_id2,
            link_id, link_id,
            enableCollision=int(enable)# Convert bool to int (0 or 1)
        )
        
    except Exception as e:
        print(f"Warning: Could not set collision state: {e}")
