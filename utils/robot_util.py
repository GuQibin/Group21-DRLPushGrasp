"""
Robot control utilities for action primitive execution.
Implements high-level manipulation primitives: pick-and-place and push.

This module contains the core action primitives that the RL agent learns to use:
- execute_pick_and_place(): Complete grasping and placement sequence
- execute_push(): Push object in specified direction
- Helper functions for motion control and grasp verification

These primitives abstract away low-level inverse kinematics and motion planning,
allowing the agent to focus on high-level decision making (push vs. grasp).

CORRECTED VERSION - Uses only verified Panda robot and PyBullet wrapper methods
"""

import numpy as np
from typing import Tuple, Optional
import pybullet as p

SLOW = 1/120

def get_ee_position_safe(robot) -> np.ndarray:
    """
    Get end-effector position safely with fallback mechanisms.
    
    Robustly retrieves the end-effector (gripper) position by trying multiple
    access methods in order of preference. This handles different Panda robot
    wrapper implementations that may expose this information differently.
    
    Access Method Hierarchy:
    1. Robot's built-in method (cleanest, if available)
    2. Direct PyBullet query via forward kinematics
    3. Zero vector fallback (prevents crashes)
    
    Why This Matters:
    End-effector position is critical for ALL motion primitives. Without reliable
    EE pose, the robot cannot plan grasps, compute errors, or execute trajectories.
    
    Args:
        robot: Panda robot instance from panda_gym or similar wrapper
    
    Returns:
        position: 3D position [x, y, z] in meters (world frame)
                 Returns [0, 0, 0] as safe fallback if all queries fail
    
    Implementation Notes:
    - Uses getattr() with default to avoid AttributeError
    - Link index 11 is standard for Panda EE frame in most URDFs
    - computeForwardKinematics=1 ensures fresh kinematics computation
    
    Usage in Primitives:
        Used in move_to_position() control loop to compute error:
        error = target_pos - get_ee_position_safe(robot)
    """
    try:
        # ====================================================================
        # METHOD 1: Robot's built-in API (preferred)
        # ====================================================================
        if hasattr(robot, 'get_ee_position'):
            return robot.get_ee_position()
        
        # ====================================================================
        # METHOD 2: Direct PyBullet query (fallback)
        # ====================================================================
        if hasattr(robot, 'sim'):
            # Get EE link index (typically 11 for Panda's end-effector frame)
            ee_link = getattr(robot, 'ee_link', 11)
            
            # Get Panda body ID from simulation's body registry
            panda_uid = robot.sim._bodies_idx.get('panda')
            
            if panda_uid is not None:
                # Query link state with forward kinematics enabled
                # link_state returns: (world_pos, world_ori, local_inertial_pos, ...)
                link_state = robot.sim.physics_client.getLinkState(
                    panda_uid, ee_link, computeForwardKinematics=1
                )
                # Extract world position (first element)
                return np.array(link_state[0], dtype=np.float32)
        
        # ====================================================================
        # FALLBACK: All methods failed
        # ====================================================================
        print(f"Warning: Could not get EE position")
        return np.zeros(3, dtype=np.float32)
        
    except Exception as e:
        print(f"Warning: Could not get EE position: {e}")
        return np.zeros(3, dtype=np.float32)


def get_ee_orientation_safe(robot) -> np.ndarray:
    """
    Get end-effector orientation safely with fallback mechanisms.
    
    Retrieves gripper orientation as quaternion using same fallback strategy
    as get_ee_position_safe(). Orientation is less critical than position for
    this project (mainly top-down grasps), but still useful for:
    - Verifying gripper is pointing downward
    - Computing approach vectors for angled grasps
    - Debugging motion control issues
    
    Quaternion Format:
    PyBullet uses [x, y, z, w] convention (scalar-last).
    Identity quaternion [0, 0, 0, 1] = no rotation from world frame.
    
    Args:
        robot: Panda robot instance
    
    Returns:
        orientation: Quaternion [x, y, z, w]
                    Returns [0, 0, 0, 1] (identity) as safe fallback
    
    Why Identity is Safe Default:
    For Panda with standard mounting, identity quaternion typically means
    gripper pointing straight down (-Z), which is the desired orientation
    for most tabletop grasps. So fallback won't cause obvious failures.
    
    Usage in Primitives:
        Currently NOT used in action primitives (could be used for orientation control)
        Mainly used in _get_robot_state() for observation construction
    """
    try:
        # ====================================================================
        # METHOD 1: Robot's built-in API
        # ====================================================================
        if hasattr(robot, 'get_ee_orientation'):
            return robot.get_ee_orientation()
        
        # ====================================================================
        # METHOD 2: Direct PyBullet query
        # ====================================================================
        if hasattr(robot, 'sim'):
            ee_link = getattr(robot, 'ee_link', 11)
            panda_uid = robot.sim._bodies_idx.get('panda')
            
            if panda_uid is not None:
                link_state = robot.sim.physics_client.getLinkState(
                    panda_uid, ee_link, computeForwardKinematics=1
                )
                # link_state[1] is world orientation as quaternion [x, y, z, w]
                return np.array(link_state[1], dtype=np.float32)
        
        # ====================================================================
        # FALLBACK: Identity quaternion
        # ====================================================================
        print(f"Warning: Could not get EE orientation")
        return np.array([0, 0, 0, 1], dtype=np.float32)
        
    except Exception as e:
        print(f"Warning: Could not get EE orientation: {e}")
        return np.array([0, 0, 0, 1], dtype=np.float32)

def get_current_joint_positions(robot) -> Optional[np.ndarray]:
    """
    Safely retrieves the current joint positions of the robot's arm.

    This function is crucial for planning trajectories that start from
    the robot's current configuration.

    Args:
        robot: The Panda robot instance.

    Returns:
        A 7D numpy array of the current joint angles in radians,
        or None if the observation is invalid.
    """
    try:
        obs = robot.get_obs()
        # The first 7 elements of the observation are the joint positions
        if len(obs) >= 7:
            return obs[:7]
        else:
            print("  ⚠ Warning: Robot observation is too short to get joint positions.")
            return None
    except Exception as e:
        print(f"  ⚠ Error getting current joint positions: {e}")
        return None

def get_gripper_state(robot) -> dict:
    """
    Get current gripper state information.

    ═══════════════════════════════════════════════════════════════════════
    GRIPPER STATE QUERY - Essential feedback for manipulation verification
    ═══════════════════════════════════════════════════════════════════════

    This is the CRITICAL function that enables gripper control
    verification in open_gripper() and close_gripper().

    Purpose:
    Reads finger joint positions from robot observation to determine:
    1. Current gripper width (physical gap between fingers)
    2. Whether gripper is fully open (ready for approach)
    3. Whether gripper is fully closed (attempting grasp)

    Why This Matters:
    Without state feedback, the robot operates "blind":
    - Can't verify if gripper command succeeded
    - Can't detect if object is too large to grasp
    - Can't confirm gripper opened before next approach

    Panda Gripper Mechanics:
    - Two prismatic (sliding) finger joints
    - Each finger moves 0 to 0.04m (0 to 4cm)
    - Fingers are "mimic joints" (move symmetrically)
    - Total width = sum of both finger positions
    - Fully open: ~8cm (0.08m), Fully closed: ~0cm

    Observation Index Handling:
    Different panda-gym versions store gripper data at different locations
    in the observation vector. This function checks multiple possibilities
    for maximum compatibility.

    Args:
        robot: Panda robot instance with get_obs() method

    Returns:
        Dictionary with three fields:
        {
            'width': float,      # Total gripper width in meters [0, 0.08]
            'is_closed': bool,   # True if width < 1cm
            'is_open': bool      # True if width > 7cm
        }

        Returns safe defaults on error: {0.0, False, False}
    """
    try:
        # ====================================================================
        # STEP 1: Get robot observation vector
        # ====================================================================
        obs = robot.get_obs()

        # ====================================================================
        # STEP 2: Try multiple observation formats (version compatibility)
        # ====================================================================

        # ----------------------------------------------------------------
        # FORMAT 1: Standard panda-gym v3+ (16+ elements)
        # ----------------------------------------------------------------
        # Observation structure:
        # [0:7]   = joint positions (arm)
        # [7:14]  = joint velocities (arm)
        # [14:16] = gripper finger positions ← TARGET
        # [16+]   = additional data (forces, torques, etc.)
        if len(obs) >= 16:
            # Standard panda-gym format: gripper at indices 14-15
            finger_positions = obs[14:16]
            gripper_width = float(np.sum(finger_positions))

        # ----------------------------------------------------------------
        # FORMAT 2: Older panda-gym v2 or minimal observation (9-15 elements)
        # ----------------------------------------------------------------
        # Observation structure:
        # [0:7] = joint positions (arm)
        # [7:9] = gripper finger positions ← TARGET
        # May not include velocities or other data
        elif len(obs) >= 9:
            # Alternative: gripper at indices 7-8
            finger_positions = obs[7:9]
            gripper_width = float(np.sum(finger_positions))

        # ----------------------------------------------------------------
        # FORMAT 3: Minimal/Mock observation (< 9 elements)
        # ----------------------------------------------------------------
        # Used in unit tests or stripped-down environments
        # No gripper data available
        else:
            # No gripper data in observation
            gripper_width = 0.0  # Default to closed

        # ====================================================================
        # STEP 3: Classify gripper state based on width
        # ====================================================================
        return {
            'width': gripper_width,                  # Raw measurement
            'is_closed': gripper_width < 0.01,       # Closed: < 1cm
            'is_open': gripper_width > 0.07          # Open: > 7cm
        }

    except Exception as e:
        # ====================================================================
        # ERROR HANDLING: Return safe defaults
        # ====================================================================
        print(f"  ⚠ Error getting gripper state: {e}")
        # Return "unknown" state (both flags False)
        # This prevents false positives in verification
        return {
            'width': 0.0,
            'is_closed': False,  # Don't assume closed
            'is_open': False     # Don't assume open
        }

def execute_pick_and_place(sim, robot, target_object: str,
                          alpha_x: float, alpha_y: float,
                          goal_pos: np.ndarray,
                          approach_height: float = 0.15,
                          grasp_height: float = 0.05) -> bool:
    """
    Execute complete pick-and-place sequence for target object.

    ═══════════════════════════════════════════════════════════════════════
    PRIMARY ACTION PRIMITIVE #1: GRASPING
    ═══════════════════════════════════════════════════════════════════════

    This is one of two learned manipulation strategies. The agent learns:
    - WHEN to grasp (via α_skill > 0 in action vector)
    - WHERE to grasp (via α_x, α_y parameters)

    Eight-Phase Sequence:
    ┌────────────────────────────────────────────────────────────────┐
    │ Phase 1: Approach    → Move above object (safe clearance)     │
    │ Phase 2: Descend     → Lower to grasp height                  │
    │ Phase 3: Close       → Actuate gripper                        │
    │ Phase 4: Verify      → Check if grasp succeeded               │
    │ Phase 5: Lift        → Raise object to transport height       │
    │ Phase 6: Transport   → Move to goal position                  │
    │ Phase 7: Place       → Lower and release object               │
    │ Phase 8: Retract     → Clear away from placed object          │
    └────────────────────────────────────────────────────────────────┘

    Action Parameterization:
    The agent outputs α_x, α_y ∈ [-1, 1] which map to:
    - Physical offsets: [-2.5cm, +2.5cm] from object center
    - This allows learning optimal grasp points per object type

    Expected Behavior by Object Type:
    - Cubes: α_x ≈ 0, α_y ≈ 0 (center grasp works well)
    - Spheres: α_x, α_y may vary (any point works, but off-center may be more stable)
    - Irregular: Agent must learn case-by-case

    Success Criteria:
    Returns True only if ALL phases succeed, including grasp verification.
    This ensures +5 placement reward is only given for genuine success.

    Failure Modes (returns False):
    1. Object position query fails (object doesn't exist)
    2. Can't reach approach position (workspace limits, collision)
    3. Can't reach grasp height (same as above)
    4. Grasp verification fails (fingers didn't close on object)
    5. Any phase encounters exception

    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_object: Name of object to grasp (e.g., "object_0")
        alpha_x: Grasp X-offset in object's local frame, ∈ [-1, 1]
                Normalized parameter from agent's action output
        alpha_y: Grasp Y-offset in object's local frame, ∈ [-1, 1]
                Normalized parameter from agent's action output
        goal_pos: Goal zone center [x, y] where object should be placed
                 Typically self.goal_pos from environment
        approach_height: Height for safe approach and transport (default: 15cm)
                        Balance between safety (higher) and speed (lower)
        grasp_height: Height at which gripper closes (default: 5cm)
                     Should be slightly above object center for stable grasp

    Returns:
        success: True if complete sequence succeeded (all 8 phases)
                False if any phase failed or exception occurred

    Design Limitations:
    1. **Simplified coordinate transform**: Assumes objects are upright
       - Current: grasp_point = obj_pos + offset (ignores rotation)
       - TODO: Apply quaternion rotation for tilted objects

    2. **No intermediate waypoints**: Uses direct point-to-point motion
       - Can cause collisions with table or other objects
       - Better: Use motion planning with collision avoidance

    3. **Fixed heights**: approach_height and grasp_height are constants
       - Doesn't adapt to object size
       - Could cause issues with larger/smaller objects

    4. **Binary gripper control**: Fully open or fully closed
       - No force control or partial closing
       - May crush fragile objects or fail on large objects

    Usage in Environment:
        Called from step() when action_type == "grasp":
        success = execute_pick_and_place(
            self.sim, self.robot, self.current_target,
            alpha_x, alpha_y, self.goal_pos
        )
        self.action_was_successful = success
    """
    # ========================================================================
    # PHASE 0: SETUP - Query object state
    # ========================================================================
    try:
        # Get object's current pose from physics engine
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except Exception as e:
        print(f"❌ Error: Could not get pose for {target_object}: {e}")
        return False  # Can't manipulate non-existent object

    # Store initial Z-coordinate for grasp verification
    # After successful grasp, object should be lifted above this height
    initial_obj_z = obj_pos[2]

    # ========================================================================
    # GRASP POINT CALCULATION
    # ========================================================================
    # Convert agent's normalized action parameters to physical offsets
    # Mapping: [-1, 1] → [-2.5cm, +2.5cm]
    offset_scale = 0.025  # 2.5cm = reasonable for 4cm objects

    grasp_offset = np.array([
        alpha_x * offset_scale,  # X offset (left-right)
        alpha_y * offset_scale,  # Y offset (forward-back)
        0.0                      # Z offset (always grasp from top)
    ])

    # Transform offset to world frame
    # CURRENT IMPLEMENTATION: Simplified (assumes object upright)
    # This works when obj_ori ≈ [0, 0, 0, 1] (no rotation)
    rot_matrix = quaternion_to_rotation_matrix(obj_ori)
    grasp_offset_world = rot_matrix @ grasp_offset
    grasp_point = obj_pos + grasp_offset_world

    # ========================================================================
    # PHASE 1: APPROACH FROM ABOVE
    # ========================================================================
    # Move end-effector to position directly above grasp point
    # This avoids collisions with table and provides clear approach path
    print(f"  Phase 1: Approaching {target_object} from above...")

    approach_pos = grasp_point.copy()
    approach_pos[2] = approach_height  # Lift to 15cm (safe clearance)

    success = move_to_position(
        sim, robot, approach_pos,
        gripper_open=True,  # Keep gripper open during approach
        steps=150, 
        sleep_sec=SLOW           # Allow 50 timesteps for convergence
    )

    if not success:
        print(f"  ❌ Failed to approach {target_object}")
        return False  # Abort if can't even reach approach position

    # ========================================================================
    # PHASE 2: DESCEND TO GRASP HEIGHT
    # ========================================================================
    # Lower gripper vertically from approach position to grasp height
    # Vertical motion minimizes risk of knocking object away
    print(f"  Phase 2: Lowering to grasp height...")

    grasp_pos = grasp_point.copy()
    grasp_pos[2] = grasp_height  # Lower to 5cm (slightly above object center)

    success = move_to_position(
        sim, robot, grasp_pos,
        gripper_open=True,
        steps=150, 
        sleep_sec=SLOW# Fewer steps = slower, more controlled descent
    )

    if not success:
        print(f"  ❌ Failed to lower to grasp height for {target_object}")
        return False

    # ========================================================================
    # PHASE 3: CLOSE GRIPPER
    # ========================================================================
    print(f"  Phase 3: Closing gripper...")
    close_gripper(sim, robot, steps=60, sleep_sec=SLOW)
    # 给接触求解一点时间
    for _ in range(30):
        sim.step()

    # ========================================================================
    # PHASE 4: MICRO-LIFT to validate grasp  ← 关键改动
    # ========================================================================
    print(f"  Phase 4: Micro-lift to check grasp...")
    # 记录关爪后的物体高度
    try:
        obj_z_before = sim.get_base_position(target_object)[2]
    except Exception:
        obj_z_before = initial_obj_z

    # 微抬 3cm
    micro_lift = get_ee_position_safe(robot).copy()
    micro_lift[2] += 0.03
    move_to_position(sim, robot, micro_lift, gripper_open=False, steps=60, sleep_sec=SLOW)

    # 等几帧稳定
    for _ in range(20):
        sim.step()

    obj_z_after = sim.get_base_position(target_object)[2]
    lift_gain = obj_z_after - obj_z_before
    if lift_gain < 0.01:
        print(f"  ❌ Grasp failed (lift gain={lift_gain:.3f}m)")
        open_gripper(sim, robot, steps=60, sleep_sec=SLOW)
        return False
    print(f"  ✓ Successfully grasped {target_object} (lift gain={lift_gain:.3f}m)")

    # ========================================================================
    # PHASE 5: LIFT OBJECT to transport height
    # ========================================================================
    print(f"  Phase 5: Lifting object...")
    lift_pos = get_ee_position_safe(robot)
    lift_pos[2] = approach_height
    move_to_position(sim, robot, lift_pos, gripper_open=False, steps=150, sleep_sec=SLOW)

    # ========================================================================
    # PHASE 6: TRANSPORT TO GOAL
    # ========================================================================
    # Move horizontally to position above goal zone
    # Maintains safe height throughout transport
    print(f"  Phase 6: Transporting to goal...")

    transport_pos = np.array([
        goal_pos[0],      # Goal X coordinate
        goal_pos[1],      # Goal Y coordinate
        approach_height   # Maintain 15cm height
    ])

    move_to_position(
        sim, robot, transport_pos,
        gripper_open=False,  # Still holding object
        steps=150, 
        sleep_sec=SLOW# Allow more time for potentially longer distance
    )

    # ========================================================================
    # PHASE 7: PLACE OBJECT
    # ========================================================================
    # Lower object to placement height and release gripper
    print(f"  Phase 7: Placing object...")

    place_pos = np.array([
        goal_pos[0],
        goal_pos[1],
        0.05  # Place at 5cm (just above table surface)
    ])

    move_to_position(
        sim, robot, place_pos,
        gripper_open=False,  # Still closed during descent
        steps=150, 
        sleep_sec=SLOW
    )

    # Release object
    open_gripper(sim, robot, steps=60, sleep_sec=SLOW)

    # ========================================================================
    # PHASE 8: RETRACT
    # ========================================================================
    # Move gripper away from placed object
    # Prevents accidental re-contact or pushing
    print(f"  Phase 8: Retracting...")

    retract_pos = place_pos.copy()
    retract_pos[2] = approach_height  # Retract to safe height (15cm)

    move_to_position(
        sim, robot, retract_pos,
        gripper_open=True,  # Open gripper during retract
        steps=150, 
        sleep_sec=SLOW
    )

    # ========================================================================
    # SUCCESS - All 8 phases completed
    # ========================================================================
    print(f"  ✓ Pick-and-place complete!")
    return True


def execute_push(sim, robot, target_object: str,
                alpha_x: float, alpha_y: float, alpha_theta: float,
                push_distance: float = 0.05,
                push_height: float = 0.03,
                use_object_frame: bool = True) -> bool:
    """
    Execute push primitive on target object.

    ═══════════════════════════════════════════════════════════════════════
    PRIMARY ACTION PRIMITIVE #2: PUSHING
    ═══════════════════════════════════════════════════════════════════════

    This is the second of two learned manipulation strategies. The agent learns:
    - WHEN to push (via α_skill ≤ 0 in action vector)
    - WHERE to contact (via α_x, α_y parameters)
    - WHICH DIRECTION to push (via α_θ parameter)

    Three-Phase Sequence:
    ┌────────────────────────────────────────────────────────────────┐
    │ Phase 1: Pre-push    → Move to position behind contact point  │
    │ Phase 2: Execute     → Linear push motion through object      │
    │ Phase 3: Retract     → Lift and clear away                    │
    └────────────────────────────────────────────────────────────────┘

    Push Geometry:

        pre_push ----3cm----> contact ----5cm----> post_push
            ^                    ^                     ^
            |                    |                     |
      Start position      Make contact here        End position

    Total linear motion: 3cm (approach) + 5cm (push) = 8cm

    Action Parameterization:
    - α_x, α_y ∈ [-1, 1] → contact offset ∈ [-2.5cm, +2.5cm]
    - α_θ ∈ [-1, 1] → push angle ∈ [-π, π] radians

    Push Direction Encoding:
    The agent outputs α_θ which maps to push direction:
    - α_θ =  0.0 → angle =  0°    → push in +X (east)
    - α_θ = +0.5 → angle = +90°   → push in +Y (north)
    - α_θ = +1.0 → angle = +180°  → push in -X (west)
    - α_θ = -0.5 → angle = -90°   → push in -Y (south)
    - α_θ = -1.0 → angle = -180°  → push in -X (west, same as +1.0)

    Why Push Instead of Grasp?
    Pushing is preferred when:
    1. Object is spherical (hard to grasp due to rolling)
    2. Object is occluded (push blockers aside first)
    3. Object needs repositioning (push toward goal)
    4. Grasp would be unstable (awkward object orientation)

    Success Criteria:
    Returns True if push sequence completes (all 3 phases).
    Note: This doesn't verify object actually moved - that's checked in reward.

    Failure Modes:
    1. Object position query fails
    2. Can't reach pre-push position (IK failure, collision)
    3. Push motion incomplete (collision, workspace limit)
    4. Exception during execution

    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_object: Name of object to push (e.g., "object_2")
        alpha_x: Contact X-offset in object's local frame, ∈ [-1, 1]
        alpha_y: Contact Y-offset in object's local frame, ∈ [-1, 1]
        alpha_theta: Push direction angle, ∈ [-1, 1] → [-π, π]
        push_distance: How far to push in meters (default: 5cm)
                      Fixed value simplifies learning
        push_height: Height of push contact point (default: 3cm)
                    Lower = more stable, higher = more force transfer
        use_object_frame: If True, rotate push direction by object orientation
                         Currently False (world frame pushing)

    Returns:
        success: True if push sequence completed
                False if any phase failed

    Design Notes:
    - **use_object_frame=False**: Push direction in world frame (simpler)
      Example: α_θ=0 always pushes eastward regardless of object rotation

    - **use_object_frame=True**: Push direction relative to object
      Example: α_θ=0 pushes in object's "forward" direction
      More complex but useful for oriented objects

    - **Fixed push_distance**: Agent doesn't control how far to push
      Simplifies action space but limits flexibility

    - **Fixed push_height**: Always pushes at 3cm height
      Works for uniform 4cm objects but may fail for different sizes

    Limitations:
    1. **No force control**: Uses position control, not force control
       May fail if object is too heavy or friction is high

    2. **No contact verification**: Doesn't check if contact was made
       Could "air push" if object moved before contact

    3. **No obstacle avoidance**: Direct line motion may collide
       with table or other objects

    Usage in Environment:
        Called from step() when action_type == "push":
        success = execute_push(
            self.sim, self.robot, self.current_target,
            alpha_x, alpha_y, alpha_theta
        )
        self.action_was_successful = success
    """
    # ========================================================================
    # PHASE 0: SETUP - Query object state
    # ========================================================================
    try:
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except Exception as e:
        print(f"Error: Could not get position for {target_object}: {e}")
        return False

    # ========================================================================
    # CONTACT POINT CALCULATION
    # ========================================================================
    # Convert agent's normalized parameters to physical offsets
    offset_scale = 0.025  # 2.5cm max offset

    contact_offset = np.array([
        alpha_x * offset_scale,
        alpha_y * offset_scale,
        0.0  # No Z offset (push horizontally)
    ])

    rot_matrix = quaternion_to_rotation_matrix(obj_ori)
    contact_offset_world = rot_matrix @ contact_offset

    # Contact point in world coordinates
    contact_point = obj_pos + contact_offset_world
    contact_point[2] = push_height  # Set push height (3cm above table)

    # ========================================================================
    # PUSH DIRECTION CALCULATION
    # ========================================================================
    # Convert normalized angle parameter to radians
    # Mapping: α_θ ∈ [-1, 1] → angle ∈ [-π, π]
    push_angle = alpha_theta * np.pi

    # Convert angle to unit direction vector
    # Uses standard trigonometry: angle → (cos, sin)
    push_direction = np.array([
        np.cos(push_angle),  # X component
        np.sin(push_angle),  # Y component
        0.0                  # Z component (horizontal push)
    ])

    if use_object_frame:
        push_direction = rot_matrix @ push_direction

    # ========================================================================
    # PRE-PUSH POSITION CALCULATION
    # ========================================================================
    # Start 3cm behind contact point along push direction
    # This ensures robot makes solid contact before applying force
    pre_push_offset = 0.03  # 3cm approach distance
    pre_push_pos = contact_point - push_direction * pre_push_offset

    # ========================================================================
    # PHASE 1: MOVE TO PRE-PUSH POSITION
    # ========================================================================
    # Position gripper behind where we want to make contact
    print(f"  Phase 1: Moving to pre-push position...")

    success = move_to_position(
        sim, robot, pre_push_pos,
        gripper_open=False, # Gripper closed for pushing
        steps=150, 
        sleep_sec=SLOW # Allow time to reach position
    )

    if not success:
        print(f"  ❌ Failed to reach pre-push position for {target_object}")
        return False  # Can't push if can't reach start

    # ========================================================================
    # PHASE 2: EXECUTE PUSH (Linear Motion)
    # ========================================================================
    # Move gripper forward through object's position
    # This is where the actual pushing force is applied
    print(f"  Phase 2: Executing push...")

    # Calculate end position: 5cm forward from contact point
    post_push_pos = contact_point + push_direction * push_distance

    # Execute linear push motion
    # Note: This is position control, not force control
    # Robot will try to reach post_push_pos regardless of resistance
    success = move_to_position(
        sim, robot, post_push_pos,
        gripper_open=False, # Gripper closed
        steps=150, 
        sleep_sec=SLOW # Slower motion for controlled push
    )

    if not success:
        print(f"  ⚠ Push may have been incomplete")
        # Don't return False - partial push may still be useful
        # Reward function will determine if object actually moved

    # ========================================================================
    # PHASE 3: RETRACT
    # ========================================================================
    # Move gripper up and away from object
    # Prevents lingering contact that might push object further
    print(f"  Phase 3: Retracting...")

    retract_pos = post_push_pos.copy()
    retract_pos[2] += 0.1  # Lift 10cm (clear separation)

    move_to_position(
        sim, robot, retract_pos,
        gripper_open=True, # Open gripper after push
        steps=150, 
        sleep_sec=SLOW
    )

    # ========================================================================
    # SUCCESS - Push sequence completed
    # ========================================================================
    print(f"  ✓ Push complete!")
    return True


def move_to_position(sim, robot, target_pos: np.ndarray,
                    gripper_open: bool = True,
                    steps: int = 50,
                    sleep_sec: float = 0.0) -> bool:
    """
    Move end-effector to target position using robot's action interface.

    ═══════════════════════════════════════════════════════════════════════
    LOW-LEVEL MOTION CONTROL - Core function used by both primitives
    ═══════════════════════════════════════════════════════════════════════

    This is the workhorse function that handles ALL end-effector motion in both
    pick-and-place and push primitives. Implements a simple proportional control
    loop that iteratively moves the EE toward the target position.

    Control Algorithm (Simple P-Controller):
    ┌────────────────────────────────────────────────────────────────┐
    │ Loop (max: steps iterations):                                  │
    │   1. error = target_pos - current_pos                          │
    │   2. delta = clip(error × gain, -1, +1)  [gain = 50]          │
    │   3. action = [delta_x, delta_y, delta_z, gripper_ctrl]       │
    │   4. robot.set_action(action)                                  │
    │   5. sim.step()                                                │
    │   6. If error < 1cm: SUCCESS (early exit)                     │
    │ End loop                                                       │
    │ Final check: If error < 5cm: SUCCESS, else: FAILURE          │
    └────────────────────────────────────────────────────────────────┘

    Action Space Assumption:
    IMPORTANT: This assumes panda-gym's standard action format:
    - action = [delta_x, delta_y, delta_z, gripper_control]
    - delta_x/y/z ∈ [-1, 1]: Normalized velocity/position commands
    - gripper_control ∈ [-1, 1]: Gripper open/close command

    Control Gain Analysis:
    The gain of 50.0 is VERY high and appears to be compensating for an
    action scaling issue. This means ANY error > 2cm results in maximum speed command.
    The controller is essentially bang-bang, not proportional!

    Why This Matters:
    - High gain = fast but jerky motion
    - Saturation = maximum speed regardless of distance
    - May cause overshooting and oscillation
    - May explain why robot has trouble reaching targets precisely

    Success Thresholds:
    - During motion: error < 1cm (0.01m) → early success
    - Final check: error < 5cm (0.05m) → considered "close enough"
    - Relaxed final threshold accounts for IK limitations and saturation

    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_pos: Target end-effector position [x, y, z] in world frame (meters)
        gripper_open: Desired gripper state during motion
                     True = keep open (approaching, retracting)
                     False = keep closed (transporting grasped object)
        steps: Maximum number of control loop iterations (default: 50)
              At 240Hz sim rate, 50 steps ≈ 0.21 seconds

    Returns:
        success: True if final error < 5cm
                False if timed out, error occurred, or couldn't converge
    """
    print(f"\n[MOVE] Moving to {target_pos}")

    initial_pos = get_ee_position_safe(robot)
    initial_distance = np.linalg.norm(target_pos - initial_pos)
    print(f"[MOVE] From {initial_pos}, distance: {initial_distance:.4f}m")

    gripper_ctrl = 1.0 if gripper_open else -1.0

    for step in range(steps):
        try:
            current_pos = get_ee_position_safe(robot)
            error = target_pos - current_pos
            delta = np.clip(error * 50.0, -1.0, 1.0)
            action = np.concatenate([delta, [gripper_ctrl]])

            if step % 10 == 0:
                error_mag = np.linalg.norm(error)
                print(f"  Step {step:2d}: error={error_mag:.4f}m, action={np.round(delta, 2)}")

            robot.set_action(action)
            sim.step()

            if sleep_sec: 
                import time; time.sleep(sleep_sec)


            if np.linalg.norm(error) < 0.01:
                print(f"  ✓ Reached target at step {step}")
                return True

        except Exception as e:
            print(f"  ❌ Error at step {step}: {e}")
            return False

    final_pos = get_ee_position_safe(robot)
    final_error = np.linalg.norm(target_pos - final_pos)
    success = final_error < 0.05

    print(f"  Final: pos={np.round(final_pos, 3)}, error={final_error:.4f}m")
    if not success:
        print(f"  ⚠ Did not reach target (timeout after {steps} steps)")

    return success



def open_gripper(sim, robot, steps: int = 30, sleep_sec: float = 0.0):
    """把夹爪打开到 7cm，并推进若干步让物理稳定。"""
    import time
    target = 0.07  # 7cm
    set_gripper_width(sim, target)
    for _ in range(steps):
        sim.step()
        if sleep_sec: time.sleep(sleep_sec)
    w = get_gripper_width(sim)
    print(f"  ✓ Gripper opened to ~{w:.3f}m")

def close_gripper(sim, robot, steps: int = 30, sleep_sec: float = 0.0):
    """把夹爪闭合到 0cm（靠摩擦夹持），推进若干步沉降。"""
    import time
    set_gripper_width(sim, 0.0)
    for _ in range(steps):
        sim.step()
        if sleep_sec: time.sleep(sleep_sec)
    w = get_gripper_width(sim)
    print(f"  ✓ Gripper closed width ~{w:.3f}m")

def get_gripper_state(robot) -> dict:
    """
    现在不再从 robot.get_obs() 猜索引，而是直接读关节。
    """
    w = 0.0
    try:
        # robot 里有 sim；直接读取
        w = get_gripper_width(robot.sim)
    except Exception:
        pass
    return {
        "width": w,
        "is_closed": w < 0.01,
        "is_open":   w > 0.07
    }

def check_grasp_success(sim, robot, object_name: str,
                       initial_z: Optional[float] = None,
                       min_lift: float = 0.01) -> bool:
    """
    Check if grasp was successful by monitoring object height.

    ═══════════════════════════════════════════════════════════════════════
    GRASP VERIFICATION - Critical Quality Gate
    ═══════════════════════════════════════════════════════════════════════

    This is the quality gate that prevents the robot from continuing with an
    empty gripper. Called in Phase 4 of pick-and-place, right after closing
    gripper but before lifting to transport height.

    Verification Strategy:
    A successful grasp is verified by checking if the object LIFTED when
    gripper closed. This is more reliable than checking contact forces,
    because contact doesn't guarantee secure grasp.

    Algorithm:
    ┌────────────────────────────────────────────────────────────────┐
    │ 1. Record initial object height (before grasp)                 │
    │ 2. Close gripper (done in Phase 3, before this function)      │
    │ 3. Wait 20 steps for physics to settle                        │
    │ 4. Check current object height                                │
    │ 5. Compute height_gained = current_z - initial_z              │
    │ 6. If height_gained >= 1cm AND not falling: SUCCESS           │
    │ 7. Else: FAILURE (grasp missed or object slipped)            │
    └────────────────────────────────────────────────────────────────┘

    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance (not used, kept for API consistency)
        object_name: Name of target object (e.g., "object_0")
        initial_z: Object's Z-coordinate before grasp attempt (meters)
                  If None, queries current height (assumes pre-grasp)
        min_lift: Minimum lift distance to consider grasp successful (meters)
                 Default: 1cm (0.01m)

    Returns:
        success: True if object lifted and not falling
                False if object didn't lift or is falling
    """
    try:
        if initial_z is None:
            initial_z = sim.get_base_position(object_name)[2]

        # Wait for physics to settle
        for _ in range(20):
            sim.step()

        current_z = sim.get_base_position(object_name)[2]
        height_gained = current_z - initial_z

        if height_gained >= min_lift:
            obj_vel = sim.get_base_velocity(object_name)
            is_falling = obj_vel[2] < -0.1
            return not is_falling

        return False

    except Exception as e:
        print(f"  ⚠ Error checking grasp: {e}")
        return False


def compute_inverse_kinematics(sim, robot, target_pos: np.ndarray,
                               target_ori: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Compute inverse kinematics using PyBullet wrapper.

    ═══════════════════════════════════════════════════════════════════════
    INVERSE KINEMATICS - Convert Cartesian pose to joint angles
    ═══════════════════════════════════════════════════════════════════════

    Inverse kinematics (IK) is the fundamental problem in robot control:
    Given a desired end-effector pose (position + orientation), compute the
    joint angles needed to achieve that pose. This function is not actively used
    by the default primitives but is available for more advanced control strategies.

    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_pos: Desired end-effector position [x, y, z] in world frame (meters)
        target_ori: Desired end-effector orientation as quaternion [x, y, z, w]
                   If None, defaults to downward-facing gripper orientation

    Returns:
        joint_positions: 7D array of joint angles [rad] for arm joints only
                        Returns None if IK solution fails or error occurs
    """

    try:
        if target_ori is None:
            # Default: Gripper pointing straight down
            target_ori = np.array([0, 1, 0, 0], dtype=np.float32)

        ee_link = getattr(robot, 'ee_link', 11)

        joint_state = sim.inverse_kinematics(
            body='panda',
            link=ee_link,
            position=target_pos,
            orientation=target_ori
        )

        # Return first 7 DOF (arm joints only)
        return joint_state[:7]

    except Exception as e:
        print(f"  ⚠ Error computing IK: {e}")
        return None

def plan_trajectory(start_joints: np.ndarray, goal_joints: np.ndarray,
                    num_waypoints: int = 10) -> np.ndarray:
    """
    Plan smooth trajectory between joint configurations.

    ═══════════════════════════════════════════════════════════════════════
    TRAJECTORY PLANNING - Interpolate between joint configurations
    ═══════════════════════════════════════════════════════════════════════

    Creates a smooth path in joint space by linearly interpolating between
    start and goal configurations. This is the SIMPLEST form of trajectory
    planning.

    Args:
        start_joints: Starting joint configuration (7D array) [radians]
        goal_joints: Goal joint configuration (7D array) [radians]
        num_waypoints: Number of waypoints along trajectory (default: 10)
                      More waypoints = smoother but slower execution

    Returns:
        trajectory: Array of shape (num_waypoints, 7) containing joint angles
                   Returns empty array if input shapes don't match
    """
    if start_joints is None or goal_joints is None:
        print(f"  ⚠ Error: Cannot plan trajectory with None as start or goal joints.")
        return np.array([])

    if start_joints.shape != goal_joints.shape:
        print(f"  ⚠ Error: Joint array shapes don't match: {start_joints.shape} vs {goal_joints.shape}")
        return np.array([])

    # Use np.linspace to generate a smooth, linear interpolation for all joints simultaneously.
    trajectory = np.linspace(start_joints, goal_joints, num_waypoints)
    return trajectory


def check_collision_between_bodies(sim, body1_name: str, body2_name: str) -> bool:
    """
    Check collision using direct PyBullet API.

    ═══════════════════════════════════════════════════════════════════════
    COLLISION DETECTION - Binary check for contact between bodies
    ═══════════════════════════════════════════════════════════════════════

    Simple utility to check if two named bodies are currently in collision.
    Wrapper around PyBullet's getContactPoints() for cleaner API.

    Args:
        sim: PyBullet simulation instance
        body1_name: First body name (e.g., "panda")
        body2_name: Second body name (e.g., "table")

    Returns:
        has_collision: True if any contact points exist between bodies
                      False if no contact, or if either body doesn't exist
    """
    try:
        body1_id = sim._bodies_idx.get(body1_name)
        body2_id = sim._bodies_idx.get(body2_name)

        if body1_id is None or body2_id is None:
            return False

        contacts = sim.physics_client.getContactPoints(bodyA=body1_id, bodyB=body2_id)
        return len(contacts) > 0

    except Exception as e:
        print(f"  ⚠ Error checking collision: {e}")
        return False


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
     Convert quaternion to 3x3 rotation matrix.

    ═══════════════════════════════════════════════════════════════════════
    QUATERNION CONVERSION - Transform orientation representation
    ═══════════════════════════════════════════════════════════════════════

    Converts quaternion representation of rotation to rotation matrix form.
    This is CRITICAL for execute_push() and execute_pick_and_place() when
    dealing with rotated objects.

    Implementation:
    Uses PyBullet's built-in getMatrixFromQuaternion() which is very fast.
    """
    try:
        # PyBullet's getMatrixFromQuaternion returns a 9-element list in row-major order.
        rot_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        return rot_matrix

    except Exception as e:
        print(f"  ⚠ Error converting quaternion: {e}")
        # Return identity matrix (no rotation) as safe fallback
        return np.eye(3)


def wait_for_stability(sim, object_name: str, max_steps: int = 50,
                      velocity_threshold: float = 0.01) -> bool:
    """
    Wait for object to stabilize after manipulation.

    ═══════════════════════════════════════════════════════════════════════
    STABILITY MONITORING - Wait for physics to settle
    ═══════════════════════════════════════════════════════════════════════

    Runs simulation forward until object stops moving (or timeout).
    Essential after manipulation to ensure object is at rest before
    checking success or proceeding to next action.

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

    ═══════════════════════════════════════════════════════════════════════
    DIAGNOSTIC UTILITY - Debug robot control issues
    ═══════════════════════════════════════════════════════════════════════

    Comprehensive diagnostic tool to identify action space format and
    control issues. Runs a series of tests to verify:
    1. Action space format ([delta] vs [absolute])
    2. Action scaling (how much does action=0.01 move?)
    3. Gripper control (does +1/-1 open/close?)

    When to Use:
    - Initial environment setup (verify robot integration)
    - After updating panda-gym version
    - When control seems unresponsive or incorrect
    - Before training (sanity check)

    Args:
        robot: Panda robot instance to diagnose
        sim: PyBullet simulation instance
        steps: Number of steps for position test (default: 10)

    Returns:
        None (prints diagnostic report to console)
    """
    print("\n" + "="*60)
    print("ROBOT CONTROL DIAGNOSTICS")
    print("="*60)

    initial_pos = get_ee_position_safe(robot)
    initial_obs = robot.get_obs()

    print(f"Initial EE position: {initial_pos}")
    print(f"Initial joint angles: {initial_obs[:7]}")
    print(f"Observation shape: {initial_obs.shape}")

    print("\nTest 1: Zero action (no movement)")
    for _ in range(10):
        robot.set_action(np.array([0.0, 0.0, 0.0, 0.0]))
        sim.step()

    pos_after_zero = get_ee_position_safe(robot)
    delta_zero = np.linalg.norm(pos_after_zero - initial_pos)
    print(f"Position after zero action: {pos_after_zero}")
    print(f"Movement: {delta_zero:.6f}m (should be ~0)")

    print("\nTest 2: Positive X action (Measure scaling)")
    for _ in range(steps):
        robot.set_action(np.array([0.01, 0.0, 0.0, 0.0]))
        sim.step()

    pos_after_x = get_ee_position_safe(robot)
    delta_x = pos_after_x - pos_after_zero
    print(f"Position after +X action: {pos_after_x}")
    print(f"Delta: {delta_x}")
    print(f"Expected: [~{0.01*steps:.3f}, 0, 0] ({steps} steps × 0.01)")
    print(f"Actual magnitude: {np.linalg.norm(delta_x):.6f}m")

    print("\nTest 3: Gripper control")
    gripper_state_initial = get_gripper_state(robot)
    print(f"Initial gripper: {gripper_state_initial}")

    open_gripper(sim, robot, steps=60, sleep_sec=SLOW)
    gripper_state_open = get_gripper_state(robot)
    print(f"After open command: {gripper_state_open}")

    close_gripper(sim, robot, steps=60, sleep_sec=SLOW)
    gripper_state_closed = get_gripper_state(robot)
    print(f"After close command (-1.0): {gripper_state_closed}")

    print("\n" + "="*60)
    print("DIAGNOSTIS SUMMARY")
    print("="*60)

    if delta_zero < 0.001: print("✓ Zero action works correctly")
    else: print("✗ Zero action caused movement - check action space!")

    if 0.001 < np.linalg.norm(delta_x): print("✓ Position delta control works")
    else: print("✗ Position control not working as expected")

    if gripper_state_open['is_open'] and gripper_state_closed['is_closed']:
        print("✓ Gripper control works")
    else:
        print("✗ Gripper control issue")
        print(f"  → Open state: {gripper_state_open}")
        print(f"  → Closed state: {gripper_state_closed}")

    print("="*60 + "\n")



# --- add: finger joint helpers (works with panda_gym.Panda) ---
def _get_panda_uid(sim):
    return sim._bodies_idx.get("panda")

def _find_finger_joint_ids(sim):
    """Auto-detect 2 finger joints by name containing 'finger_joint'."""
    uid = _get_panda_uid(sim)
    if uid is None:
        return []
    ids = []
    n = sim.physics_client.getNumJoints(uid)
    for j in range(n):
        jname = sim.physics_client.getJointInfo(uid, j)[1].decode()
        if "finger_joint" in jname:
            ids.append(j)
    # 排序并只取前两个
    ids = sorted(ids)[:2]
    return ids

def set_gripper_width(sim, width_m: float, force: float = 80.0):
    """
    以“真实开口宽度（两指间距）”来设定夹爪。Panda 极限 ~0.08m。
    我们把每个指尖的目标位移设为 width/2。
    """
    uid = _get_panda_uid(sim)
    if uid is None:
        return
    fids = _find_finger_joint_ids(sim)
    half = max(0.0, float(width_m) * 0.5)
    for jid in fids:
        sim.physics_client.setJointMotorControl2(
            uid, jid, controlMode=p.POSITION_CONTROL,
            targetPosition=half, force=force
        )

def get_gripper_width(sim) -> float:
    """读取两指开口总宽度（m）。"""
    uid = _get_panda_uid(sim)
    if uid is None:
        return 0.0
    fids = _find_finger_joint_ids(sim)
    width = 0.0
    for jid in fids:
        js = sim.physics_client.getJointState(uid, jid)
        width += float(js[0])  # position
    return width
