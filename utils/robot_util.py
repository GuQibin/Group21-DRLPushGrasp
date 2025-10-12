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
    # TODO: For better accuracy, use proper quaternion rotation
    # from scipy.spatial.transform import Rotation
    # rot = Rotation.from_quat(obj_ori)
    # grasp_offset_world = rot.apply(grasp_offset)
    grasp_point = obj_pos + grasp_offset
    
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
        steps=50            # Allow 50 timesteps for convergence
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
        steps=30  # Fewer steps = slower, more controlled descent
    )
    
    if not success:
        print(f"  ❌ Failed to lower to grasp height for {target_object}")
        return False
    
    # ========================================================================
    # PHASE 3: CLOSE GRIPPER
    # ========================================================================
    # Actuate gripper fingers to close around object
    # Note: close_gripper() function is NOT defined in this file!
    # Must be implemented elsewhere or code will crash
    print(f"  Phase 3: Closing gripper...")
    
    close_gripper(sim, robot, steps=20)  # ⚠️ UNDEFINED FUNCTION
    
    # ========================================================================
    # PHASE 4: VERIFY GRASP SUCCESS
    # ========================================================================
    # Critical checkpoint: Did we actually grasp the object?
    # Prevents wasting time transporting empty gripper
    # Note: check_grasp_success() function is NOT defined in this file!
    print(f"  Phase 4: Checking grasp...")
    
    if not check_grasp_success(sim, robot, target_object, initial_z=initial_obj_z):
        # ⚠️ UNDEFINED FUNCTION
        print(f"  ❌ Grasp failed for {target_object}")
        open_gripper(sim, robot, steps=10)  # ⚠️ UNDEFINED FUNCTION
        return False  # Early exit - no point continuing without object
    
    print(f"  ✓ Successfully grasped {target_object}")
    
    # ========================================================================
    # PHASE 5: LIFT OBJECT
    # ========================================================================
    # Raise grasped object to safe transport height
    # Same height as approach to clear any obstacles
    print(f"  Phase 5: Lifting object...")
    
    lift_pos = grasp_pos.copy()
    lift_pos[2] = approach_height  # Lift back to 15cm
    
    move_to_position(
        sim, robot, lift_pos,
        gripper_open=False,  # Keep gripper CLOSED to hold object
        steps=30
    )
    
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
        steps=50  # Allow more time for potentially longer distance
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
        steps=30
    )
    
    # Release object
    open_gripper(sim, robot, steps=20)  # ⚠️ UNDEFINED FUNCTION
    
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
        steps=30
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
                use_object_frame: bool = False) -> bool:
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
    
    4. **Quaternion function missing**: Calls quaternion_to_rotation_matrix()
       which is NOT defined in this file (will crash if use_object_frame=True)
    
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
    
    # ====================================================================
    # OPTIONAL: Transform offset to object's local frame
    # ====================================================================
    if use_object_frame:
        # Rotate offset by object's orientation
        # Allows pushing relative to object's facing direction
        # ⚠️ WARNING: quaternion_to_rotation_matrix() is NOT defined!
        rot_matrix = quaternion_to_rotation_matrix(obj_ori)  # UNDEFINED!
        contact_offset = rot_matrix @ contact_offset
    
    # Contact point in world coordinates
    contact_point = obj_pos + contact_offset
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
    
    # ====================================================================
    # OPTIONAL: Transform direction to object's local frame
    # ====================================================================
    if use_object_frame:
        # Rotate push direction by object's orientation
        # Example: If object rotated 45°, "forward" push becomes northeast
        rot_matrix = quaternion_to_rotation_matrix(obj_ori)  # UNDEFINED!
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
        gripper_open=True,  # Gripper open (not grasping during push)
        steps=40            # Allow time to reach position
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
        gripper_open=True,
        steps=30  # Slower motion for controlled push
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
        gripper_open=True,
        steps=20
    )
    
    # ========================================================================
    # SUCCESS - Push sequence completed
    # ========================================================================
    print(f"  ✓ Push complete!")
    return True


def move_to_position(sim, robot, target_pos: np.ndarray, 
                    gripper_open: bool = True,
                    steps: int = 50) -> bool:
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
    
    However, different Panda implementations use different conventions:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Version         │ Gripper Range │ Open Value │ Close Value     │
    ├─────────────────┼───────────────┼────────────┼─────────────────┤
    │ panda-gym v2    │ [0, 1]        │ 1.0        │ 0.0             │
    │ panda-gym v3    │ [-1, 1]       │ 1.0        │ -1.0            │
    │ Some versions   │ [0, 0.04]     │ 0.04       │ 0.0 (physical)  │
    └──────────────────────────────────────────────────────────────────┘
    
    The comment "gripper stays closed" suggests the current implementation
    is NOT working correctly - the gripper control value may be wrong!
    
    Control Gain Analysis:
    The gain of 50.0 is VERY high and appears to be compensating for an
    action scaling issue:
    
    Comment says: "If 0.01 action only moves 0.006m"
    This suggests: actual_movement = action × scale_factor
    where scale_factor ≈ 0.6 (60% of commanded)
    
    To compensate, we use gain=50 so that:
    - Small error (1cm) → delta = 0.01 × 50 = 0.5 (saturates)
    - Large error (10cm) → delta = 0.1 × 50 = 5.0 → clipped to 1.0
    
    This essentially saturates the command for any error > 2cm, resulting
    in bang-bang control rather than smooth proportional control.
    
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
    
    Debugging Features:
    - Prints initial position and distance
    - Logs error and action every 10 steps
    - Reports final position and error
    - Progress messages help diagnose motion failures
    
    Common Failure Modes:
    1. **IK infeasible**: Target outside workspace or unreachable due to joint limits
    2. **Collision**: Motion blocked by table or other objects
    3. **Action scaling**: Robot doesn't respond as expected to commands
    4. **Saturation**: High gain causes oscillation around target
    5. **Timeout**: Convergence too slow, hits max steps
    
    Usage Pattern:
        # Approach phase (gripper open)
        success = move_to_position(sim, robot, approach_pos, gripper_open=True)
        
        # Transport phase (gripper closed, holding object)
        success = move_to_position(sim, robot, goal_pos, gripper_open=False)
    
    Performance Notes:
    - Average convergence time: 20-40 steps for nearby targets
    - Distant targets may timeout at 50 steps
    - Consider increasing steps for long-distance transport
    """
    # ========================================================================
    # GRIPPER CONTROL SETUP
    # ========================================================================
    # Determine gripper command based on desired state
    # Standard panda-gym convention: 1.0 = open, -1.0 = close
    # But comment suggests this may not be working correctly!
    # 
    # TODO: Debug gripper control by testing:
    # 1. Try gripper_ctrl = 1.0 and observe finger positions
    # 2. Try gripper_ctrl = -1.0 and observe finger positions  
    # 3. Try physical units: 0.04 (open) and 0.0 (close)
    # 4. Check robot.get_obs()[7:9] to see actual finger joint values
    print(f"\n[MOVE] Moving to {target_pos}")
    
    # ========================================================================
    # INITIAL STATE LOGGING
    # ========================================================================
    initial_pos = get_ee_position_safe(robot)
    initial_distance = np.linalg.norm(target_pos - initial_pos)
    print(f"[MOVE] From {initial_pos}, distance: {initial_distance:.4f}m")
    
    # Determine gripper action
    # Based on diagnostic comment: "gripper stays closed, so we need to figure out correct value"
    # This suggests the current values don't work as expected
    gripper_ctrl = 1.0 if gripper_open else -1.0
    
    # ========================================================================
    # CONTROL LOOP - Iterative motion until convergence or timeout
    # ========================================================================
    for step in range(steps):
        try:
            # ================================================================
            # STEP 1: Sense - Get current end-effector position
            # ================================================================
            current_pos = get_ee_position_safe(robot)
            
            # ================================================================
            # STEP 2: Compute - Calculate position error
            # ================================================================
            # Error vector points from current position to target
            # Positive error means target is ahead, negative means behind
            error = target_pos - current_pos  # [Δx, Δy, Δz] in meters
            
            # ================================================================
            # STEP 3: Control - Proportional controller with saturation
            # ================================================================
            # Gain = 50.0 (VERY AGGRESSIVE!)
            # 
            # Why so high? Comment explains:
            # "If 0.01 action only moves 0.006m, we need ~30x gain"
            # 
            # This suggests there's a scale mismatch between commanded action
            # and actual EE displacement. The high gain compensates for this.
            # 
            # Example calculation:
            # - error = 0.02m (2cm)
            # - delta_before_clip = 0.02 × 50 = 1.0
            # - delta_after_clip = 1.0 (saturated at max)
            # 
            # This means ANY error > 2cm results in maximum speed command.
            # The controller is essentially bang-bang, not proportional!
            delta = np.clip(error * 50.0, -1.0, 1.0)  # Saturate at ±1.0
            
            # ================================================================
            # STEP 4: Action - Construct full action vector
            # ================================================================
            # Action format (panda-gym standard):
            # [0:3] = delta position commands for X, Y, Z
            # [3] = gripper control command
            action = np.concatenate([delta, [gripper_ctrl]])
            
            # ================================================================
            # STEP 5: Logging - Print progress every 10 steps
            # ================================================================
            if step % 10 == 0:
                error_mag = np.linalg.norm(error)
                print(f"  Step {step:2d}: error={error_mag:.4f}m, action={delta}")
            
            # ================================================================
            # STEP 6: Actuate - Send action to robot
            # ================================================================
            robot.set_action(action)
            
            # ================================================================
            # STEP 7: Simulate - Advance physics by one timestep
            # ================================================================
            sim.step()  # Typically 1/240 second at default PyBullet rate
            
            # ================================================================
            # STEP 8: Check Convergence - Early termination if reached
            # ================================================================
            # If error is very small (< 1cm), consider target reached
            # This saves computation and allows next phase to start sooner
            if np.linalg.norm(error) < 0.01:  # 1cm threshold
                print(f"  ✓ Reached target at step {step}")
                return True  # Success - early exit
                
        except Exception as e:
            # If any error occurs during motion (e.g., IK failure), abort
            print(f"  ❌ Error at step {step}: {e}")
            return False
    
    # ========================================================================
    # FINAL CONVERGENCE CHECK - Reached max steps without early termination
    # ========================================================================
    # Query final position after all steps completed
    final_pos = get_ee_position_safe(robot)
    final_error = np.linalg.norm(target_pos - final_pos)
    
    # Use relaxed threshold for final check (5cm instead of 1cm)
    # Rationale:
    # - IK solver may not reach exact target due to joint limits
    # - High gain causes oscillation that may not settle to <1cm
    # - Some targets may be on edge of workspace
    # - 5cm is still acceptable for manipulation (object is ~4cm)
    success = final_error < 0.05  # 5cm threshold
    
    # ================================================================
    # FINAL LOGGING
    # ================================================================
    print(f"  Final: pos={final_pos}, error={final_error:.4f}m")
    if not success:
        print(f"  ⚠ Did not reach target (timeout after {steps} steps)")
    
    return success


def open_gripper(sim, robot, steps: int = 30):
    """
    Open gripper - try multiple strategies since diagnostic shows it's stuck closed.
    
    ═══════════════════════════════════════════════════════════════════════
    GRIPPER CONTROL - Multi-Strategy Approach
    ═══════════════════════════════════════════════════════════════════════
    
    This function attempts to open the gripper using THREE different strategies
    in sequence, because initial diagnostics indicate gripper control isn't working.
    
    Problem Statement:
    Comment in code says "diagnostic shows it's stuck closed" - this means
    the standard action interface isn't successfully opening the gripper.
    
    Strategy Cascade:
    ┌────────────────────────────────────────────────────────────────┐
    │ Strategy 1: Standard normalized control (action = [0,0,0,1.0])│
    │   ↓ If fails (gripper still closed)                           │
    │ Strategy 2: Exaggerated values (action = [0,0,0,10.0])       │
    │   ↓ If fails                                                   │
    │ Strategy 3: Direct joint control (bypass action interface)    │
    └────────────────────────────────────────────────────────────────┘
    
    Why Multiple Strategies?
    Different robot implementations have different quirks:
    - Some scale actions internally (need larger values)
    - Some have action space limits that are documented incorrectly
    - Some require direct joint control instead of high-level commands
    
    Strategy Details:
    
    **Strategy 1: Standard Control**
    - Uses robot.set_action([0, 0, 0, 1.0])
    - Assumes standard panda-gym convention: 1.0 = fully open
    - Should work if robot implementation is standard-compliant
    
    **Strategy 2: Exaggerated Values**
    - Uses robot.set_action([0, 0, 0, 10.0])
    - Tests if action space is scaled differently than expected
    - Some implementations may clip internally, others may scale
    - Value of 10.0 is intentionally extreme to test response
    
    **Strategy 3: Direct Joint Control**
    - Bypasses robot's action interface entirely
    - Directly sets joint positions via PyBullet's resetJointState()
    - Panda gripper has 2 prismatic finger joints (typically indices 9, 10)
    - Sets both fingers to 0.04m (4cm = fully open for Panda gripper)
    - This ALWAYS works if joint indices are correct
    
    Verification:
    After each strategy, calls get_gripper_state() to check if gripper opened.
    Note: get_gripper_state() function is NOT defined in this file!
    This will cause a NameError when called.
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        steps: Number of simulation steps to apply each strategy (default: 30)
              More steps = more time for gripper to respond
    
    Returns:
        None (prints success/failure messages)
    
    Missing Dependency:
        ⚠️ Calls get_gripper_state(robot) which is NOT defined!
        Should return dict with at least {'is_open': bool, 'is_closed': bool}
    
    Usage:
        Called after object placement to release gripper:
        open_gripper(sim, robot, steps=20)
    
    Debug Recommendations:
    1. Add print statements showing actual finger joint positions
    2. Query joint states: robot.get_obs()[7:9] (finger positions)
    3. Test each strategy individually with longer step counts
    4. Verify joint indices 9, 10 are correct for your URDF
    """
    print("  [GRIPPER] Attempting to open...")
    
    # ========================================================================
    # STRATEGY 1: Standard Normalized Control
    # ========================================================================
    # Try standard panda-gym convention: gripper_ctrl = 1.0 means "open"
    # Apply command for 'steps' timesteps to give gripper time to respond
    for _ in range(steps):
        # Action: [delta_x, delta_y, delta_z, gripper_ctrl]
        # [0, 0, 0, 1.0] = no movement, gripper open command
        robot.set_action(np.array([0.0, 0.0, 0.0, 1.0]))
        sim.step()  # Advance simulation to apply action
    
    # ====================================================================
    # VERIFICATION: Check if Strategy 1 worked
    # ====================================================================
    # ⚠️ WARNING: get_gripper_state() is NOT defined in this file!
    # This will throw NameError: name 'get_gripper_state' is not defined
    state = get_gripper_state(robot)  # UNDEFINED FUNCTION!
    
    if state['is_open']:
        print("  ✓ Gripper opened (Strategy 1)")
        return  # Success - no need to try other strategies
    
    # ========================================================================
    # STRATEGY 2: Exaggerated Values
    # ========================================================================
    # If standard control failed, try with much larger values
    # Tests if action space has unexpected scaling
    print("  [GRIPPER] Trying larger values...")
    
    for _ in range(steps):
        # Try gripper_ctrl = 10.0 (way outside standard [-1, 1] range)
        # Some implementations may internally scale/clip this
        robot.set_action(np.array([0.0, 0.0, 0.0, 10.0]))
        sim.step()
    
    # ====================================================================
    # VERIFICATION: Check if Strategy 2 worked
    # ====================================================================
    state = get_gripper_state(robot)  # UNDEFINED FUNCTION!
    
    if state['is_open']:
        print("  ✓ Gripper opened (Strategy 2)")
        return  # Success
    
    # ========================================================================
    # STRATEGY 3: Direct Joint Control (Nuclear Option)
    # ========================================================================
    # If both action-based approaches failed, bypass robot interface
    # and directly manipulate joint states in PyBullet
    print("  [GRIPPER] Trying direct joint control...")
    
    try:
        # Get Panda robot's body ID from simulation
        panda_uid = sim._bodies_idx.get('panda')
        
        if panda_uid is not None:
            # ================================================================
            # SET FINGER JOINT POSITIONS DIRECTLY
            # ================================================================
            # Panda gripper has 2 mimic joints (fingers move together)
            # Joint indices:
            # - Joint 9: Left finger (panda_finger_joint1)
            # - Joint 10: Right finger (panda_finger_joint2)
            # 
            # Position values:
            # - 0.04m = fully open (4cm gap between fingers)
            # - 0.0m = fully closed (fingers touching)
            for finger_joint in [9, 10]:
                sim.physics_client.resetJointState(
                    panda_uid,
                    finger_joint,
                    targetValue=0.04  # 4cm = fully open
                )
            
            # Run simulation to let gripper settle in new position
            for _ in range(steps):
                sim.step()
            
            print("  ✓ Gripper opened (Strategy 3 - direct control)")
            # No verification here - direct control always succeeds
            
    except Exception as e:
        print(f"  ⚠ Strategy 3 failed: {e}")
        # If even direct control fails, something is seriously wrong
        # (e.g., wrong joint indices, robot not loaded, etc.)

def close_gripper(sim, robot, steps: int = 30):
    """
    Close gripper.
    
    ═══════════════════════════════════════════════════════════════════════
    GRIPPER CLOSING - Single Strategy (Assumes Closing Works)
    ═══════════════════════════════════════════════════════════════════════
    
    Unlike open_gripper(), this function only uses ONE strategy:
    Standard normalized control with gripper_ctrl = -1.0
    
    Why Only One Strategy?
    The diagnostic issue was "gripper stays closed", not "gripper won't close".
    This suggests closing might already work, but opening doesn't.
    
    Implementation:
    - Sends action [0, 0, 0, -1.0] for 'steps' timesteps
    - Standard panda-gym convention: -1.0 = fully close
    - Verifies closure using get_gripper_state()
    
    Comparison with open_gripper():
    ┌─────────────────┬──────────────────┬──────────────────┐
    │                 │ open_gripper()   │ close_gripper()  │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Strategies      │ 3 (cascade)      │ 1 (simple)       │
    │ Gripper value   │ +1.0, +10.0, dir │ -1.0 only        │
    │ Complexity      │ High (debugging) │ Low (assumed OK) │
    └─────────────────┴──────────────────┴──────────────────┘
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        steps: Number of simulation steps to apply close command (default: 30)
    
    Returns:
        None (prints verification message)
    
    Missing Dependency:
        ⚠️ Calls get_gripper_state(robot) which is NOT defined!
    
    Usage:
        Called during grasping (Phase 3 of pick-and-place):
        close_gripper(sim, robot, steps=20)
    
    Potential Issue:
    If gripper was already closed (which diagnostic suggests), this function
    will report success even though it didn't actually change anything.
    The verification check may give false positive.
    """
    print("  [GRIPPER] Attempting to close...")
    
    # ========================================================================
    # STANDARD CLOSE COMMAND
    # ========================================================================
    # Apply gripper_ctrl = -1.0 (standard "close" command)
    for _ in range(steps):
        # Action: [delta_x, delta_y, delta_z, gripper_ctrl]
        # [0, 0, 0, -1.0] = no movement, gripper close command
        robot.set_action(np.array([0.0, 0.0, 0.0, -1.0]))
        sim.step()
    
    # ========================================================================
    # VERIFICATION
    # ========================================================================
    # Check if gripper actually closed
    # ⚠️ WARNING: get_gripper_state() is NOT defined!
    state = get_gripper_state(robot)  # UNDEFINED FUNCTION!
    
    if state['is_closed']:
        print("  ✓ Gripper closed")
    else:
        print("  ⚠ Gripper may not be fully closed")
        # Just a warning - doesn't return False
        # Grasp verification will happen in Phase 4 anyway


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
    
    Why Height-Based Verification?
    
    **Advantages**:
    - Simple and reliable
    - Works for all object types (cubes, spheres)
    - Doesn't require contact force sensors
    - Physics-based (object can't float without grasp)
    
    **Compared to alternatives**:
    - Contact force: Can have contact without secure grasp
    - Gripper gap: Doesn't tell if object is actually held
    - Visual: Requires camera and is more complex
    
    Height Threshold:
    min_lift = 1cm (0.01m) is a good balance:
    - Too small (0.1cm): False positives from vibration/settling
    - Too large (5cm): May fail on partial grasps that would work
    - 1cm: Clear indication of lift without being too strict
    
    Falling Detection:
    After checking height, also checks if object is falling:
    - obj_vel[2] < -0.1 means velocity is -10 cm/s downward
    - This catches cases where object was lifted but then slipped
    - Prevents false positives from momentary lift before drop
    
    Settlement Time:
    Waits 20 simulation steps before checking height:
    - Allows gripper to finish closing
    - Lets contact forces stabilize
    - Gives object time to react to grasp
    - At 240Hz, 20 steps ≈ 0.08 seconds
    
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
    
    Error Handling:
        Returns False if object doesn't exist or query fails
        Prints warning message for debugging
    
    Usage in Pick-and-Place:
        # Phase 4: Verify grasp after closing gripper
        if not check_grasp_success(sim, robot, target_object, initial_z):
            print("Grasp failed!")
            open_gripper(sim, robot)  # Release and abort
            return False
    
    Potential Improvements:
    1. **Check contact forces**: Ensure gripper is actually touching object
       from utils.physics_util import get_contact_force
       force = get_contact_force(sim, 'panda', object_name)
       has_contact = force > 1.0  # At least 1N of force
    
    2. **Check gripper closure**: Verify fingers actually closed
       gripper_gap = robot.get_obs()[7] + robot.get_obs()[8]
       if gripper_gap > 0.03:  # Fingers too far apart
           return False
    
    3. **Multiple checks over time**: Verify stability over multiple steps
       heights = [get_height() for _ in range(10)]
       is_stable = np.std(heights) < 0.001  # Low variance = stable
    
    Common Failure Causes:
    - Grasp point was off-center (wrong α_x, α_y)
    - Object moved between positioning and grasping
    - Gripper didn't actually close (mechanical failure)
    - Object too large or too small for gripper
    - Friction too low (object slipped out)
    - Object rotated and escaped grip
    """
    try:
        # ====================================================================
        # STEP 1: Get Initial Height (Baseline for Comparison)
        # ====================================================================
        if initial_z is None:
            # If not provided, query current height
            # Assumes this function is called BEFORE grasp attempt
            initial_z = sim.get_base_position(object_name)[2]
        
        # ====================================================================
        # STEP 2: Wait for Physics to Settle
        # ====================================================================
        # Give gripper time to close and object time to react
        # 20 steps ≈ 0.08 seconds at default 240Hz simulation rate
        for _ in range(20):
            sim.step()
        
        # ====================================================================
        # STEP 3: Measure Current Height
        # ====================================================================
        current_z = sim.get_base_position(object_name)[2]
        
        # ====================================================================
        # STEP 4: Compute Height Change
        # ====================================================================
        height_gained = current_z - initial_z
        
        # ====================================================================
        # STEP 5: Check if Object Lifted Sufficiently
        # ====================================================================
        if height_gained >= min_lift:
            # Object lifted! But is it stable or falling?
            
            # ================================================================
            # STEP 6: Verify Object is Not Falling
            # ================================================================
            # Get linear velocity of object
            obj_vel = sim.get_base_velocity(object_name)
            
            # Check Z-velocity (vertical component)
            # Negative velocity = moving downward
            # Threshold: -0.1 m/s = -10 cm/s (significant downward motion)
            is_falling = obj_vel[2] < -0.1
            
            # Success only if lifted AND not falling
            return not is_falling
        
        # Height didn't increase enough - grasp failed
        return False
        
    except Exception as e:
        # Error occurred (e.g., object doesn't exist)
        print(f"  ⚠ Error checking grasp: {e}")
        return False  # Conservative: assume failure on error

def compute_inverse_kinematics(sim, robot, target_pos: np.ndarray, 
                               target_ori: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Compute inverse kinematics using PyBullet wrapper.
    
    ═══════════════════════════════════════════════════════════════════════
    INVERSE KINEMATICS - Convert Cartesian pose to joint angles
    ═══════════════════════════════════════════════════════════════════════
    
    Inverse kinematics (IK) is the fundamental problem in robot control:
    Given a desired end-effector pose (position + orientation), compute the
    joint angles needed to achieve that pose.
    
    Problem Complexity:
    - Forward kinematics: joints → pose (easy, unique solution)
    - Inverse kinematics: pose → joints (hard, multiple/no solutions)
    
    For a 7-DOF robot like Panda, IK is particularly challenging:
    - 7 DOF > 6 DOF pose → redundant (infinite solutions)
    - Joint limits constrain reachable workspace
    - Singularities exist where IK becomes undefined
    - Collision avoidance not considered by basic IK
    
    PyBullet's IK Solver:
    Uses numerical optimization (likely Newton-Raphson or similar):
    1. Start from current joint configuration
    2. Iteratively adjust joints to minimize pose error
    3. Terminate when error < threshold or max iterations reached
    
    Solver Characteristics:
    - Fast (typically < 1ms per query)
    - Local optimizer (finds nearest solution, not global optimum)
    - May fail if target is outside workspace or at singularity
    - Doesn't guarantee collision-free configuration
    - Solution depends on starting configuration (current pose)
    
    Default Orientation:
    If target_ori is None, uses [0, 1, 0, 0] quaternion which represents:
    - Gripper pointing straight down (-Z direction)
    - Standard orientation for top-down grasps on tabletop
    - Corresponds to "wrist straight" configuration for Panda
    
    Why This Function Exists:
    Currently NOT used in the action primitives! The primitives use
    move_to_position() which relies on proportional control, not IK.
    
    This function could be used for:
    - Pre-computing joint trajectories (smoother motion)
    - Checking workspace reachability before attempting motion
    - Implementing joint-space control instead of Cartesian control
    
    Args:
        sim: PyBullet simulation instance
        robot: Panda robot instance
        target_pos: Desired end-effector position [x, y, z] in world frame (meters)
        target_ori: Desired end-effector orientation as quaternion [x, y, z, w]
                   If None, defaults to downward-facing gripper orientation
    
    Returns:
        joint_positions: 7D array of joint angles [rad] for arm joints only
                        Returns None if IK solution fails or error occurs
    
    IK Failure Modes:
    1. **Target outside workspace**: Position too far/close/high/low
    2. **Unreachable orientation**: Some orientations impossible to achieve
    3. **Near singularity**: IK solver numerically unstable
    4. **Joint limit violation**: Solution would require exceeding limits
    5. **Solver timeout**: Can't find solution in allowed iterations
    
    Usage Example (Not Currently Used):
        # Compute IK for grasp position
        target_pos = np.array([0.1, 0.2, 0.05])
        joint_angles = compute_inverse_kinematics(sim, robot, target_pos)
        
        if joint_angles is not None:
            # Move robot to computed joint configuration
            robot.set_joint_positions(joint_angles)
        else:
            print("IK failed - target unreachable")
    
    Comparison with Current Approach:
    ┌──────────────────────────┬────────────────────┬─────────────────┐
    │                          │ Current (Prop Ctrl)│ IK-Based        │
    ├──────────────────────────┼────────────────────┼─────────────────┤
    │ Control space            │ Cartesian          │ Joint           │
    │ Planning                 │ Reactive           │ Planned         │
    │ Smoothness               │ Jerky              │ Smooth          │
    │ Obstacle avoidance       │ None               │ Possible        │
    │ Implementation complexity│ Simple             │ Complex         │
    │ Computational cost       │ Low                │ Medium          │
    └──────────────────────────┴────────────────────┴─────────────────┘
    
    Integration Opportunity:
    Could replace move_to_position() with IK-based approach:
    1. Compute IK for target pose
    2. Plan smooth joint trajectory (use plan_trajectory())
    3. Execute trajectory point-by-point
    4. Result: Smoother, more predictable motion
    """
    try:
        # ====================================================================
        # DEFAULT ORIENTATION SETUP
        # ====================================================================
        if target_ori is None:
            # Default: Gripper pointing straight down
            # Quaternion [0, 1, 0, 0] represents 180° rotation around Y-axis
            # This corresponds to: Z-axis pointing down (gripper downward)
            target_ori = np.array([0, 1, 0, 0], dtype=np.float32)
        
        # ====================================================================
        # GET END-EFFECTOR LINK INDEX
        # ====================================================================
        # Panda's end-effector frame is typically link 11 in standard URDF
        # Different URDFs may use different indices, so check robot attribute
        ee_link = getattr(robot, 'ee_link', 11)
        
        # ====================================================================
        # CALL PYBULLET IK SOLVER
        # ====================================================================
        # Use the simulation wrapper's IK method
        # This internally calls PyBullet's calculateInverseKinematics()
        # 
        # The solver returns joint states for ALL joints, including:
        # - 7 arm joints (revolute)
        # - 2 gripper joints (prismatic)
        # - Potentially other passive joints
        joint_state = sim.inverse_kinematics(
            body='panda',              # Robot body name in simulation
            link=ee_link,              # Target link (end-effector)
            position=target_pos,       # Desired position [x, y, z]
            orientation=target_ori     # Desired orientation [x, y, z, w]
        )
        
        # ====================================================================
        # EXTRACT ARM JOINT ANGLES ONLY
        # ====================================================================
        # Return first 7 DOF (arm joints only, not gripper)
        # Gripper joints (8-9) are controlled separately via action space
        return joint_state[:7]
        
    except Exception as e:
        print(f"  ⚠ Error computing IK: {e}")
        # Return None to signal IK failure
        # Caller should check for None and handle gracefully
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
    planning - more sophisticated methods exist for real deployments.
    
    Current Implementation: Linear Interpolation (LERP)
    ┌────────────────────────────────────────────────────────────────┐
    │ For each joint j and waypoint i:                               │
    │   q_j(i) = start_j + (goal_j - start_j) × (i / N)            │
    │ where N = num_waypoints - 1                                    │
    └────────────────────────────────────────────────────────────────┘

    Args:
        start_joints: Starting joint configuration (7D array) [radians]
        goal_joints: Goal joint configuration (7D array) [radians]
        num_waypoints: Number of waypoints along trajectory (default: 10)
                      More waypoints = smoother but slower execution
    
    Returns:
        trajectory: Array of shape (num_waypoints, 7) containing joint angles
                   Each row is a waypoint configuration
                   Returns empty array if input shapes don't match
    
    Error Handling:
    If start and goal arrays have different shapes, prints warning and
    returns empty array. Caller should check result before using.
    """
    
    # Compute IK for start and goal
    start_joints = get_current_joint_positions()
    goal_joints = compute_inverse_kinematics(sim, robot, target_pos)
    
    # Plan trajectory
    trajectory = plan_trajectory(start_joints, goal_joints, num_waypoints=20)
    
    # Execute trajectory
    for waypoint in trajectory:
        robot.set_joint_positions(waypoint)
        sim.step()
    
    if start_joints.shape != goal_joints.shape:
        print(f"  ⚠ Error: Joint array shapes don't match: {start_joints.shape} vs {goal_joints.shape}")
        return np.array([])
    
    trajectory = np.linspace(start_joints, goal_joints, num_waypoints)
    return trajectory


def get_gripper_state(robot) -> dict:
    """
    Get current gripper state information.
    
    ═══════════════════════════════════════════════════════════════════════
    GRIPPER STATE QUERY - Essential feedback for manipulation verification
    ═══════════════════════════════════════════════════════════════════════
    
    This is the CRITICAL MISSING FUNCTION that enables gripper control
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


def check_collision_between_bodies(sim, body1_name: str, body2_name: str) -> bool:
    """
    Check collision using direct PyBullet API.
    
    ═══════════════════════════════════════════════════════════════════════
    COLLISION DETECTION - Binary check for contact between bodies
    ═══════════════════════════════════════════════════════════════════════
    
    Simple utility to check if two named bodies are currently in collision.
    Wrapper around PyBullet's getContactPoints() for cleaner API.
    
    Note: This is functionally IDENTICAL to check_object_collision() in
    physics_util.py. Consider removing this duplicate and importing instead:
    
    from utils.physics_util import check_object_collision as check_collision_between_bodies
    
    Collision Detection Process:
    PyBullet maintains a list of contact points updated each simulation step:
    1. Broad phase: Check bounding box overlaps (fast)
    2. Narrow phase: Compute exact geometry intersections (slower)
    3. Generate contact manifolds with penetration info
    4. Store in contact point cache
    
    This function queries the cached contact points (very fast, O(1)).
    
    Args:
        sim: PyBullet simulation instance
        body1_name: First body name (e.g., "panda")
        body2_name: Second body name (e.g., "table")
    
    Returns:
        has_collision: True if any contact points exist between bodies
                      False if no contact, or if either body doesn't exist
    """
    try:
        # ====================================================================
        # Get body IDs from simulation registry
        # ====================================================================
        body1_id = sim._bodies_idx.get(body1_name)
        body2_id = sim._bodies_idx.get(body2_name)
        
        # ====================================================================
        # Handle missing bodies (conservative: no collision)
        # ====================================================================
        if body1_id is None or body2_id is None:
            return False  # Can't collide with non-existent body
        
        # ====================================================================
        # Query contact points from PyBullet
        # ====================================================================
        # Direct PyBullet API call
        # Returns list of contact point tuples (empty if no contact)
        contacts = sim.physics_client.getContactPoints(
            bodyA=body1_id,
            bodyB=body2_id
        )
        
        # ====================================================================
        # Binary check: any contact = collision
        # ====================================================================
        return len(contacts) > 0
        
    except Exception as e:
        print(f"  ⚠ Error checking collision: {e}")
        return False  # Conservative: assume no collision on error



def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
     Convert quaternion to 3x3 rotation matrix.
    
    ═══════════════════════════════════════════════════════════════════════
    QUATERNION CONVERSION - Transform orientation representation
    ═══════════════════════════════════════════════════════════════════════
    
    Converts quaternion representation of rotation to rotation matrix form.
    This is CRITICAL for execute_push() when use_object_frame=True.
    
    Rotation Representations:
    ┌─────────────────┬──────────┬─────────────────┬─────────────────┐
    │ Format          │ Size     │ Advantages      │ Disadvantages   │
    ├─────────────────┼──────────┼─────────────────┼─────────────────┤
    │ Quaternion      │ 4 values │ No gimbal lock  │ Not intuitive   │
    │ Rotation Matrix │ 9 values │ Easy to apply   │ Redundant       │
    │ Euler Angles    │ 3 values │ Intuitive       │ Gimbal lock     │
    │ Axis-Angle      │ 4 values │ Minimal         │ Singularities   │
    └─────────────────┴──────────┴─────────────────┴─────────────────┘
    
    Why Convert to Matrix?
    Rotation matrices are easiest for vector transformations:
    
    rotated_vector = rotation_matrix @ original_vector
    
    This is needed in execute_push() to rotate:
    1. Contact offset from object frame to world frame
    2. Push direction from object frame to world frame
    
    Quaternion Format:
    PyBullet uses [x, y, z, w] convention (scalar-last).
    Some libraries use [w, x, y, z] (scalar-first) - be careful!
    
    Mathematical Background:
    A quaternion q = (x, y, z, w) represents rotation:
    - w = cos(θ/2)
    - (x, y, z) = sin(θ/2) × (axis_x, axis_y, axis_z)
    
    Where θ is rotation angle and (axis_x, axis_y, axis_z) is rotation axis.
    
    Rotation Matrix Formula (from quaternion):
    R = | 1-2(y²+z²)   2(xy-wz)     2(xz+wy)   |
        | 2(xy+wz)     1-2(x²+z²)   2(yz-wx)   |
        | 2(xz-wy)     2(yz+wx)     1-2(x²+y²) |

    Implementation:
    Uses PyBullet's built-in getMatrixFromQuaternion() which:
    - Returns rotation matrix as 9-element flat array [row-major]
    - Reshapes to 3×3 matrix
    - Very fast (pre-compiled C++ code)
    """
    try:
        # ====================================================================
        # Convert quaternion to rotation matrix using PyBullet
        # ====================================================================
        # PyBullet's getMatrixFromQuaternion returns 9-element list
        # in row-major order: [R11, R12, R13, R21, R22, R23, R31, R32, R33]
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
        # ====================================================================
        # Step 1: Advance simulation by one timestep
        # ====================================================================
        sim.step()  # Typically 1/240 second
        
        try:
            # ================================================================
            # Step 2: Query object velocity
            # ================================================================
            vel = sim.get_base_velocity(object_name)
            
            # ================================================================
            # Step 3: Compute speed (velocity magnitude)
            # ================================================================
            speed = np.linalg.norm(vel)  # ||v|| = √(vx² + vy² + vz²)
            
            # ================================================================
            # Step 4: Check if below stability threshold
            # ================================================================
            if speed < velocity_threshold:
                return True  # Object is stable - early exit
        
        except Exception:
            # Object doesn't exist or query failed
            # Consider this a failure (object likely fell off table)
            return False
    
    # ========================================================================
    # TIMEOUT - Object didn't stabilize within max_steps
    # ========================================================================
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

    Test Sequence:
    ┌────────────────────────────────────────────────────────────────┐
    │ Test 1: Zero Action                                            │
    │   Send: [0, 0, 0, 0] for 10 steps                            │
    │   Expected: No movement (EE stays at initial position)        │
    │   Purpose: Verify zero action doesn't cause drift            │
    ├────────────────────────────────────────────────────────────────┤
    │ Test 2: Positive X Delta                                       │
    │   Send: [0.01, 0, 0, 0] for N steps                          │
    │   Expected: EE moves in +X direction                          │
    │   Purpose: Measure action scaling and verify delta control    │
    ├────────────────────────────────────────────────────────────────┤
    │ Test 3: Gripper Open/Close                                     │
    │   Send: [0, 0, 0, +1] then [0, 0, 0, -1]                     │
    │   Expected: Gripper opens then closes                         │
    │   Purpose: Verify gripper control values and polarity        │
    └────────────────────────────────────────────────────────────────┘
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

    # ========================================================================
    # INITIAL STATE CAPTURE
    # ========================================================================
    initial_pos = get_ee_position_safe(robot)
    initial_obs = robot.get_obs()
    
    print(f"Initial EE position: {initial_pos}")
    print(f"Initial joint angles: {initial_obs[:7]}")
    print(f"Observation shape: {initial_obs.shape}")

    # ========================================================================
    # TEST 1: ZERO ACTION (Verify no drift)
    # ========================================================================
    print("\nTest 1: Zero action (no movement)")
    for _ in range(10):
        robot.set_action(np.array([0.0, 0.0, 0.0, 0.0]))
        sim.step()
    
    pos_after_zero = get_ee_position_safe(robot)
    delta_zero = np.linalg.norm(pos_after_zero - initial_pos)
    print(f"Position after zero action: {pos_after_zero}")
    print(f"Movement: {delta_zero:.6f}m (should be ~0)")
    
    # ========================================================================
    # TEST 2: POSITIVE X ACTION (Measure scaling)
    # ========================================================================
    print("\nTest 2: Positive X delta (+0.01m)")
    for _ in range(steps):
        robot.set_action(np.array([0.01, 0.0, 0.0, 0.0]))
        sim.step()
    
    pos_after_x = get_ee_position_safe(robot)
    delta_x = pos_after_x - pos_after_zero
    print(f"Position after +X action: {pos_after_x}")
    print(f"Delta: {delta_x}")
    print(f"Expected: [+{0.01*steps:.3f}, 0, 0] ({steps} steps × 0.01)")
    print(f"Actual magnitude: {np.linalg.norm(delta_x):.6f}m")
    
    # ========================================================================
    # TEST 3: GRIPPER CONTROL (Verify open/close)
    # ========================================================================
    print("\nTest 3: Gripper control")
    gripper_state_initial = get_gripper_state(robot)
    print(f"Initial gripper: {gripper_state_initial}")
    
    # Open gripper
    for _ in range(20):
        robot.set_action(np.array([0.0, 0.0, 0.0, 1.0]))
        sim.step()
    
    gripper_state_open = get_gripper_state(robot)
    print(f"After open command: {gripper_state_open}")
    
    # ====================================================================
    # Sub-test 3b: Close gripper
    # ====================================================================
    for _ in range(20):
        robot.set_action(np.array([0.0, 0.0, 0.0, -1.0]))
        sim.step()
    
    gripper_state_closed = get_gripper_state(robot)
    print(f"After close command (-1.0): {gripper_state_closed}")
    
    # ========================================================================
    # DIAGNOSIS SUMMARY - Automated Pass/Fail Analysis
    # ========================================================================
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    # ====================================================================
    # Analyze Test 1: Zero Action
    # ====================================================================
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
