import numpy as np
from typing import Tuple, Optional
import pybullet as p

SLOW = 1/120
# SLOW = 0

def get_ee_position_safe(robot) -> np.ndarray:
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
    try:
        # ====================================================================
        # STEP 1: Get robot observation vector
        # ====================================================================
        obs = robot.get_obs()

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
                           workspace_bounds: Tuple[float, float, float, float], # <-- 1. 新增参数
                           approach_height: float = 0.15,
                           grasp_height: float = 0.03) -> bool:
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
    initial_obj_z = obj_pos[2]

    # ========================================================================
    # GRASP POINT CALCULATION
    # ========================================================================
    # Convert agent's normalized action parameters to physical offsets
    offset_scale = 0.025  # 2.5cm = reasonable for 4cm objects

    grasp_offset = np.array([
        alpha_x * offset_scale,  # X offset (left-right)
        alpha_y * offset_scale,  # Y offset (forward-back)
        0.0  # Z offset (always grasp from top)
    ])

    # Transform offset to world frame
    rot_matrix = quaternion_to_rotation_matrix(obj_ori)
    grasp_offset_world = rot_matrix @ grasp_offset
    grasp_point = obj_pos + grasp_offset_world

    # ========================================================================
    #  Clip the target grasp point to stay within workspace bounds
    # ========================================================================
    # workspace_bounds tuple structure is (x_min, x_max, y_min, y_max)
    original_grasp_point_xy = grasp_point[:2].copy()
    grasp_point[0] = np.clip(grasp_point[0], workspace_bounds[0], workspace_bounds[1])
    grasp_point[1] = np.clip(grasp_point[1], workspace_bounds[2], workspace_bounds[3])

    if not np.allclose(original_grasp_point_xy, grasp_point[:2]):
        print(
            f"  [DEBUG] grasp_point clipped from ({original_grasp_point_xy[0]:.3f}, {original_grasp_point_xy[1]:.3f}) to ({grasp_point[0]:.3f}, {grasp_point[1]:.3f})")
    # ========================================================================

    # ========================================================================
    # PHASE 1: APPROACH FROM ABOVE
    # ========================================================================
    print(f"  Phase 1: Approaching {target_object} from above...")
    approach_pos = grasp_point.copy()
    approach_pos[2] = approach_height
    if compute_inverse_kinematics(sim, robot, approach_pos) is None:
        print(f"  ❌ Approach position is unreachable. Aborting grasp.")
        return False
    success = move_to_position(sim, robot, approach_pos, gripper_open=True, steps=150, sleep_sec=SLOW)
    if not success:
        print(f"  ❌ Failed to approach {target_object}")
        return False

    # ========================================================================
    # PHASE 2: DESCEND TO GRASP HEIGHT
    # ========================================================================
    print(f"  Phase 2: Lowering to grasp height...")
    grasp_pos = grasp_point.copy()
    grasp_pos[2] = grasp_height
    success = move_to_position(sim, robot, grasp_pos, gripper_open=True, steps=150, sleep_sec=SLOW)
    if not success:
        print(f"  ❌ Failed to lower to grasp height for {target_object}")
        return False

    # ========================================================================
    # PHASE 3: CLOSE GRIPPER
    # ========================================================================
    print(f"  Phase 3: Closing gripper...")
    close_gripper(sim, robot, steps=60, sleep_sec=SLOW)
    for _ in range(30): sim.step()

    # ========================================================================
    # PHASE 4: MICRO-LIFT to validate grasp
    # ========================================================================
    print(f"  Phase 4: Micro-lift to check grasp...")
    try:
        obj_z_before = sim.get_base_position(target_object)[2]
    except Exception:
        obj_z_before = initial_obj_z
    micro_lift = get_ee_position_safe(robot).copy()
    micro_lift[2] += 0.03
    move_to_position(sim, robot, micro_lift, gripper_open=False, steps=60, sleep_sec=SLOW)
    for _ in range(20): sim.step()
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
    lift_pos = get_ee_position_safe(robot)
    lift_pos[2] = approach_height
    move_to_position(sim, robot, lift_pos, gripper_open=False, steps=150, sleep_sec=SLOW)

    # ========================================================================
    # PHASE 6: TRANSPORT TO GOAL
    # ========================================================================
    transport_pos = np.array([goal_pos[0], goal_pos[1], approach_height])
    move_to_position(sim, robot, transport_pos, gripper_open=False, steps=150, sleep_sec=SLOW)

    # ========================================================================
    # PHASE 7: PLACE OBJECT
    # ========================================================================
    place_pos = np.array([goal_pos[0], goal_pos[1], 0.05])
    move_to_position(sim, robot, place_pos, gripper_open=False, steps=150, sleep_sec=SLOW)
    open_gripper(sim, robot, steps=60, sleep_sec=SLOW)

    # ========================================================================
    # PHASE 8: RETRACT
    # ========================================================================
    retract_pos = place_pos.copy()
    retract_pos[2] = approach_height
    move_to_position(sim, robot, retract_pos, gripper_open=True, steps=150, sleep_sec=SLOW)

    print(f"  ✓ Pick-and-place complete!")
    return True


def execute_push(sim, robot, target_object: str,
                 alpha_x: float, alpha_y: float, alpha_theta: float,
                 workspace_bounds: Tuple[float, float, float, float],  # <-- 1. 新增参数
                 push_distance: float = 0.05,
                 push_height: float = 0.03,
                 use_object_frame: bool = True) -> bool:
    """
    Execute push primitive on target object.

    (函数顶部的详细注释保持不变)
    ...
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
    # CONTACT POINT and PUSH DIRECTION CALCULATION
    # ========================================================================
    offset_scale = 0.025
    contact_offset = np.array([alpha_x * offset_scale, alpha_y * offset_scale, 0.0])
    rot_matrix = quaternion_to_rotation_matrix(obj_ori)
    contact_offset_world = rot_matrix @ contact_offset
    contact_point = obj_pos + contact_offset_world
    contact_point[2] = push_height

    push_angle = alpha_theta * np.pi
    push_direction = np.array([np.cos(push_angle), np.sin(push_angle), 0.0])
    if use_object_frame:
        push_direction = rot_matrix @ push_direction

    # ========================================================================
    # CALCULATE PUSH TRAJECTORY POINTS
    # ========================================================================
    pre_push_offset = 0.03
    pre_push_pos = contact_point - push_direction * pre_push_offset
    post_push_pos = contact_point + push_direction * push_distance

    # ========================================================================
    # Clip the trajectory points to stay within workspace bounds
    # ========================================================================
    pre_push_pos[0] = np.clip(pre_push_pos[0], workspace_bounds[0], workspace_bounds[1])
    pre_push_pos[1] = np.clip(pre_push_pos[1], workspace_bounds[2], workspace_bounds[3])

    post_push_pos[0] = np.clip(post_push_pos[0], workspace_bounds[0], workspace_bounds[1])
    post_push_pos[1] = np.clip(post_push_pos[1], workspace_bounds[2], workspace_bounds[3])
    # ========================================================================
    if compute_inverse_kinematics(sim, robot, pre_push_pos) is None:
        print(f"  ❌ Pre-push position is unreachable. Aborting push.")
        return False
    # ========================================================================
    # PHASE 1: MOVE TO PRE-PUSH POSITION
    # ========================================================================
    print(f"  Phase 1: Moving to pre-push position...")
    success = move_to_position(sim, robot, pre_push_pos, gripper_open=False, steps=150, sleep_sec=SLOW)
    if not success:
        print(f"  ❌ Failed to reach pre-push position for {target_object}")
        return False

    # ========================================================================
    # PHASE 2: EXECUTE PUSH (Linear Motion)
    # ========================================================================
    print(f"  Phase 2: Executing push...")
    success = move_to_position(sim, robot, post_push_pos, gripper_open=False, steps=150, sleep_sec=SLOW)
    if not success:
        print(f"  ⚠ Push may have been incomplete")

    # ========================================================================
    # PHASE 3: RETRACT
    # ========================================================================
    print(f"  Phase 3: Retracting...")
    retract_pos = post_push_pos.copy()
    retract_pos[2] += 0.1
    move_to_position(sim, robot, retract_pos, gripper_open=True, steps=150, sleep_sec=SLOW)

    print(f"  ✓ Push complete!")
    return True

def move_to_position(sim, robot, target_pos: np.ndarray,
                    gripper_open: bool = True,
                    steps: int = 50,
                    sleep_sec: float = 0.0) -> bool:


    initial_pos = get_ee_position_safe(robot)
    initial_distance = np.linalg.norm(target_pos - initial_pos)

    gripper_ctrl = 1.0 if gripper_open else -1.0

    for step in range(steps):
        try:
            current_pos = get_ee_position_safe(robot)
            error = target_pos - current_pos
            delta = np.clip(error * 25.0, -1.0, 1.0)
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
    success = final_error < 0.015

    print(f"  Final: pos={np.round(final_pos, 3)}, error={final_error:.4f}m")
    if not success:
        print(f"  ⚠ Did not reach target (timeout after {steps} steps)")

    return success



def open_gripper(sim, robot, steps: int = 30, sleep_sec: float = 0.0):
    """Open the gripper to 7 cm, then step the simulator for a few iterations to let physics settle."""
    import time
    target = 0.07  # 7cm
    set_gripper_width(sim, target)
    for _ in range(steps):
        sim.step()
        if sleep_sec: time.sleep(sleep_sec)
    w = get_gripper_width(sim)

def close_gripper(sim, robot, steps: int = 30, sleep_sec: float = 0.0):
    """Close the gripper to 0 cm (hold by friction), then step for a few iterations to settle."""
    import time
    set_gripper_width(sim, 0.0)
    for _ in range(steps):
        sim.step()
        if sleep_sec: time.sleep(sleep_sec)
    w = get_gripper_width(sim)

def get_gripper_state(robot) -> dict:
    """
    No longer guessing indices from robot.get_obs(); read the joint directly instead.
    """
    w = 0.0
    try:
        # robot contains sim; read directly
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


    initial_pos = get_ee_position_safe(robot)
    initial_obs = robot.get_obs()


    for _ in range(10):
        robot.set_action(np.array([0.0, 0.0, 0.0, 0.0]))
        sim.step()

    pos_after_zero = get_ee_position_safe(robot)
    delta_zero = np.linalg.norm(pos_after_zero - initial_pos)

    for _ in range(steps):
        robot.set_action(np.array([0.01, 0.0, 0.0, 0.0]))
        sim.step()

    pos_after_x = get_ee_position_safe(robot)
    delta_x = pos_after_x - pos_after_zero

    gripper_state_initial = get_gripper_state(robot)


    open_gripper(sim, robot, steps=60, sleep_sec=SLOW)
    gripper_state_open = get_gripper_state(robot)


    close_gripper(sim, robot, steps=60, sleep_sec=SLOW)
    gripper_state_closed = get_gripper_state(robot)

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
    # Only get the first two
    ids = sorted(ids)[:2]
    return ids


def set_gripper_width(sim, width_m: float, force: float = 80.0):
    """
    Set the gripper based on the "actual opening width (two-finger spacing)". Panda limit ~0.08m.
We set the target displacement of each fingertip as width/2.
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
    """Set the gripper based on the "actual opening width (two-finger spacing)". Panda reads the total width of the two-finger opening at its limit.（m）。"""
    uid = _get_panda_uid(sim)
    if uid is None:
        return 0.0
    fids = _find_finger_joint_ids(sim)
    width = 0.0
    for jid in fids:
        js = sim.physics_client.getJointState(uid, jid)
        width += float(js[0])  # position
    return width
