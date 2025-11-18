import numpy as np
from typing import Tuple, Optional
import pybullet as p
import time  # Added time import for sleep_sec

SLOW = 1 / 120


# SLOW = 0

# --- ADDED: Gripper control utilities (Required by open_gripper/close_gripper) ---
def set_gripper_width(sim, width: float):
    """
    Sets the target gripper opening width using POSITION_CONTROL.
    """
    try:
        panda_uid = sim._bodies_idx.get("panda")
        if panda_uid is None:
            return

        # NOTE: Joint indices 9 and 10 usually correspond to the fingers
        half_width = width / 2.0

        # Assuming standard Panda finger joint indices (9 and 10)
        sim.physics_client.setJointMotorControl2(
            panda_uid, 9, p.POSITION_CONTROL, targetPosition=half_width, force=200
        )
        sim.physics_client.setJointMotorControl2(
            panda_uid, 10, p.POSITION_CONTROL, targetPosition=half_width, force=200
        )

    except Exception as e:
        print(f"Warning: Failed to set gripper width: {e}")


def get_gripper_width(sim) -> float:
    """
    Reads the current gripper opening width (sum of finger positions).
    """
    try:
        panda_uid = sim._bodies_idx.get("panda")
        if panda_uid is None:
            return 0.0

        # Read current position of finger joints (index 9 and 10)
        joint_state_9 = sim.physics_client.getJointState(panda_uid, 9)
        joint_state_10 = sim.physics_client.getJointState(panda_uid, 10)

        # The total width is the sum of the positions (assuming 0 is closed)
        width = joint_state_9[0] + joint_state_10[0]
        return float(width)

    except Exception as e:
        return 0.0


# --- END ADDED GRIPPER UTILITIES ---


def get_ee_position_safe(robot) -> np.ndarray:
    """
    Safely retrieves the end-effector (EE) position of the robot.
    """
    try:
        # METHOD 1: Robot's built-in API (preferred)
        if hasattr(robot, 'get_ee_position'):
            return robot.get_ee_position()

        # METHOD 2: Direct PyBullet query (fallback)
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
    Safely retrieves the robot's end-effector orientation (quaternion).
    """
    try:
        if hasattr(robot, 'get_ee_orientation'):
            return robot.get_ee_orientation()

        if hasattr(robot, 'sim'):
            ee_link = getattr(robot, 'ee_link', 11)
            panda_uid = robot.sim._bodies_idx.get('panda')

            if panda_uid is not None:
                link_state = robot.sim.physics_client.getLinkState(
                    panda_uid, ee_link, computeForwardKinematics=1
                )
                return np.array(link_state[1], dtype=np.float32)

        print(f"Warning: Could not get EE orientation")
        return np.array([0, 0, 0, 1], dtype=np.float32)

    except Exception as e:
        print(f"Warning: Could not get EE orientation: {e}")
        return np.array([0, 0, 0, 1], dtype=np.float32)


def get_current_joint_positions(robot) -> Optional[np.ndarray]:
    """
    Safely retrieves the current joint positions of the robot's arm.
    """
    try:
        obs = robot.get_obs()
        if len(obs) >= 7:
            return obs[:7]
        else:
            print("  ⚠ Warning: Robot observation is too short to get joint positions.")
            return None
    except Exception as e:
        print(f"  ⚠ Error getting current joint positions: {e}")
        return None


# MODIFIED: Uses dedicated get_gripper_width helper
def get_gripper_state(robot) -> dict:
    """
    Retrieves the current gripper state using the dedicated helper function.
    """
    w = 0.0
    try:
        w = get_gripper_width(robot.sim)
    except Exception:
        pass
    return {
        "width": w,
        "is_closed": w < 0.01,
        "is_open": w > 0.07
    }


def execute_pick_and_place(sim, robot, target_object: str,
                           alpha_x: float, alpha_y: float,
                           goal_pos: np.ndarray,
                           workspace_bounds: Tuple[float, float, float, float],
                           approach_height: float = 0.15,
                           grasp_height: float = 0.03) -> bool:
    """
    Executes a complete pick-and-place routine.
    (Assumes alpha_x, alpha_y are 0.0 for rule-based center grasp)
    """
    try:
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except Exception as e:
        print(f"❌ Error: Could not get pose for {target_object}: {e}")
        return False

    initial_obj_z = obj_pos[2]

    # GRASP POINT CALCULATION (Fixed center grasp)
    offset_scale = 0.025
    grasp_offset = np.array([alpha_x * offset_scale, alpha_y * offset_scale, 0.0])

    rot_matrix = quaternion_to_rotation_matrix(obj_ori)
    grasp_offset_world = rot_matrix @ grasp_offset
    grasp_point = obj_pos + grasp_offset_world

    original_grasp_point_xy = grasp_point[:2].copy()
    grasp_point[0] = np.clip(grasp_point[0], workspace_bounds[0], workspace_bounds[1])
    grasp_point[1] = np.clip(grasp_point[1], workspace_bounds[2], workspace_bounds[3])

    if not np.allclose(original_grasp_point_xy, grasp_point[:2]):
        print(
            f"  [DEBUG] grasp_point clipped from ({original_grasp_point_xy[0]:.3f}, {original_grasp_point_xy[1]:.3f}) to ({grasp_point[0]:.3f}, {grasp_point[1]:.3f})")

    # PHASE 1: APPROACH
    # print(f"  Phase 1: Approaching {target_object} from above...")
    approach_pos = grasp_point.copy()
    approach_pos[2] = approach_height
    if compute_inverse_kinematics(sim, robot, approach_pos) is None:
        print(f"  ❌ Approach position is unreachable. Aborting grasp.")
        return False
    success = move_to_position(sim, robot, approach_pos, gripper_open=True, steps=150, sleep_sec=SLOW)
    if not success:
        # print(f"  ❌ Failed to approach {target_object}")
        return False

    # PHASE 2: DESCEND
    # print(f"  Phase 2: Lowering to grasp height...")
    grasp_pos = grasp_point.copy()
    grasp_pos[2] = grasp_height
    success = move_to_position(sim, robot, grasp_pos, gripper_open=True, steps=150, sleep_sec=SLOW)
    if not success:
        # print(f"  ❌ Failed to lower to grasp height for {target_object}")
        return False

    # PHASE 3: CLOSE GRIPPER
    # print(f"  Phase 3: Closing gripper...")
    close_gripper(sim, robot, steps=60, sleep_sec=SLOW)
    for _ in range(30): sim.step()

    # PHASE 4: MICRO-LIFT to validate grasp
    # print(f"  Phase 4: Micro-lift to check grasp...")
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
        # print(f"  ❌ Grasp failed (lift gain={lift_gain:.3f}m)")
        open_gripper(sim, robot, steps=60, sleep_sec=SLOW)
        return False
    # print(f"  ✓ Successfully grasped {target_object} (lift gain={lift_gain:.3f}m)")

    # PHASE 5: LIFT OBJECT
    lift_pos = get_ee_position_safe(robot)
    lift_pos[2] = approach_height
    move_to_position(sim, robot, lift_pos, gripper_open=False, steps=150, sleep_sec=SLOW)

    # PHASE 6: TRANSPORT TO GOAL
    transport_pos = np.array([goal_pos[0], goal_pos[1], approach_height])
    move_to_position(sim, robot, transport_pos, gripper_open=False, steps=150, sleep_sec=SLOW)

    # PHASE 7: PLACE OBJECT
    place_pos = np.array([goal_pos[0], goal_pos[1], 0.05])
    move_to_position(sim, robot, place_pos, gripper_open=False, steps=150, sleep_sec=SLOW)
    open_gripper(sim, robot, steps=60, sleep_sec=SLOW)

    # PHASE 8: RETRACT
    retract_pos = place_pos.copy()
    retract_pos[2] = approach_height
    move_to_position(sim, robot, retract_pos, gripper_open=True, steps=150, sleep_sec=SLOW)

    # print(f"  ✓ Pick-and-place complete!")
    return True


def execute_push(sim, robot, target_object: str,
                 alpha_x: float, alpha_y: float, alpha_theta: float,
                 workspace_bounds: Tuple[float, float, float, float],

                 push_distance: float = 0.035,
                 push_height: float = 0.02,
                 use_object_frame: bool = True) -> bool:
    """
    MODIFIED: Execute push primitive with FIXED ORTHOGONAL OFFSET (to ensure torque).
    (Ignores alpha_x, alpha_y input)
    """
    try:
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except Exception as e:
        print(f"Error: Could not get position for {target_object}: {e}")
        return False

    # 1. Calculate Push Direction Vector (V_push)
    push_angle = alpha_theta * np.pi
    V_push = np.array([np.cos(push_angle), np.sin(push_angle), 0.0])

    rot_matrix = quaternion_to_rotation_matrix(obj_ori)
    if use_object_frame:
        V_push = rot_matrix @ V_push

    # 2. Calculate Orthogonal Offset Vector (V_perp)
    FIXED_OFFSET_DISTANCE = 0.015  # 1.5 cm fixed offset for torque

    V_perp_raw = np.array([-V_push[1], V_push[0], 0.0])  # 90 deg rotation

    if np.linalg.norm(V_perp_raw) > 1e-6:
        V_perp = V_perp_raw / np.linalg.norm(V_perp_raw)
    else:
        V_perp = np.array([0.0, 0.0, 0.0])

    # 3. Calculate Final Contact Point (P_obj center + P_perp offset)
    P_offset = V_perp * FIXED_OFFSET_DISTANCE

    contact_point = obj_pos + P_offset
    contact_point[2] = push_height

    # CALCULATE PUSH TRAJECTORY POINTS
    pre_push_offset = 0.03

    try:
        body_id = sim._bodies_idx.get(target_object)
        if body_id is not None:
            shape_data = sim.physics_client.getCollisionShapeData(body_id, -1)
            if shape_data:
                geom_type = shape_data[0][2]
                dims = shape_data[0][3]

                if geom_type == p.GEOM_BOX:
                    half_x, half_y, _ = dims
                    dir_x, dir_y = V_push[0], V_push[1]
                    proj_radius = abs(dir_x) * half_x + abs(dir_y) * half_y
                    pre_push_offset = proj_radius + 0.01
    except Exception:
        pass

    pre_push_pos = contact_point - V_push * pre_push_offset
    post_push_pos = contact_point + V_push * push_distance

    if compute_inverse_kinematics(sim, robot, pre_push_pos) is None:
        print(f"  ❌ Pre-push position is unreachable. Aborting push.")
        return False

    # PHASE 1: MOVE TO PRE-PUSH POSITION
    # print(f"  Phase 1: Moving to pre-push position...")
    success = move_to_position(sim, robot, pre_push_pos, gripper_open=False, steps=300, sleep_sec=SLOW)
    if not success:
        # print(f"  ❌ Failed to reach pre-push position for {target_object}")
        return False

    # PHASE 2: EXECUTE PUSH
    # print(f"  Phase 2: Executing push...")
    success = move_to_position(sim, robot, post_push_pos, gripper_open=False, steps=300, sleep_sec=SLOW)
    if not success:
        # print(f"  ⚠ Push may have been incomplete")
        pass

    # PHASE 3: RETRACT
    # print(f"  Phase 3: Retracting...")
    retract_pos = post_push_pos.copy()
    retract_pos[2] += 0.1

    move_to_position(sim, robot, retract_pos, gripper_open=False, steps=150, sleep_sec=SLOW)

    open_gripper(sim, robot, steps=60, sleep_sec=SLOW)
    # print(f"  ✓ Push complete!")
    return True


def move_to_position(sim, robot, target_pos: np.ndarray,
                     gripper_open: bool = True,
                     steps: int = 50,
                     sleep_sec: float = 0.0) -> bool:
    """
    Moves the robot's end-effector smoothly toward a target 3D position using P-control.
    """
    initial_pos = get_ee_position_safe(robot)
    initial_distance = np.linalg.norm(target_pos - initial_pos)

    gripper_ctrl = 1.0 if gripper_open else -1.0

    for step in range(steps):
        try:
            current_pos = get_ee_position_safe(robot)
            error = target_pos - current_pos
            delta = np.clip(error * 5.0, -1.0, 1.0)
            action = np.concatenate([delta, [gripper_ctrl]])

            if step % 50 == 0 and step > 0:  # Reduced log frequency
                error_mag = np.linalg.norm(error)
                # print(f"  Step {step:2d}: error={error_mag:.4f}m, action={np.round(delta, 2)}")

            robot.set_action(action)
            sim.step()

            if sleep_sec:
                time.sleep(sleep_sec)

            if np.linalg.norm(error) < 0.01:
                # print(f"  ✓ Reached target at step {step}")
                return True

        except Exception as e:
            print(f"  ❌ Error at step {step}: {e}")
            return False

    final_pos = get_ee_position_safe(robot)
    final_error = np.linalg.norm(target_pos - final_pos)
    success = final_error < 0.035

    # print(f"  Final: pos={np.round(final_pos, 3)}, error={final_error:.4f}m")
    if not success:
        # print(f"  ⚠ Did not reach target (timeout after {steps} steps)")
        pass

    return success


def open_gripper(sim, robot, steps: int = 30, sleep_sec: float = 0.0):
    """Open the gripper to 7 cm."""
    target = 0.07  # 7cm
    set_gripper_width(sim, target)
    for _ in range(steps):
        sim.step()
        if sleep_sec: time.sleep(sleep_sec)


def close_gripper(sim, robot, steps: int = 30, sleep_sec: float = 0.0):
    """Close the gripper to 0 cm (hold by friction)."""
    set_gripper_width(sim, 0.0)
    for _ in range(steps):
        sim.step()
        if sleep_sec: time.sleep(sleep_sec)


def check_grasp_success(sim, robot, object_name: str,
                        initial_z: Optional[float] = None,
                        min_lift: float = 0.01) -> bool:
    """
    Checks whether a grasped object has been successfully lifted.
    """
    try:
        if initial_z is None:
            initial_z = sim.get_base_position(object_name)[2]

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
    Computes the inverse kinematics (IK) solution for the robot's end-effector.
    """
    try:
        if target_ori is None:
            target_ori = np.array([0, 1, 0, 0], dtype=np.float32)

        ee_link = getattr(robot, 'ee_link', 11)

        joint_state = sim.inverse_kinematics(
            body='panda',
            link=ee_link,
            position=target_pos,
            orientation=target_ori
        )
        return joint_state[:7]

    except Exception as e:
        # print(f"  ⚠ Error computing IK: {e}") # suppress log
        return None


def plan_trajectory(start_joints: np.ndarray, goal_joints: np.ndarray,
                    num_waypoints: int = 10) -> np.ndarray:
    """
    Generates a smooth linear joint-space trajectory between two configurations.
    """
    if start_joints is None or goal_joints is None:
        print(f"  ⚠ Error: Cannot plan trajectory with None as start or goal joints.")
        return np.array([])

    if start_joints.shape != goal_joints.shape:
        print(f"  ⚠ Error: Joint array shapes don't match: {start_joints.shape} vs {goal_joints.shape}")
        return np.array([])

    trajectory = np.linspace(start_joints, goal_joints, num_waypoints)
    return trajectory


def check_collision_between_bodies(sim, body1_name: str, body2_name: str) -> bool:
    """
    Checks whether two bodies are in contact in the PyBullet simulation.
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
    Converts a quaternion [x, y, z, w] into a 3×3 rotation matrix.
    """
    try:
        rot_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        return rot_matrix

    except Exception as e:
        print(f"  ⚠ Error converting quaternion: {e}")
        return np.eye(3)


def wait_for_stability(sim, object_name: str, max_steps: int = 50,
                       velocity_threshold: float = 0.01) -> bool:
    """
    Waits until a simulated object becomes stable (i.e., stops moving).
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
    Performs a basic diagnostic test on robot control functionality.
    """
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

    if delta_zero < 0.001:
        print("✓ Zero action works correctly")
    else:
        print("✗ Zero action caused movement - check action space!")

    if 0.001 < np.linalg.norm(delta_x):
        print("✓ Position delta control works")
    else:
        print("✗ Position control not working as expected")

    if gripper_state_open['is_open'] and gripper_state_closed['is_closed']:
        print("✓ Gripper control works")
    else:
        print("✗ Gripper control issue")
        print(f"  → Open state: {gripper_state_open}")
        print(f"  → Closed state: {gripper_state_closed}")

    print("=" * 60 + "\n")


def _get_panda_uid(sim):
    return sim._bodies_idx.get("panda")