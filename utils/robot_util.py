import numpy as np
from typing import Tuple, Optional
import pybullet as p
import time

# Set to 0 for training speed, or 1/240 for real-time visualization
SLOW = 0


def _scaled_steps(base_steps: int, motion_scale: float, min_steps: int = 15) -> int:
    """Zip controller step counts down when motion_scale < 1 (faster motion)."""
    # scale < 1.0 means faster (fewer steps)
    # scale > 1.0 means slower (more steps)
    # In previous logic, motion_scale=0.5 meant "faster/less steps"?
    # Or did it mean "50% speed"?
    # Usually for training we want motion_scale to reduce steps.
    # Let's assume motion_scale=0.5 means "half the time" (faster).
    scale = max(0.1, float(motion_scale))
    return max(min_steps, int(round(base_steps * scale)))


# --- GRIPPER UTILITIES ---
def set_gripper_width(sim, width: float):
    """Sets the target gripper opening width using POSITION_CONTROL."""
    try:
        panda_uid = sim._bodies_idx.get("panda")
        if panda_uid is None: return
        half_width = width / 2.0
        sim.physics_client.setJointMotorControl2(panda_uid, 9, p.POSITION_CONTROL, targetPosition=half_width, force=200)
        sim.physics_client.setJointMotorControl2(panda_uid, 10, p.POSITION_CONTROL, targetPosition=half_width,
                                                 force=200)
    except Exception as e:
        print(f"Warning: Failed to set gripper width: {e}")


def get_gripper_width(sim) -> float:
    """Reads the current gripper opening width."""
    try:
        panda_uid = sim._bodies_idx.get("panda")
        if panda_uid is None: return 0.0
        j9 = sim.physics_client.getJointState(panda_uid, 9)
        j10 = sim.physics_client.getJointState(panda_uid, 10)
        return float(j9[0] + j10[0])
    except Exception:
        return 0.0


def get_gripper_state(robot) -> dict:
    w = 0.0
    try:
        w = get_gripper_width(robot.sim)
    except:
        pass
    return {"width": w, "is_closed": w < 0.01, "is_open": w > 0.07}


# --- ROBOT STATE UTILITIES ---
def get_ee_position_safe(robot) -> np.ndarray:
    try:
        if hasattr(robot, 'get_ee_position'): return robot.get_ee_position()
        if hasattr(robot, 'sim'):
            ee_link = getattr(robot, 'ee_link', 11)
            panda_uid = robot.sim._bodies_idx.get('panda')
            if panda_uid is not None:
                ls = robot.sim.physics_client.getLinkState(panda_uid, ee_link, computeForwardKinematics=1)
                return np.array(ls[0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    except:
        return np.zeros(3, dtype=np.float32)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    try:
        return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    except:
        return np.eye(3)


# --- MOTION PRIMITIVES ---

def move_to_position(sim, robot, target_pos: np.ndarray,
                     gripper_open: bool = True,
                     steps: int = 50,
                     sleep_sec: float = 0.0) -> bool:
    """Moves EE to target_pos using simple P-control logic."""
    gripper_ctrl = 1.0 if gripper_open else -1.0
    for step in range(steps):
        try:
            current_pos = get_ee_position_safe(robot)
            error = target_pos - current_pos
            delta = np.clip(error * 5.0, -1.0, 1.0)
            action = np.concatenate([delta, [gripper_ctrl]])
            robot.set_action(action)
            sim.step()
            if sleep_sec > 0: time.sleep(sleep_sec)
            if np.linalg.norm(error) < 0.01: return True
        except:
            return False

    final_pos = get_ee_position_safe(robot)
    return np.linalg.norm(target_pos - final_pos) < 0.035


def open_gripper(sim, robot, steps: int = 30, sleep_sec: float = 0.0):
    set_gripper_width(sim, 0.08)
    for _ in range(steps):
        sim.step()
        if sleep_sec: time.sleep(sleep_sec)


def close_gripper(sim, robot, steps: int = 30, sleep_sec: float = 0.0):
    set_gripper_width(sim, 0.0)
    for _ in range(steps):
        sim.step()
        if sleep_sec: time.sleep(sleep_sec)


def compute_inverse_kinematics(sim, robot, target_pos: np.ndarray) -> Optional[np.ndarray]:
    try:
        ee_link = getattr(robot, 'ee_link', 11)
        js = sim.inverse_kinematics(body='panda', link=ee_link, position=target_pos, orientation=[0, 1, 0, 0])
        return js[:7]
    except:
        return None


# --- HIGH LEVEL ACTIONS ---

def execute_pick_and_place(sim, robot, target_object: str,
                           alpha_x: float, alpha_y: float,
                           goal_pos: np.ndarray,
                           workspace_bounds: Tuple[float, float, float, float],
                           approach_height: float = 0.15,
                           grasp_height: float = 0.03,
                           motion_scale: float = 1.0) -> bool:
    """
    Executes pick-and-place with motion scaling support.
    """
    try:
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except:
        return False

    initial_obj_z = obj_pos[2]

    # Grasp Point
    offset_scale = 0.025
    grasp_offset = np.array([alpha_x * offset_scale, alpha_y * offset_scale, 0.0])
    rot_matrix = quaternion_to_rotation_matrix(obj_ori)
    grasp_point = obj_pos + rot_matrix @ grasp_offset

    # Clip to bounds
    grasp_point[0] = np.clip(grasp_point[0], workspace_bounds[0], workspace_bounds[1])
    grasp_point[1] = np.clip(grasp_point[1], workspace_bounds[2], workspace_bounds[3])

    # 1. Approach
    approach_pos = grasp_point.copy();
    approach_pos[2] = approach_height
    if compute_inverse_kinematics(sim, robot, approach_pos) is None: return False

    steps_approach = _scaled_steps(100, motion_scale, 30)
    if not move_to_position(sim, robot, approach_pos, True, steps_approach, SLOW): return False

    # 2. Descend
    grasp_pos = grasp_point.copy();
    grasp_pos[2] = grasp_height
    steps_descend = _scaled_steps(100, motion_scale, 30)
    if not move_to_position(sim, robot, grasp_pos, True, steps_descend, SLOW): return False

    # 3. Close
    steps_grip = _scaled_steps(50, motion_scale, 15)
    close_gripper(sim, robot, steps_grip, SLOW)
    for _ in range(10): sim.step()  # Settle

    # 4. Lift
    lift_pos = approach_pos.copy()
    if not move_to_position(sim, robot, lift_pos, False, steps_approach, SLOW): return False

    # Check Success (Micro-lift check)
    try:
        curr_z = sim.get_base_position(target_object)[2]
    except:
        curr_z = initial_obj_z
    if curr_z - initial_obj_z < 0.01:
        open_gripper(sim, robot, steps_grip, SLOW)
        return False

    # 5. Transport
    transport_pos = np.array([goal_pos[0], goal_pos[1], approach_height])
    if not move_to_position(sim, robot, transport_pos, False, steps_approach, SLOW): return False

    # 6. Place
    place_pos = transport_pos.copy();
    place_pos[2] = 0.05
    move_to_position(sim, robot, place_pos, False, steps_descend, SLOW)
    open_gripper(sim, robot, steps_grip, SLOW)

    # 7. Retract
    move_to_position(sim, robot, transport_pos, True, steps_approach, SLOW)

    return True


def execute_push(sim, robot, target_object: str,
                 alpha_x: float, alpha_y: float, alpha_theta: float,
                 workspace_bounds: Tuple[float, float, float, float],
                 push_distance: float = 0.05,  # Increased slightly
                 push_height: float = 0.02,
                 use_object_frame: bool = True,
                 motion_scale: float = 1.0) -> bool:  # <--- Added motion_scale here
    """
    Executes push with motion scaling support.
    """
    try:
        obj_pos = np.array(sim.get_base_position(target_object))
        obj_ori = sim.get_base_orientation(target_object)
    except:
        return False

    # 1. Calculate Vectors
    push_angle = alpha_theta * np.pi
    V_push = np.array([np.cos(push_angle), np.sin(push_angle), 0.0])

    if use_object_frame:
        rot_matrix = quaternion_to_rotation_matrix(obj_ori)
        V_push = rot_matrix @ V_push

    # Orthogonal offset for torque
    V_perp = np.array([-V_push[1], V_push[0], 0.0])
    contact_point = obj_pos + (V_perp * 0.015)  # 1.5cm offset
    contact_point[2] = push_height

    # 2. Trajectory Points
    pre_push_offset = 0.04
    # Simple bounding box check could go here to adjust offset

    pre_push_pos = contact_point - V_push * pre_push_offset
    post_push_pos = contact_point + V_push * push_distance

    if compute_inverse_kinematics(sim, robot, pre_push_pos) is None: return False

    # 3. Execute
    steps_move = _scaled_steps(150, motion_scale, 40)
    steps_push = _scaled_steps(100, motion_scale, 30)

    # Move to Pre-push
    if not move_to_position(sim, robot, pre_push_pos, False, steps_move, SLOW): return False

    # Push
    success = move_to_position(sim, robot, post_push_pos, False, steps_push, SLOW)

    # Retract Up
    retract_pos = post_push_pos.copy();
    retract_pos[2] += 0.1
    move_to_position(sim, robot, retract_pos, False, steps_move, SLOW)

    return success