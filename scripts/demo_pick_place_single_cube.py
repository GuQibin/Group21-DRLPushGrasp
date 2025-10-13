"""
Demo: Single-cube pick-and-place with Panda + PyBullet (corner placement).

Run:
    python -m scripts.demo_pick_place_single_cube
"""

import time
import numpy as np
import pybullet as p
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda import Panda

# -------------------------------
# Config: choose which table corner to place the robot
# "NE" | "NW" | "SE" | "SW"
# -------------------------------
CORNER = "SE"   # 改这里就能换角落

# 桌面参数（需与create_box一致）
TABLE_HALF_X = 0.4
TABLE_HALF_Y = 0.4
TABLE_Z_TOP  = 0.0  # 我们把桌面顶面放在 z=0（见 create_box 的位置）

# 机器人离桌边的外扩距离（往外挪一点，别与桌子重叠）
ROBOT_MARGIN = 0.05

# 物体/目标距离角落的“内缩”量（防止太贴边）
INSET = 0.3


# -------------------------------
# Utility helpers
# -------------------------------
def find_arm_and_gripper_joints(client, robot_uid):
    """Return (arm_joint_ids[7], finger_joint_ids[2], ee_link_index)."""
    arm_joint_ids = []
    finger_joint_ids = []
    ee_link_index = None

    n_j = client.getNumJoints(robot_uid)
    for jid in range(n_j):
        info = client.getJointInfo(robot_uid, jid)
        jname = info[1].decode()
        lname = info[12].decode()
        jtype = info[2]

        if jtype == p.JOINT_REVOLUTE and len(arm_joint_ids) < 7:
            arm_joint_ids.append(jid)

        if "hand_tcp" in lname or "hand" in lname:
            ee_link_index = jid

        if "finger_joint" in jname:
            finger_joint_ids.append(jid)

    if ee_link_index is None and n_j > 0:
        ee_link_index = n_j - 1

    return arm_joint_ids, finger_joint_ids[:2], ee_link_index


def set_gripper_opening(client, robot_uid, finger_joint_ids, opening_width=0.06, force=50.0):
    half = max(0.0, opening_width * 0.5)
    for jid in finger_joint_ids:
        client.setJointMotorControl2(
            robot_uid, jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=half,
            force=force
        )


def move_ee_ik(client, robot_uid, arm_joint_ids, ee_link_index,
               target_pos, target_quat,
               steps=120, sleep_per_step=1/120, max_force=200.0):
    cur_state = client.getLinkState(robot_uid, ee_link_index, computeForwardKinematics=1)
    cur_pos = np.array(cur_state[0])
    cur_quat = np.array(cur_state[1])

    for t in range(1, steps + 1):
        a = t / steps
        pos = (1 - a) * cur_pos + a * np.array(target_pos)
        quat = (1 - a) * cur_quat + a * np.array(target_quat)
        quat = quat / (np.linalg.norm(quat) + 1e-9)

        q_sol = client.calculateInverseKinematics(robot_uid, ee_link_index, pos, quat)
        targets = [q_sol[j] for j in range(len(arm_joint_ids))]

        client.setJointMotorControlArray(
            robot_uid, arm_joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=targets,
            forces=[max_force] * len(arm_joint_ids),
            positionGains=[0.08] * len(arm_joint_ids)
        )

        client.stepSimulation()
        time.sleep(sleep_per_step)


def corner_layout(corner: str, cube_half: float):
    """
    根据角落选择返回 (robot_base_pos, cube_pos, goal_pos).
    - robot_base_pos 放在桌面外一点点（ROBOT_MARGIN）
    - cube_pos 放在该角落内缩 INSET 的位置
    - goal_pos 放在对角线方向的另一侧，也内缩 INSET
    """
    # 角落符号到符号系数
    if corner.upper() == "NE":
        sx, sy = +1, +1
        opp_sx, opp_sy = -1, -1
    elif corner.upper() == "NW":
        sx, sy = -1, +1
        opp_sx, opp_sy = +1, -1
    elif corner.upper() == "SE":
        sx, sy = +1, -1
        opp_sx, opp_sy = -1, +1
    elif corner.upper() == "SW":
        sx, sy = -1, -1
        opp_sx, opp_sy = +1, +1
    else:
        raise ValueError("CORNER must be one of 'NE','NW','SE','SW'")

    # 机器人底座放在桌外（桌边再往外 ROBOT_MARGIN）
    base_x = sx * (TABLE_HALF_X + ROBOT_MARGIN)
    base_y = sy * (TABLE_HALF_Y + ROBOT_MARGIN)
    base_z = 0.0

    # 方块放在该角落的桌面上，向内缩 INSET
    cube_x = sx * (TABLE_HALF_X - INSET)
    cube_y = sy * (TABLE_HALF_Y - INSET)
    cube_z = cube_half  # 放在桌面上

    # 目标点放在对角线的另一侧，也向内缩 INSET
    goal_x = opp_sx * (TABLE_HALF_X - INSET)
    goal_y = opp_sy * (TABLE_HALF_Y - INSET)
    goal_z = cube_half

    return np.array([base_x, base_y, base_z]), np.array([cube_x, cube_y, cube_z]), np.array([goal_x, goal_y, goal_z])

def set_link_friction(client, body_uid, link_name_contains=("finger",), lateral=1.2, spinning=0.01, rolling=0.0):
    """提高夹爪与物体的摩擦，避免打滑。"""
    n = client.getNumJoints(body_uid)
    for jid in range(n):
        info = client.getJointInfo(body_uid, jid)
        lname = info[12].decode()
        if any(k in lname for k in link_name_contains):
            client.changeDynamics(body_uid, jid, lateralFriction=lateral,
                                  spinningFriction=spinning, rollingFriction=rolling)


def lower_until_touch(client, robot_uid, arm_joint_ids, ee_link_index, cube_uid,
                      start_pos, quat, dz_step=0.002, max_down=0.04, max_force=200.0, sleep=1/240):
    """
    从 start_pos 逐步向下搜索，直到手指与方块发生接触或达到最大下探距离。
    返回：触碰到时的末端位置（若未触碰则返回最终位置）。
    """
    z0 = start_pos[2]
    # 逐步下降
    travel = 0.0
    pos = np.array(start_pos, dtype=float)
    while travel < max_down:
        # IK 到当前 pos
        q_sol = client.calculateInverseKinematics(robot_uid, ee_link_index, pos, quat)
        targets = [q_sol[j] for j in range(len(arm_joint_ids))]
        client.setJointMotorControlArray(robot_uid, arm_joint_ids,
                                         controlMode=p.POSITION_CONTROL,
                                         targetPositions=targets,
                                         forces=[max_force]*len(arm_joint_ids),
                                         positionGains=[0.08]*len(arm_joint_ids))
        client.stepSimulation()
        time.sleep(sleep)

        # 检测与方块是否接触（手指任一 link）
        contact = client.getContactPoints(bodyA=robot_uid, bodyB=cube_uid)
        if contact:
            # 有接触，停止下探
            return pos

        # 向下迈一步
        pos = pos.copy()
        pos[2] -= dz_step
        travel += dz_step

    return pos  # 未触碰到也返回当前高度


def main():
    # -------------------------------
    # 1) Create sim, table, robot
    # -------------------------------
    sim = PyBullet(render_mode="human")  # GUI
    client = sim.physics_client
    client.setGravity(0, 0, -9.81)

    # 桌子：顶面 z=0
    sim.create_box(
        body_name="table",
        half_extents=np.array([TABLE_HALF_X, TABLE_HALF_Y, 0.01]),
        mass=0.0,
        position=np.array([0.0, 0.0, -0.01]),
        rgba_color=np.array([0.8, 0.8, 0.8, 1.0])
    )

    # 角落布局
    cube_half = 0.02  # 4 cm
    base_pos, cube_pos, goal = corner_layout(CORNER, cube_half)

    # 放机器人到角落
    robot = Panda(sim, block_gripper=False, base_position=base_pos)
    panda_uid = sim._bodies_idx.get("panda")
    arm_joints, finger_joints, ee_link = find_arm_and_gripper_joints(client, panda_uid)

    # ===== 关键：初始化到非奇点的“ready”姿态（单位：弧度）=====
    ready = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Franka常用home/ready
    for i, jid in enumerate(arm_joints):
        p.resetJointState(panda_uid, jid, float(ready[i]))
    p.setJointMotorControlArray(
        panda_uid, arm_joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=ready.tolist(),
        forces=[200.0]*len(arm_joints),
        positionGains=[0.08]*len(arm_joints)
    )

    # 初始就把夹爪张开一点（避免贴脸）
    set_gripper_opening(client, panda_uid, finger_joints, opening_width=0.075, force=120.0)

        # -------------------------------
    # 2) Spawn a single cube
    # -------------------------------
    sim.create_box(
        body_name="cube",
        half_extents=np.array([cube_half, cube_half, cube_half]),
        mass=0.3,
        position=cube_pos,
        rgba_color=np.array([0.2, 0.6, 0.9, 1.0])
    )
    cube_uid = sim._bodies_idx.get("cube")

    # 提高方块摩擦，避免被挤跑；也给指尖连杆加摩擦，避免打滑
    p.changeDynamics(cube_uid, -1, lateralFriction=1.0, spinningFriction=0.01, restitution=0.0)
    set_link_friction(client, panda_uid, ("finger",), lateral=1.2)

    # 可视化目标区域
    for i in range(4):
        a = i * np.pi / 2
        b = (i + 1) * np.pi / 2
        p1 = [goal[0] + 0.06 * np.cos(a), goal[1] + 0.06 * np.sin(a), 0.002]
        p2 = [goal[0] + 0.06 * np.cos(b), goal[1] + 0.06 * np.sin(b), 0.002]
        client.addUserDebugLine(p1, p2, [0, 1, 0], lineWidth=4)

    # -------------------------------
    # 3) 预姿态：张开夹爪 → 走到方块上方 → 下探到接触
    # -------------------------------
    # 先开大一些，并提高关节力（避免被物体反推）
    set_gripper_opening(client, panda_uid, finger_joints, opening_width=0.075, force=120.0)

    # 工具朝下
    down_quat = p.getQuaternionFromEuler([np.pi, 0, 0])

    hover = cube_pos + np.array([0.0, 0.0, 0.15])     # 上方 15 cm
    pregrasp = cube_pos + np.array([0.0, 0.0, 0.06])  # 顶面上方 ~4 cm

    print(f"[Corner={CORNER}] base={base_pos}, cube={cube_pos}, goal={goal}")

    print("→ Move above cube...")
    move_ee_ik(client, panda_uid, arm_joints, ee_link, hover, down_quat, steps=200)

    print("→ Move to pre-grasp...")
    move_ee_ik(client, panda_uid, arm_joints, ee_link, pregrasp, down_quat, steps=140)

    # 关键：缓慢向下直到手指触到方块（而不是在空中关爪）
    print("→ Lower until touch...")
    contact_pos = lower_until_touch(
        client, panda_uid, arm_joints, ee_link, cube_uid,
        start_pos=pregrasp, quat=down_quat,
        dz_step=0.002, max_down=0.05, max_force=200.0, sleep=1/240
    )

    # -------------------------------
    # 4) 闭合夹爪（更大力）并等待稳定若干帧
    # -------------------------------
    print("→ Close gripper (strong)...")
    set_gripper_opening(client, panda_uid, finger_joints, opening_width=0.0, force=180.0)
    for _ in range(120):  # 给接触求解时间
        client.stepSimulation(); time.sleep(1/240)

    # -------------------------------
    # 5) 抬起（慢一点，避免滑落）
    # -------------------------------
    lift = contact_pos + np.array([0.0, 0.0, 0.12])
    print("→ Lift...")
    move_ee_ik(client, panda_uid, arm_joints, ee_link, lift, down_quat, steps=180)

    # -------------------------------
    # 6) 移动到放置点并下降
    # -------------------------------
    place_hover = goal + np.array([0.0, 0.0, 0.12])
    place_touch = goal + np.array([0.0, 0.0, 0.02 + cube_half])

    print("→ Move to place hover...")
    move_ee_ik(client, panda_uid, arm_joints, ee_link, place_hover, down_quat, steps=200)

    print("→ Descend to place...")
    move_ee_ik(client, panda_uid, arm_joints, ee_link, place_touch, down_quat, steps=140)

    # -------------------------------
    # 7) 松爪并后撤
    # -------------------------------
    print("→ Open gripper and retreat...")
    set_gripper_opening(client, panda_uid, finger_joints, opening_width=0.075, force=120.0)
    for _ in range(60):
        client.stepSimulation(); time.sleep(1/240)

    move_ee_ik(client, panda_uid, arm_joints, ee_link, place_hover, down_quat, steps=140)
    print("✓ Done. Window will stay open for 2 seconds.")
    time.sleep(2)

    # -------------------------------
    # 8) Cleanup
    # -------------------------------
    sim.close()



if __name__ == "__main__":
    main()
