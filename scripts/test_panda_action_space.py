"""
Diagnose Panda robot action space format
"""
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import envs
from utils.robot_util import get_ee_position_safe

env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
obs, info = env.reset()

robot = env.unwrapped.robot
sim = env.unwrapped.sim

print("="*60)
print("PANDA ROBOT ACTION SPACE DIAGNOSTIC")
print("="*60)

# Check robot class
print(f"\nRobot class: {type(robot)}")
print(f"Robot module: {type(robot).__module__}")

# Check if robot has these attributes
print(f"\nRobot attributes:")
print(f"  has get_ee_position: {hasattr(robot, 'get_ee_position')}")
print(f"  has set_action: {hasattr(robot, 'set_action')}")
print(f"  has control_type: {hasattr(robot, 'control_type')}")

if hasattr(robot, 'control_type'):
    print(f"  control_type value: {robot.control_type}")

# Get initial state
initial_pos = get_ee_position_safe(robot)
initial_obs = robot.get_obs()

print(f"\nInitial state:")
print(f"  EE position: {initial_pos}")
print(f"  Joint angles: {initial_obs[:7]}")
print(f"  Obs shape: {initial_obs.shape}")

# Test 1: Zero action (should not move)
print(f"\n{'='*60}")
print("TEST 1: Zero Action")
print("="*60)
print("Sending [0, 0, 0, 0] for 20 steps...")

for _ in range(20):
    robot.set_action(np.array([0.0, 0.0, 0.0, 0.0]))
    sim.step()

pos_after_zero = get_ee_position_safe(robot)
movement = np.linalg.norm(pos_after_zero - initial_pos)
print(f"Result: {pos_after_zero}")
print(f"Movement: {movement:.6f}m")
print(f"✓ PASS" if movement < 0.001 else "✗ FAIL - Robot moved!")

# Test 2: Small delta action
print(f"\n{'='*60}")
print("TEST 2: Small Delta Action")
print("="*60)
print("Sending [0.01, 0, 0, 0] for 20 steps...")
print("Expected: Move ~0.2m in +X direction")

for _ in range(20):
    robot.set_action(np.array([0.01, 0.0, 0.0, 0.0]))
    sim.step()

pos_after_delta = get_ee_position_safe(robot)
delta_movement = pos_after_delta - pos_after_zero
print(f"Result: {pos_after_delta}")
print(f"Delta: {delta_movement}")
print(f"Magnitude: {np.linalg.norm(delta_movement):.6f}m")

if 0.15 < np.linalg.norm(delta_movement) < 0.25:
    print("✓ PASS - Delta control works")
elif np.linalg.norm(delta_movement) < 0.01:
    print("✗ FAIL - Robot didn't move (action ignored?)")
else:
    print("⚠ UNCERTAIN - Unexpected movement")

# Test 3: Large delta action
print(f"\n{'='*60}")
print("TEST 3: Large Delta Action")
print("="*60)
print("Sending [0.05, 0, 0, 0] for 10 steps...")

for _ in range(10):
    robot.set_action(np.array([0.05, 0.0, 0.0, 0.0]))
    sim.step()

pos_after_large = get_ee_position_safe(robot)
large_delta = pos_after_large - pos_after_delta
print(f"Result: {pos_after_large}")
print(f"Delta: {large_delta}")
print(f"Magnitude: {np.linalg.norm(large_delta):.6f}m")

# Test 4: Check action space limits
print(f"\n{'='*60}")
print("TEST 4: Action Space Info")
print("="*60)
print(f"Action space: {env.action_space}")
print(f"Action space shape: {env.action_space.shape}")
print(f"Action space bounds: [{env.action_space.low}, {env.action_space.high}]")

# Test 5: Gripper control
print(f"\n{'='*60}")
print("TEST 5: Gripper Control")
print("="*60)

from utils.robot_util import get_gripper_state

initial_gripper = get_gripper_state(robot)
print(f"Initial gripper: {initial_gripper}")

print("Opening gripper (action[3] = 1.0)...")
for _ in range(30):
    robot.set_action(np.array([0.0, 0.0, 0.0, 1.0]))
    sim.step()

open_gripper = get_gripper_state(robot)
print(f"After open: {open_gripper}")
print(f"✓ PASS" if open_gripper['is_open'] else "✗ FAIL - Gripper didn't open")

print("Closing gripper (action[3] = -1.0)...")
for _ in range(30):
    robot.set_action(np.array([0.0, 0.0, 0.0, -1.0]))
    sim.step()

closed_gripper = get_gripper_state(robot)
print(f"After close: {closed_gripper}")
print(f"✓ PASS" if closed_gripper['is_closed'] else "✗ FAIL - Gripper didn't close")

print(f"\n{'='*60}")
print("SUMMARY")
print("="*60)
print("Based on these tests, the Panda robot:")
print("1. Action format: [?, ?, ?, gripper]")
print("2. Control type: ?")
print("3. Recommended fix: ?")
print("="*60)

env.close()
