"""
Demo script for Milestone 2: Minimal Training Loop (REINFORCE)

This script demonstrates a *complete, end-to-end* training process
for a single episode to show how the network's parameters are
updated based on real environment rewards.

It includes:
1.  Part 1: Data collection in a simple scene.
2.  Part 1.5: A summary of the network's inputs (obs) and outputs (actions AND values).
3.  Part 2: A demonstration of the backward pass and parameter updates.

Run from the project root directory:
python -m scripts.demo_nn
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import gymnasium as gym
from pathlib import Path
from typing import List

# --- 1. Set up path to import necessary modules ---
sys.path.append(str(Path(__file__).parent.parent))
try:
    import envs
    from scripts.ppo_scratch import ActorCritic
except ImportError as e:
    print(f"Error: Could not import necessary modules. (Error: {e})")
    print("Please ensure:")
    print("1. You are running this from the project root (Group21-DRLPushGrasp/).")
    print("2. 'ppo_scratch.py' is in the 'scripts/' directory.")
    print("3. You have an empty '__init__.py' file in the 'scripts/' directory.")
    sys.exit(1)

# --- 2. Define Network and Environment Parameters ---
OBS_DIM = 348
ACT_DIM = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 2
ACTIVATION = "tanh"
LEARNING_RATE = 3e-4
GAMMA = 0.99

print("=" * 70)
print("ME5418 Milestone 2: Neural Network Training Demo (Single Episode)")
print("=" * 70)

# --- 3. Create Network, Optimizer, and Environment ---
print("Initializing Network, Optimizer, and Environment...")
try:
    net = ActorCritic(OBS_DIM, ACT_DIM, HIDDEN_SIZE, NUM_LAYERS, ACTIVATION)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")
    print("✓ Network, Optimizer, and Environment instantiated successfully.\n")
except Exception as e:
    print(f"❌ FAILED to instantiate: {e}")
    sys.exit(1)

# --- 4. Run ONE complete training episode ---
print("-" * 70)
print("Part 1: Running ONE full episode in a SIMPLE scene...")
print("-" * 70)

# Storage for all data collected during the episode
episode_rewards: List[float] = []
episode_log_probs: List[torch.Tensor] = []
episode_action_vectors: List[np.ndarray] = []
episode_values: List[float] = [] # <-- [新增] 存储每一步的价值估计

# Use 'options' to reset to a simple, single-object scene
print("Resetting to a simple, single-object demo scene for clarity...")
demo_options = {
    "single_object_demo": True,
    "object_type": "cube",
    "object_pos": [0.1, 0.1]
}
obs, _ = env.reset(seed=42, options=demo_options)

terminated = False
truncated = False
episode_length = 0

while not (terminated or truncated):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    # [Forward Pass]
    # net.act() 返回 (a, logp, v, mu, std, raw) - 6个值
    # 【核心修改】: 显式捕获 'value'
    action_squashed, log_prob, value, *rest = net.act(obs_tensor)

    # [Execute Action]
    action_np = action_squashed.detach().numpy().flatten()
    next_obs, reward, terminated, truncated, info = env.step(action_np)

    # [Store Data]
    episode_rewards.append(reward)
    episode_log_probs.append(log_prob)
    episode_action_vectors.append(action_np)
    episode_values.append(value.item()) # <-- [新增] 存储这个价值

    obs = next_obs
    episode_length += 1

print(f"\n✓ Episode finished after {episode_length} steps.")
print(f"  Total Reward collected: {sum(episode_rewards):.2f}\n")
env.close() # Close the GUI window

# --- 5. (Episode Summary) ---
print("-" * 70)
print("Part 1.5: Episode Summary (What the Network Saw & Did)")
print("-" * 70)

print(f"  Network Input Summary:")
print(f"    Total Observations Processed: {episode_length}")
print(f"    Observation Vector Shape: ({OBS_DIM},)")

print(f"\n  Network Output Summary (Actions & Values):")
print(f"    Action Vector Shape: ({ACT_DIM},), Value is a scalar")
print(f"    --- Action Log (Untrained Network's Decisions) ---")

push_count = 0
grasp_count = 0
# 【核心修改】: 同时遍历 actions 和 values
for i, (action_vec, value) in enumerate(zip(episode_action_vectors, episode_values)):
    a_skill = action_vec[0]
    action_type = "Grasp" if a_skill > 0 else "Push"
    if action_type == "Grasp":
        grasp_count += 1
    else:
        push_count += 1

    # 【核心修改】: 打印语句中加入了 V(s) (价值)
    print(f"  Step {i+1:2d}: V(s)={value:7.3f} | Type={action_type:<5} | Raw Vector = [SKILL:{a_skill:6.2f}, X:{action_vec[1]:6.2f}, Y:{action_vec[2]:6.2f}, THETA:{action_vec[3]:6.2f}]")

print("    --------------------------------------------------")
print(f"    Total Grasp decisions: {grasp_count}")
print(f"    Total Push decisions:  {push_count}")
print("\n  This data will now be used to perform the backward pass...")


# --- 6. Perform a single training update (Backward Pass) ---
print("-" * 70)
print("Part 2: Performing ONE training update (Backward Pass)...")
print("-" * 70)

try:
    # 1. Calculate Discounted Returns (G_t)
    returns: List[float] = []
    discounted_return = 0.0
    for r in reversed(episode_rewards):
        discounted_return = r + GAMMA * discounted_return
        returns.insert(0, discounted_return)

    returns_tensor = torch.tensor(returns, dtype=torch.float32)

    if len(returns_tensor) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
    print(f"  1. Calculated {len(returns)} discounted returns and normalized them.")

    # 2. Calculate REINFORCE Policy Loss
    # (注意: 这个简单的REINFORCE演示没有训练Critic, PPO会训练)
    policy_loss = []
    for log_p, R in zip(episode_log_probs, returns_tensor):
        policy_loss.append(-log_p * R)

    policy_loss = torch.stack(policy_loss).sum()
    print(f"  2. Calculated total policy loss (based on actions in summary): {policy_loss.item():.4f}")

    # 3. Get the value of one network parameter *before* the update
    # 我们同时检查 Actor 和 Critic 的权重
    param_actor_before = net.mu_head.weight.data[0][0].item()
    param_critic_before = net.v_head.weight.data[0][0].item()
    print(f"  3. Sample Actor weight *before* update: {param_actor_before:.6f}")
    print(f"     Sample Critic weight *before* update: {param_critic_before:.6f} (Note: Critic is not trained in this demo)")


    # 4. [Backward Pass]: Clear old gradients, calculate new gradients
    optimizer.zero_grad()
    policy_loss.backward()
    print("  4. Backward pass executed (loss.backward()). Gradients are computed.")

    # 5. [Parameter Update]: Optimizer updates network weights
    optimizer.step()
    print("  5. Optimizer step executed (optimizer.step()). Parameters are updated.")

    # 6. Get the value of the *same* parameter *after* the update
    param_actor_after = net.mu_head.weight.data[0][0].item()
    param_critic_after = net.v_head.weight.data[0][0].item()
    print(f"  6. Sample Actor weight *after* update:  {param_actor_after:.6f}")
    print(f"     Sample Critic weight *after* update:  {param_critic_after:.6f}")

    # 7. Verification
    print("\n  --- VERIFICATION ---")
    if param_actor_before != param_actor_after:
        print(f"  ✓ SUCCESS (Actor): Parameter value changed! (Change: {param_actor_after - param_actor_before:.6e})")
        print("  This demonstrates a complete end-to-end training update for the Actor.")
    else:
        print(f"  ⚠ WARNING (Actor): Parameter value did not change.")

    if param_critic_before == param_critic_after:
        print("  ✓ SUCCESS (Critic): Parameter value unchanged, as expected (REINFORCE loss does not train Critic).")
    else:
        print(f"  ⚠ WARNING (Critic): Parameter value changed, which is unexpected.")


except Exception as e:
    print(f"❌ FAILED backward pass: {e}")
    sys.exit(1)

print("=" * 70)
print("✅ Minimal Training Loop Demo Passed!")
print("=" * 70)