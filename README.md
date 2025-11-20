# Group21-DRLPushGrasp
**Hierarchical Reinforcement Learning for Multi-attribute Object Manipulation**
Authored by: Gu Qibin (A0329840Y), Zhang Jiacheng (A0329995A), Goh Zheng Cong (A0332295A)

---

## Project Overview (Final report)

This repository contains the implementation for a thesis project on "Hierarchical Reinforcement Learning for Multi-attribute Object Manipulation." The system trains a Franka Emika Panda robot in simulation to clear a cluttered tabletop by intelligently switching between rule-based grasping for unoccluded objects and reinforcement learning (RL)-learned pushing to clear occlusions, using a hierarchical control architecture for sample-efficient learning.

-----

## Key Features

* Hierarchical action space: A 4D continuous action space ([`α_skill, α_x, α_y, α_θ`]) that unifies discrete skill selection (push/grasp) with continuous parameterization.
* Occlusion-aware reasoning: Explicit spatial representation in the observation space enables the policy to learn when to clear blockers before attempting to grasp a target.
* Stable PPO training: Custom implementation of Proximal Policy Optimization (PPO) with reward shaping, Generalized Advantage Estimation (GAE), and curriculum learning for robust policy convergence.
* Modular architecture: Decoupled motion primitives (`robot_util`), physics reasoning (`physics_util`), and object reasoning (`object_util`) for maintainability and extensibility.

---

## System Architecture

The system is built on a hierarchical RL framework:
* High-Level RL Policy: A PPO-based agent that outputs a 4D action vector for skill selection and parameterization
* Mid-Level Motion Primitives: Pre-programmed, robust controllers for `execute_pick_and_place` and `execute_push`
* Low-Level Control: PyBullet physics simulation with inverse kinematics and collision checking

# Observation Space
A structured 300+ dimensional vector containing:
* Robot state (joint positions, velocities, end-effector pose, gripper width)
* Object features (positions, velocities, shape descriptors, graspability scores)
* Spatial relationships (pairwise distance matrix, explicit occlusion masks)
* Goal context (goal position and size)

# Action Space
`α_skill`: Skill selection threshold (grasp if < 0, push otherwise)
`α_x`, `α_y`: Continuous local offsets from target object centroid (±2.5cm)
`α_θ`: Push direction in radians (mapped from [`-π, π`])

---

## Quick Start

# Prerequisites

* Python 3.8+
* PyBullet
* PyTorch
* Gymnasium
* NumPy

---
## Directory Structure

```text
Group21-DRLPushGrasp/
├── environment.yaml                # Conda env spec (Python 3.8 + pip pkgs)
├── README.md                       # This file
├── envs/
│   ├── __init__.py                 # Registers the custom Gym env
│   └── strategic_env.py            # Core environment (StrategicPushAndGraspEnv)
├── scripts/
│   ├── __init__.py                 # [CRITICAL] Makes 'scripts' a Python package
│   ├── ppo_scratch.py              # [CORE] Full implementation of PPO and ActorCritic NN
│   ├── demo_nn.py                  # [DEMO] Milestone 2 demo script (fwd/bwd pass)
│   └── test_custom_env.py          # [OPTIONAL] M1 environment smoke test script for gym
└── utils/
    ├── object_util.py              # Object-related utilities
    ├── physics_util.py             # Physics/collision-related utilities
    └── robot_util.py               # Robot action primitives


## ⚙️ Environment Setup

### Create Conda Environment
If you already have the `environment.yaml` file:

```Bash
# Create the Conda environment from the spec (installs Python 3.8 + pip pkgs)
conda env create -f environment.yaml

# Activate the environment (make sure the name matches the 'name:' in YAML)
conda activate me5418-demo
```



## Training the PPO Agent
R
un the main training script with the default configuration:

```Bash
python -m scripts.ppo_scratch
```

### Expected Output

You will see:

PyBullet Visualization: A simulation window opens showing the robot training on push-and-grasp tasks with procedurally generated cluttered scenes.

Terminal Training Logs: Live progress monitoring including:
* Episode returns and lengths
* PPO loss components (policy loss, value loss, entropy)
* Advantage statistics and learning rate updates
* KL divergence for early stopping detection

Training metrics

```Bash
[Episode   25] Steps=  2048 | Return=  45.20 | Length= 87 | Avg10=  32.15 | Time=00:00:45
[Update] Steps=  2048 | PolicyLoss=0.1245 | ValueLoss=0.0456 | Entropy=1.2345 | KL=0.0123 | EntCoef=0.0085 | LR=2.85e-4
```

Checkpointing: Model weights saved periodically to `checkpoints/` directory.


## Neural Network Validation Demo 

The project includes a comprehensive validation script to isolate and test the neural network architecture:

```Bash
# Run the network validation demo
python -m scripts.validate_network
```

### Expected Output

You will see:

# Expected Validation Output

1. Environment Initialization: PyBullet window opens with a single-object test scene

2. Forward Pass Demonstration:
* Episode execution with network-inferred actions
* Will be shown action vectors and rewards

3. Backward Pass Validation:
* REINFORCE loss calculation based on collected rewards
* Network weight comparison before/after optimization step
* `✓ SUCCESS: Parameter value changed!` confirmation message

This validates the complete training pipeline from state encoding to gradient updates.


###  Run Environment Smoke Test
If you wish to test the environment's stability with purely random actions (produces a lot of log spam), you can run:

```Bash
python -m scripts.test_custom_env
```

# 🚀 Appendix: Core Utilities & Action Primitives

## Object Utilities (`utils/object_util.py`)

Centralizes object-level reasoning for the Strategic Push-Grasp environment:
* Shape Descriptor Encoding: 8D feature vectors (object type, dimensions, volume, graspability)
* Spatial Analysis: Pairwise distance matrices and occlusion detection
* Scene Management: Safe object spawning and goal state checking
* Target Selection: Heuristic-based (nearest to goal) target prioritization

## Physics Utilities (`utils/physics_util.py`)

Robust wrappers around PyBullet's low-level API:
* Collision Detection: Robot-table, object-object, and self-collision checking
* Workspace Management: Boundary violation detection and object stability checks
* Contact Analysis: Force measurement and detailed contact point information
* Ray Casting: Line-of-sight checking for occlusion reasoning

## Robot Utilities (`utils/robot_util.py`)

High-level manipulation primitives and control abstractions:

* Motion Primitives: execute_pick_and_place and execute_push with kinematic feasibility checks
* Gripper Control: Synchronized finger control for grasping operations
* Inverse Kinematics: Safe joint trajectory planning via PyBullet IK solver
* Diagnostic Tools: Robot state monitoring and control validation

---

## 🚀 Core Action Primitives

```Bash
execute_pick_and_place(sim, robot, target_object, alpha_x, alpha_y, goal_pos, workspace_bounds, motion_scale=1.0) → bool
```

Eight-phase grasping pipeline:
1. Approach: Move to pre-grasp position above target
2. Descend: Lower to grasp height with gripper open
3. Close Gripper: Secure object with force closure
4. Lift Verification: Micro-lift with success validation (>1cm height gain)
5. Transport: Move to goal position above target zone
6. Place: Lower object to placement height
7. Release: Open gripper to deposit object
8. Retract: Return to safe height

```Bash
execute_push(sim, robot, target_object, alpha_x, alpha_y, alpha_theta, workspace_bounds, motion_scale=1.0) → bool
```
Three-phase pushing routine:
1. Pre-push Positioning: Approach with fixed orthogonal offset for torque
2. Push Execution: Linear motion along parameterized direction vector
3. Retract: Clear contact and return to neutral position

---

## Performance Characteristics
* Training Efficiency: ~200,000 environment steps for convergence (academic GPU feasible)
* Scene Complexity: 8-12 objects with guaranteed occlusion scenarios
* Success Metrics: >85% object clearance rate in novel configurations
* Emergent Behavior: Policy learns occlusion-clearing strategies without explicit programming

---

## Troubleshooting

# Common Issues
* PyBullet Window Not Opening: Ensure display is available or set render_mode='rgb_array'
* Training Instability: Adjust clip_eps (0.1-0.3) and learning rate (1e-4 to 3e-4)
* Collision Explosions: Verify object friction parameters and simulation timestep

---

## Citation
````
@article{gallouedec2021pandagym,
  title        = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
  author       = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
  year         = 2021,
  journal      = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
}


