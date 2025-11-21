# Group21-DRLPushGrasp

**Multi-Primitive Robotic Manipulation via Proximal Policy Optimization (PPO)-Directed Push: Occlusion-Aware Target Selection in Cluttered Table Top Environments**

Authored by: 
* Gu Qibin (A0329840Y),
* Zhang Jiacheng (A0329995A),
* Goh Zheng Cong (A0332295A)

---

## Project Overview (Final report)

This repository contains the implementation for a thesis project on "Multi-Primitive Robotic Manipulation via Proximal Policy Optimization (PPO): Occlusion-Aware Target Selection in Cluttered Table Top Environments." The system trains a Franka Emika Panda robot in simulation to clear a cluttered tabletop using a rule-based hybrid control system that combines deterministic grasping with RL-learned pushing strategies.

-----

## Key Features

* Rule-Based Hybrid Control: Automatic grasping of unoccluded objects combined with RL-learned discrete pushing for occluded targets
* Occlusion-aware reasoning: Explicit spatial representation in the observation space enables the policy to learn when to clear blockers before attempting to grasp a target.
* Stable PPO training: Custom implementation of Proximal Policy Optimization (PPO) with reward shaping, Generalized Advantage Estimation (GAE), and curriculum learning for robust policy convergence. Achieves 90% success rate within 20,000 environment steps
* Discrete Action Space: 8 fixed push directions learned via PPO with categorical policy
* Modular architecture: Decoupled motion primitives (`robot_util`), physics reasoning (`physics_util`), and object reasoning (`object_util`) for maintainability and extensibility.

---

## System Architecture

The system implements a sophisticated rule-based control system with learned push components:

Control Flow:
1. Target Assessment: Heuristic selects nearest uncollected object to goal
2. Occlusion Check: If target is occluded → RL push policy
3. Skill Execution:
* Unoccluded: Automatic grasp-and-place to goal platform
* Occluded: RL-selected push direction from 8 discrete options

# Observation Space
A structured 300+ dimensional vector containing:
* Robot state (joint positions, velocities, end-effector pose, gripper width)
* Object features (positions, velocities, shape descriptors, graspability scores)
* Spatial relationships (pairwise distance matrix, explicit occlusion masks)
* Goal context (goal position and size)

# Action Space

8 Discrete Push Directions:
* 0: 0° (+X direction)
* 1: 45°
* 2: 90° (+Y direction)
* 3: 135°
* 4: 180° (-X direction)
* 5: -135°
* 6: -90° (-Y direction)
* 7: -45°

---

## Quick Start

Prerequisites

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
Run the main training script with the default configuration:

```Bash
python -m scripts.ppo_scratch
```

### Expected Output

You will see:

PyBullet Visualization: A simulation window opens showing the robot training on push-and-grasp tasks with procedurally generated cluttered scenes.

Terminal Training Logs: Live progress monitoring including:
* Episode returns and lengths
* PPO loss components (policy loss, value loss, entropy)
* Success rates and average performance metrics
* Push direction selection statistics

Training metrics

```Bash
[Episode   25] Steps=  2048 | Return=  45.20 | Length= 87 | Avg10=  32.15 | Time=00:00:45
[Update] Steps=  2048 | PolicyLoss=0.1245 | ValueLoss=0.0456 | Entropy=1.2345 | KL=0.0123 | EntCoef=0.0085 | LR=2.85e-4
```
* Peak Performance: 90% success rate achieved at 12,288 steps
* Average Return: 164.5 at best performance
* Efficient Execution: 29.9 steps per episode at peak efficiency

Checkpointing: Model weights saved periodically to `checkpoints/` directory.

---

## Performance Results

# Training Evaluation (20,000 steps):
* Best Success Rate: 90%
* Peak Average Return: 164.5
* Most Efficient Episode: 29.9 steps
* Consistent Performance: 60-90% success rate maintained

# Key Insights:
* The discrete push policy effectively learns to clear occlusions using only 8 fixed directions
* Rule-based grasping provides reliable object transport to goal platform
* Combined system solves complex clutter scenarios with high reliability

---

## Neural Network Validation Demo 

<img width="750" height="696" alt="Image_20251121223224_114_4" src="https://github.com/user-attachments/assets/71164d95-f9c3-4661-a0de-fb51378df6ea" />


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

---

###  Run Environment Smoke Test
If you wish to test the environment's stability with purely random actions (produces a lot of log spam), you can run:

```Bash
python -m scripts.test_custom_env
```

## Demo Video
https://github.com/user-attachments/assets/deaf7ecc-4a43-48ef-a841-1b6cb5a00487

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

`execute_pick_and_place()` - Rule-Based Grasping

Automatic execution for unoccluded objects:
1. Approach object from above
2. Descend to grasp height
3. Close gripper with verification
4. Lift and transport to goal platform
5. Place object and retract

`execute_push()` - RL-Enhanced Pushing

Policy-selected direction for occluded objects:
1. Calculate push vector from discrete direction index
2. Approach with orthogonal offset for torque
3. Execute linear push along selected direction
4. Retract and reset for next action

---

## Rule-Based Decision Logic

```
# Priority 1: Grasp non-occluded objects
unoccluded_objects = [obj for obj in objects if not occluded(obj)]
if unoccluded_objects:
    target = select_nearest_to_goal(unoccluded_objects)
    execute_pick_and_place(target)

# Priority 2: Push occluded objects  
else:
    target = select_nearest_to_goal(all_objects)
    push_direction = policy.act(observation)  # Discrete 0-7
    execute_push(target, push_direction)
```

---

## Technical Insights

# Why This Architecture Works:

* Decomposed Complexity: Separates reliable grasping from learned pushing
* Sample Efficiency: RL only learns push directions, not entire manipulation
* Robustness: Rule-based components prevent catastrophic failures
* Interpretability: Clear decision boundaries between skills

# Performance Characteristics
* Training Stability: Consistent 60-90% success despite physics stochasticity
* Generalization: Effective on novel object configurations
* Efficiency: Fast inference with discrete action selection

---

## Citation
````
@article{gallouedec2021pandagym,
  title        = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
  author       = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
  year         = 2021,
  journal      = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
}










