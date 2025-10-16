# 🤖 Group21-DRLPushGrasp  
**Hierarchical Reinforcement Learning for Multi-attribute Object Manipulation**  
*Joint-Space Control of Push-Grasp Strategies in Constrained Environments*  

---

## 🧩 Project Overview  
This project implements a **hierarchical reinforcement learning (HRL)** framework that enables a robotic manipulator to **jointly plan and execute push-and-grasp strategies** in cluttered or constrained environments.  

It builds upon a custom PyBullet simulation environment with high-level task abstractions (push vs. grasp) and low-level joint-space control, allowing the agent to learn effective manipulation behaviors.  

---

## 📁 Directory Structure  
Group21-DRLPushGrasp/
├── environment.yaml # Conda env spec (Python 3.8 + pip pkgs)
├── LICENSE
├── README.md # This file
├── requirements.txt # (Optional) pip-style dependency list
├── envs/
│ ├── init.py # Registers the custom env(s)
│ └── strategic_env.py # Core environment implementation
├── scripts/
│ └── test_custom_env.py # Simple loop to test the env end-to-end
└── utils/
├── object_util.py # Object spawning / utilities
├── physics_util.py # Physics helpers (e.g., step/settle)
├── robot_util.py # Robot (gripper/arm) helper functions


---

## ⚙️ Environment Setup

### 1️⃣ Create Conda Environment
If you already have the `environment.yaml` file:

```bash
# Create the Conda environment from the spec (installs Python 3.8 + pip pkgs)
conda env create -f environment.yaml

# Activate the environment (make sure the name matches the 'name:' in YAML)
conda activate me5418

---

# 🚀 Run Demo
# Run the full environment loop to verify registration & stepping
python -m scripts.test_custom_env


give out some comments on this object_util and important functions in it in the README.md

---

## 🧠 Object Utilities (`utils/object_util.py`)

This module centralizes **object-level reasoning** for the Strategic Push–Grasp environment:
shape encoding for NN inputs, pairwise spatial reasoning, occlusion analysis, safe spawning,
and simple (non-learned) target selection.

## 🏗️ Physics Utilities (`utils/physics_util.py`)

Utilities that wrap PyBullet’s low-level API into safer, typed helpers for the Strategic Push–Grasp environment. They cover **workspace bounds, collisions, contact forces, stability checks, ray tests, and visualization**. All functions include conservative error handling to keep training loops robust.

## 🤖 Robot Utilities (`utils/robot_util.py`)

High-level **manipulation primitives** (pick–place and push) and robust helpers for
end-effector (EE) state, inverse kinematics, motion control, gripper control, and diagnostics.
These wrap various panda-gym/PyBullet details behind a stable API so the RL policy
can focus on **when** to push vs. grasp—not *how* to drive every joint.

---

### Core Action Primitives

- **`execute_pick_and_place(sim, robot, target_object, alpha_x, alpha_y, goal_pos, workspace_bounds, approach_height=0.15, grasp_height=0.03) -> bool`**  
  Eight-phase grasp pipeline (approach → descend → close → verify → lift → transport → place → retract).  
  - **Inputs:** normalized offsets `alpha_x/alpha_y ∈ [-1,1]` mapped to ±2.5 cm around the object center; workspace clipping enforced.  
  - **Verification:** micro-lift checks object Z-gain (>1 cm) to confirm a *real* grasp.  
  - **Returns:** `True` only if all phases succeed (prevents false positive rewards).

- **`execute_push(sim, robot, target_object, alpha_x, alpha_y, alpha_theta, workspace_bounds, push_distance=0.05, push_height=0.03, use_object_frame=True) -> bool`**  
  Contact-point selection + straight-line push along a direction parameterized by `alpha_theta` (mapped to angle).  
  - Clips pre/post push waypoints into workspace bounds.  
  - Clean 3-phase routine (pre-push → push → retract).

> Both primitives convert object orientation **quaternion → rotation matrix** and transform offsets
> into world frame, so they work on rotated objects.