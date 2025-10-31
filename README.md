# Group21-DRLPushGrasp
**Hierarchical Reinforcement Learning for Multi-attribute Object Manipulation**
*ME5418 Project Milestone 2: Neural Network Implementation*

---

## Project Overview (Milestone 2)

This project aims to implement a reinforcement learning (RL) framework, enabling a robotic manipulator to learn to co-plan "push" and "grasp" strategies in cluttered environments.

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



## Milestone 2: Neural Network Demo 



The core deliverable for this milestone is a demo script to **isolate and validate the neural network**. It loads our real environment, sets up a simple single-object scene, and demonstrates a complete **forward pass**, **backward pass**, and **parameter update** in an end-to-end training micro-loop.

 please run the following from the project root directory:

```Bash
python -m scripts.demo_nn
```

### Expected Output

You will see:

1. A PyBullet window pop up, showing a simple single-object scene (the robot will execute a few steps).
2. The window will close, and the terminal will print **"Part 1.5: Episode Summary"**, showing the exact action vectors the network **outputted** during the episode.
3. Next, **"Part 2: Backward Pass"** will execute. It will:
   - Calculate a real REINFORCE loss based on the collected rewards.
   - Print a sample network weight **before** the update.
   - Execute `loss.backward()` and `optimizer.step()`.
   - Print the same network weight **after** the update.
4. Finally, you will see a `✓ SUCCESS: Parameter value changed!` message, **proving our network architecture is correct and trainable.**



### (Optional) Run Environment Smoke Test (Milestone 1)



If you wish to test the environment's stability with purely random actions (produces a lot of log spam), you can run:

```Bash
python -m scripts.test_custom_env
```



# Appendix: Core Utilities & Action Primitives

## Object Utilities (`utils/object_util.py`)
- **`
This module centralizes **object-level reasoning** for the Strategic Push–Grasp environment:
shape encoding for NN inputs, pairwise spatial reasoning, occlusion analysis, safe spawning,
and simple (non-learned) target selection.
`**  
## Physics Utilities (`utils/physics_util.py`)
- **`
Utilities that wrap PyBullet’s low-level API into safer, typed helpers for the Strategic Push–Grasp environment. They cover **workspace bounds, collisions, contact forces, stability checks, ray tests, and visualization**. All functions include conservative error handling to keep training loops robust.
`**  
## Robot Utilities (`utils/robot_util.py`)
- **`
High-level **manipulation primitives** (pick–place and push) and robust helpers for
end-effector (EE) state, inverse kinematics, motion control, gripper control, and diagnostics.
These wrap various panda-gym/PyBullet details behind a stable API so the RL policy
can focus on **when** to push vs. grasp—not *how* to drive every joint.
`**  

---

## Core Action Primitives

- **`execute_pick_and_place(sim, robot, target_object, alpha_x, alpha_y, goal_pos, workspace_bounds, approach_height=0.15, grasp_height=0.03) -> bool`**  
  Eight-phase grasp pipeline (approach → descend → close → verify → lift → transport → place → retract).  
  - **Inputs:** normalized offsets `alpha_x/alpha_y ∈ [-1,1]` mapped to ±2.5 cm around the object center; workspace clipping enforced.  
  - **Verification:** micro-lift checks object Z-gain (>1 cm) to confirm a *real* grasp.  
  - **Returns:** `True` only if all phases succeed (prevents false positive rewards).

- **`execute_push(sim, robot, target_object, alpha_x, alpha_y, alpha_theta, workspace_bounds, push_distance=0.05, push_height=0.03, use_object_frame=True) -> bool`**  
  Contact-point selection + straight-line push along a direction parameterized by `alpha_theta` (mapped to angle).  
  - Clips pre/post push waypoints into workspace bounds.  
  - Clean 3-phase routine (pre-push → push → retract).

---

## Citation
````
@article{gallouedec2021pandagym,
  title        = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
  author       = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
  year         = 2021,
  journal      = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
}
