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
├── envs/
│   ├── __init__.py          # 使环境可以被注册和发现
│   └── strategic_env.py     # 我们的核心环境文件
├── scripts/
│   └── test_custom_env.py   # 用于验证我们新环境的脚本
├── README.md                # 项目说明文件
└── requirements.txt         # 项目依赖文件

---

## ⚙️ Environment Setup  

### 1️⃣ Create Conda Environment  
If you already have the `environment.yaml` file:  
```bash
conda env create -f environment.yaml -n me5418
conda activate me5418


🚀 Run Tests

You can verify individual components or run quick environment tests:
# Quick initialization test
python -m scripts.minimal_test

# Validate reward logic
python -m scripts.test_complete_reward

# Test full environment loop
python -m scripts.test_custom_env

# Single-step debug test
python -m scripts.test_env_once

# Utility and physics validation
python -m scripts.test_objects_util
python -m scripts.test_physics_util
python -m scripts.test_robot_util
python -m scripts.test_state_action