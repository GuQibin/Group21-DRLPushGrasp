# Group21-DRLPushGrasp
Hierarchical Reinforcement Learning for Multi-attribute Object Manipulation:  Joint-Space Control of Push-Grasp Strategies in Constrained Environments


Group21-DRLPushGrasp/
├── envs/
│   ├── __init__.py          # 使环境可以被注册和发现
│   └── strategic_env.py     # 我们的核心环境文件
├── scripts/
│   └── test_custom_env.py   # 用于验证我们新环境的脚本
├── README.md                # 项目说明文件
└── requirements.txt         # 项目依赖文件


python -m scripts.minimal_test
python -m scripts.test_complete_reward
python -m scripts.test_custom_env
python -m scripts.test_env_once
python -m scripts.test_objects_util
python -m scripts.test_physics_util
python -m scripts.test_robot_util
python -m scripts.test_state_action


ZHANG TODO:
1. Export a conda env yaml
2. Modify the robot excution strategy

GU TODO:
1. 修改初始化的物体种类限定

CONG TODO:
1. 给每个函数加注释