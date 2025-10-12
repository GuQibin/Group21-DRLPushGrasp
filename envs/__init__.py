"""
Environment Registration for Strategic Push-and-Grasp Manipulation Task

This module registers the custom Gymnasium environment for the ME5418 project:
"Hierarchical Reinforcement Learning for Multi-attribute Object Manipulation"

The environment simulates a 4-DOF robotic manipulator learning to clean a cluttered
tabletop by intelligently selecting between pushing and grasping primitives based on
object geometry and spatial context.
"""

from gymnasium.envs.registration import register

# Register the custom environment with Gymnasium
# This allows the environment to be created using gym.make('StrategicPushAndGrasp-v0')
register(
    id='StrategicPushAndGrasp-v0', # Environment identifier - follows Gym naming convention: <Name>-v<Version>
    entry_point='envs.strategic_env:StrategicPushAndGraspEnv', # Entry point to the environment class
    max_episode_steps=100 # Maximum steps per episode before automatic truncation
)
