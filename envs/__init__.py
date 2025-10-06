from gymnasium.envs.registration import register

register(
    id='StrategicPushAndGrasp-v0',
    entry_point='envs.strategic_env:StrategicPushAndGraspEnv',
    max_episode_steps=100
)