# python -m scripts.ppo_shape_demo
from envs.strategic_env import StrategicPushAndGraspEnv
env = StrategicPushAndGraspEnv(render_mode='rgb_array')
obs, info = env.reset(seed=0)
print(obs.shape, obs.dtype, type(info))   # (348,) float32 dict
obs, r, term, trunc, info = env.step(env.action_space.sample())
print(obs.shape, r, term, trunc, type(info))
