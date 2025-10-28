import os
import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

LOG_DIR = "runs/ppo_pushgrasp"
MODEL = os.path.join(LOG_DIR, "best_model.zip")  # 或 ppo_pushgrasp_final.zip

def make_env(render_mode="human"):
    def _thunk():
        return gym.make("StrategicPushAndGrasp-v0", render_mode=render_mode)
    return _thunk

def main():
    env = DummyVecEnv([make_env(render_mode="human")])

    # 加载归一化统计（若训练时使用了 VecNormalize）
    stats_path = os.path.join(LOG_DIR, "vecnormalize_stats.pkl")
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(MODEL)

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()
