import os
import gymnasium as gym
import envs  # 必须导入以触发注册
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Dict, Any

LOG_DIR = "runs/ppo_pushgrasp"
os.makedirs(LOG_DIR, exist_ok=True)

class ResetOptionsWrapper(gym.Wrapper):
    """
    在 reset() 注入默认 options 的简单 wrapper，便于做 curriculum。
    例如：前期单物体训练，后期切换多物体。
    """
    def __init__(self, env: gym.Env, default_options: Optional[Dict[str, Any]] = None):
        super().__init__(env)
        self.default_options = default_options or {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        merged = dict(self.default_options)
        if options:
            merged.update(options)
        return self.env.reset(seed=seed, options=merged)

def make_env(render_mode=None, seed=42, reset_options: Optional[Dict[str, Any]] = None):
    def _thunk():
        e = gym.make("StrategicPushAndGrasp-v0", render_mode=render_mode)
        if reset_options:
            e = ResetOptionsWrapper(e, reset_options)
        e = Monitor(e)  # 日志 episode 长度、回报
        return e
    return _thunk

def main():
    seed = 42

    # 可选：先用单物体演示做“热身”（更易学到抓取/推动基础）
    curriculum_options = {
        "single_object_demo": False,  # True 时将只生成单个物体
        # "goal_pos": [-0.2, -0.2], "goal_size": 0.1,
        # "object_type": "cube", "object_pos": [0.45, 0.0], "cube_half": 0.02
    }

    # 训练与评估环境（无渲染）
    train_env = DummyVecEnv([make_env(render_mode='rgb_array', seed=seed, reset_options=curriculum_options)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(render_mode='rgb_array', seed=seed+1, reset_options=curriculum_options)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 评估回调（每 eval_freq 步评估一次，保存最佳模型）
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    policy_kwargs = dict(net_arch=[256, 256])  # 348D 观测，建议较宽 MLP
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        seed=seed,
        verbose=1,
    )

    total_timesteps = 500_000  # 先小规模试跑，可再增大
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 保存模型与归一化统计
    model_path = os.path.join(LOG_DIR, "ppo_pushgrasp_final")
    model.save(model_path)
    train_env.save(os.path.join(LOG_DIR, "vecnormalize_stats.pkl"))

    print(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()
