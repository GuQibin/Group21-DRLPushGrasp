# Run with "python -m scripts.ppo_scratch"

# ppo_scratch.py
import os
import time
import math
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import gymnasium as gym

import time

# --- progress utils (NEW) ---
def _fmt_secs(s: float) -> str:
    s = max(0, float(s))
    h = int(s // 3600); s -= 3600*h
    m = int(s // 60);   s -= 60*m
    return f"{h:02d}:{m:02d}:{int(s):02d}"


# ====== 引入你的环境（按你的工程结构改这两行）======
# from envs.strategic_pushgrasp_env import StrategicPushAndGraspEnv
from envs.strategic_env import StrategicPushAndGraspEnv
# ==================================================

#added cong
class RewardNormalizer:
    """
    Running normalization of rewards to stabilize training.
    This is the MOST IMPORTANT fix for reducing oscillation (80% improvement).
    """
    def __init__(self, gamma=0.99, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.gamma = gamma
        self.returns = 0.0
        
    def update(self, reward):
        """Update running statistics with new reward"""
        self.returns = reward + self.gamma * self.returns
        self.count += 1
        delta = self.returns - self.mean
        self.mean += delta / self.count
        delta2 = self.returns - self.mean
        self.var += delta * delta2
        
    def normalize(self, reward):
        """Normalize single reward"""
        std = np.sqrt(self.var / self.count)
        return reward / (std + 1e-8)
    
    def reset_episode(self):
        """Reset episode-level return tracking"""
        self.returns = 0.0


# ---- 工具：把动作用 tanh 限幅到 [-1,1]，同时修正 log_prob（Tanh 正态的对数雅可比项） ----
def squash_action_and_log_prob(mu, std, raw_action=None, eps=1e-6):
    dist = Normal(mu, std)
    if raw_action is None:
        raw_action = dist.rsample()   # reparameterization
    log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)
    # tanh squash
    action = torch.tanh(raw_action)
    # change of variables: log|det d(tanh)/dx| = sum log(1 - tanh(x)^2)
    log_prob -= torch.log(1 - action.pow(2) + eps).sum(-1, keepdim=True)
    return action, log_prob, raw_action

# ---- 网络 ----
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, num_layers=2, activation="tanh"):
        super().__init__()

        # 用“类/工厂”，每次循环都 new 一个激活层实例
        def _act_factory(name: str):
            name = name.lower()
            if name == "tanh": return nn.Tanh
            if name == "relu": return nn.ReLU
            if name == "leakyrelu": return lambda: nn.LeakyReLU(negative_slope=0.01)
            if name == "elu": return nn.ELU
            return nn.Tanh

        Act = _act_factory(activation)

        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(Act())           # ✅ 每层一个新实例
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)


        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))
        self.v_head = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        h = self.backbone(obs)
        mu = self.mu_head(h)
        v = self.v_head(h)
        std = torch.exp(self.logstd).expand_as(mu)
        return mu, std, v

    def act(self, obs):
        mu, std, v = self.forward(obs)
        a, logp, raw = squash_action_and_log_prob(mu, std)
        return a, logp, v, mu, std, raw

    def evaluate_actions(self, obs, raw_actions):
        """给定 obs 和 '未tanh前' 的 raw_actions，复现 log_prob（做 PPO ratio）"""
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        logp = dist.log_prob(raw_actions).sum(-1, keepdim=True)
        # tanh 的雅可比修正：注意这里也要把 raw_actions 映成 tanh(action)
        a = torch.tanh(raw_actions)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)  # 原高斯熵（未扣雅可比），够用
        return logp, entropy, v

# ---- Rollout Buffer（on-policy）----
class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False

        self.obs   = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.raw_a = torch.zeros((size, act_dim), dtype=torch.float32, device=device) # 未tanh前
        self.a     = torch.zeros((size, act_dim), dtype=torch.float32, device=device) # tanh后动作（可选）
        self.logp  = torch.zeros((size, 1),       dtype=torch.float32, device=device)
        self.v     = torch.zeros((size, 1),       dtype=torch.float32, device=device)
        self.r     = torch.zeros((size, 1),       dtype=torch.float32, device=device)
        self.done  = torch.zeros((size, 1),       dtype=torch.float32, device=device)

        # GAE 之后填充
        self.adv   = torch.zeros((size, 1),       dtype=torch.float32, device=device)
        self.ret   = torch.zeros((size, 1),       dtype=torch.float32, device=device)

    def add(self, obs, raw_a, a, logp, v, r, done):
        self.obs[self.ptr]   = obs
        self.raw_a[self.ptr] = raw_a
        self.a[self.ptr]     = a
        self.logp[self.ptr]  = logp
        self.v[self.ptr]     = v
        self.r[self.ptr]     = r
        self.done[self.ptr]  = done
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True
            self.ptr = 0

    def compute_gae(self, last_v, gamma=0.99, lam=0.95):
        T = self.size if self.full else self.ptr
        adv = torch.zeros((T, 1), device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.done[t]
            next_v = last_v if t == T-1 else self.v[t+1]
            delta = self.r[t] + gamma*next_v*nonterminal - self.v[t]
            gae = delta + gamma*lam*nonterminal*gae
            adv[t] = gae
        ret = adv + self.v[:T]
        self.adv[:T] = adv
        self.ret[:T] = ret

    def get(self, batch_size):
        T = self.size if self.full else self.ptr
        idx = torch.randperm(T, device=self.device)
        for start in range(0, T, batch_size):
            end = start + batch_size
            mb = idx[start:end]
            yield (
                self.obs[mb],
                self.raw_a[mb],
                self.a[mb],
                self.logp[mb],
                self.v[mb],
                self.adv[mb],
                self.ret[mb],
            )

# ---- 训练器 ---- added cong
@dataclass
class PPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 8192 # original: 4096
    update_epochs: int = 4 # original: 10
    mini_batch: int = 512 #original: 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 1.0 # original: 0.5
    ent_coef_start: float = 0.01     # added cong
    ent_coef_end: float = 0.001      # added cong
    # ent_coef: float = 0.01 
    lr: float = 3e-4
    max_grad_norm: float = 0.5 # original: 1.0
    use_lr_schedule: bool = True     # added cong
    lr_min: float = 1e-5        # added cong
    target_kl: float = 0.02          # NEW: Stop if KL divergence exceeds this - added cong
    normalize_rewards: bool = True   # NEW: Enable reward normalization - added cong
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ===== 新增网络结构参数 =====
    hidden_size: int = 256          # 每层隐藏维度
    num_layers: int = 2             # 隐藏层层数
    activation: str = "tanh"        # 激活函数类型 ('tanh', 'relu', etc.)

#======added cong=====
def get_entropy_coef(step, total_steps, start_coef, end_coef):
    """Linear decay of entropy coefficient over training"""
    progress = min(step / total_steps, 1.0)
    return start_coef + (end_coef - start_coef) * progress
#======added cong=====

def make_env(render=False):
    env = StrategicPushAndGraspEnv(render_mode="human" if render else 'rgb_array')
    return env

def ppo_train(cfg=PPOConfig()):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # === 打印模型参数配置 ===
    print("\n================ PPO CONFIGURATION ================")
    for k, v in vars(cfg).items():
        print(f"{k:<20}: {v}")
    print("===================================================")
    time.sleep(2)

    env = make_env(render=False)
    obs, _ = env.reset(seed=cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"\n[ENV INFO] obs_dim = {obs_dim}, act_dim = {act_dim}")
    time.sleep(2)

    net = ActorCritic(
    obs_dim,
    act_dim,
    hidden_size=cfg.hidden_size,
    num_layers=cfg.num_layers,
    activation=cfg.activation
    ).to(cfg.device)

    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    # added cong: Learning rate scheduler (cosine annealing)
    scheduler = None
    if cfg.use_lr_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        total_updates = cfg.total_steps // cfg.rollout_steps
        scheduler = CosineAnnealingLR(opt, T_max=total_updates, eta_min=cfg.lr_min)
        print(f"Learning rate scheduler enabled: {cfg.lr} → {cfg.lr_min}")

    buf = RolloutBuffer(cfg.rollout_steps, obs_dim, act_dim, cfg.device)

    # ========================================================================
    # added cong NEW: Initialize reward normalizer
    # ========================================================================
    reward_normalizer = None
    if cfg.normalize_rewards:
        reward_normalizer = RewardNormalizer(gamma=cfg.gamma)
        print(f"Reward normalizer initialized")

    ep_ret = 0.0
    ep_len = 0
    global_steps = 0

    # --- progress meters (NEW) ---
    start_time = time.time()
    last_report_time = start_time
    rollout_count = 0
    update_count = 0
    steps_since_last_report = 0

    # --- 训练日志 (NEW) removed cong ---
    ## loss_log = []
    ## return_log = []


    import matplotlib.pyplot as plt
    save_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Info] Plot dir: {save_dir}")

    while global_steps < cfg.total_steps:
        # ====== 收集一段 on-policy 轨迹 ======
        ep_returns_this_rollout = []   # (NEW) 记录这一轮次内结束的所有 episode 的 return
        for _ in range(cfg.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            with torch.no_grad():
                a_t, logp_t, v_t, mu, std, raw = net.act(obs_t)
            # 注意：env 期望的是 tanh 后的动作 a_t ∈ [-1,1]
            action = a_t.squeeze(0).cpu().numpy()
            next_obs, r, terminated, truncated, info = env.step(action)
            done_flag = float(terminated or truncated)

            # ===== added cong Normalize reward before storing ====
            if reward_normalizer is not None:
                reward_normalizer.update(r)
                r_normalized = reward_normalizer.normalize(r)
            else:
                r_normalized = r
            # ======================================================
            
            # 存入 buffer（保存 raw action 以复现 logprob）
            buf.add(
                obs_t.squeeze(0),
                raw.squeeze(0),
                a_t.squeeze(0),
                logp_t.squeeze(0),
                v_t.squeeze(0),
                torch.tensor([r], dtype=torch.float32, device=cfg.device),
                torch.tensor([done_flag], dtype=torch.float32, device=cfg.device),
            )

            ep_ret += r  # Track original (not normalized) for logging
            ep_len += 1
            global_steps += 1

            obs = next_obs

            if done_flag > 0.5:
                # Episode ended - added cong
                episode_returns.append(ep_ret)
                episode_lengths.append(ep_len)

                # Calculate statistics for last 10 episodes -  added cong
                recent_returns = episode_returns[-10:] if len(episode_returns) >= 10 else episode_returns
                avg_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)

                print(f"[Episode End] Steps={global_steps:6d} | Return={ep_ret:7.2f} | "
                      f"Length={ep_len:3d} | Avg(10)={avg_return:7.2f}±{std_return:5.2f}")
                
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

                if reward_normalizer is not None:
                    reward_normalizer.reset_episode()

        # ====== GAE / Return ======
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            _, _, last_v = net.forward(obs_t)
        buf.compute_gae(last_v=last_v.squeeze(0), gamma=cfg.gamma, lam=cfg.gae_lambda)

        # 标准化优势（重要）
        adv = buf.adv[:buf.ptr if not buf.full else buf.size]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf.adv[:adv.shape[0]] = adv

        # ================================================================
        # NEW: Get current entropy coefficient (decays over training) - added cong
        # ================================================================
        current_ent_coef = get_entropy_coef(
            global_steps, 
            cfg.total_steps,
            cfg.ent_coef_start,
            cfg.ent_coef_end
        )

        # ====== PPO 更新（多 epoch 小批次）======
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(cfg.update_epochs):
            early_stop = False
            
            for mb_obs, mb_raw_a, _, mb_logp_old, _, mb_adv, mb_ret in buf.get(cfg.mini_batch):
                new_logp, entropy, v = net.evaluate_actions(mb_obs, mb_raw_a)

                ratio = torch.exp(new_logp - mb_logp_old)
                
                # Check KL divergence for early stopping
                with torch.no_grad():
                    approx_kl = (mb_logp_old - new_logp).mean().item()
                    if approx_kl > cfg.target_kl:
                        print(f"  ⚠️ Early stopping at epoch {epoch+1}/{cfg.update_epochs}, KL={approx_kl:.4f} > {cfg.target_kl}")
                        early_stop = True
                        break
                
                # 策略损失（带剪切）
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = 0.5 * (mb_ret - v).pow(2).mean()

                # 熵正则（鼓励探索）- 使用动态系数
                entropy_loss = -entropy.mean()

                # Total loss with current entropy coefficient
                loss = policy_loss + cfg.vf_coef * value_loss + current_ent_coef * entropy_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
            
            if early_stop:
                break

        # ================================================================
        # NEW: Step learning rate scheduler
        # ================================================================
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg.lr

        # Compute average losses
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0

        print(f"[Update] Steps={global_steps:6d} | "
              f"PolicyLoss={avg_policy_loss:.4f} | ValueLoss={avg_value_loss:.4f} | "
              f"Entropy={avg_entropy:.4f} | EntCoef={current_ent_coef:.4f} | LR={current_lr:.2e}")

    env.close()

    # ========================================================================
    # Final statistics
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    if len(episode_returns) > 0:
        print(f"Total episodes: {len(episode_returns)}")
        print(f"Final 10 episodes avg return: {np.mean(episode_returns[-10:]):.2f}")
        print(f"Best episode return: {np.max(episode_returns):.2f}")
        print(f"Final 10 episodes avg length: {np.mean(episode_lengths[-10:]):.1f}")
    print("="*70 + "\n")
    
    return net


# ====== 评估：用确定性策略跑若干回合，统计指标 ======
def evaluate_policy(net, make_env_fn, episodes=10, render=False, record_dir=None, seed=123):
    """
    Args:
        net: 训练好的 ActorCritic
        make_env_fn: 一个无参函数，返回一个新的 env 实例（和训练时相同环境）
        episodes: 评估回合数
        render: 是否人眼渲染
        record_dir: 若提供路径，将用 RecordVideo 录制评估视频（gymnasium>=0.29）
        seed: 评估环境的随机种子起点
    Returns:
        stats: dict，包含 avg_return、avg_length、success_rate（若 info 有该字段）
    """
    import gymnasium as gym
    from collections import defaultdict

    # 每次评估都新建环境，避免被训练 loop 的状态影响
    env = make_env_fn(render=render)

    # 可选：视频录制（需要 env 支持 render_mode="rgb_array"）
    if record_dir is not None:
        try:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=record_dir,
                episode_trigger=lambda i: True,  # 录每一回合
                name_prefix="eval"
            )
            print(f"[Eval] Recording to: {record_dir}")
        except Exception as e:
            print(f"[Eval] RecordVideo init failed: {e}")

    net.eval()
    rng = np.random.RandomState(seed)
    returns, lengths = [], []
    succ_cnt = 0
    succ_total = 0

    with torch.inference_mode():
        for ep in range(episodes):
            # 为了可重复性，回合级种子递增
            obs, _ = env.reset(seed=int(rng.randint(0, 1e9)))
            ep_ret, ep_len = 0.0, 0

            while True:
                # 确定性动作：用均值 mu，经 tanh 限幅
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=next(net.parameters()).device).unsqueeze(0)
                mu, _, _v = net.forward(obs_t)
                a = torch.tanh(mu)  # 确定性 = 不采样，直接用 mu
                action = a.squeeze(0).cpu().numpy()

                obs, r, terminated, truncated, info = env.step(action)
                ep_ret += r
                ep_len += 1

                if render and hasattr(env, "render"):
                    env.render()

                done = terminated or truncated
                if done:
                    returns.append(ep_ret)
                    lengths.append(ep_len)
                    # 统计成功（如果 info 有返回）
                    if isinstance(info, dict):
                        # 常见键名：success / is_success / success_rate（你的环境可自定义）
                        for key in ("success", "is_success"):
                            if key in info:
                                succ_total += 1
                                succ_cnt += 1 if (float(info[key]) > 0.5) else 0
                                break
                    print(f"[Eval Episode {ep+1}/{episodes}] return={ep_ret:.2f}, len={ep_len}")
                    break

    env.close()

    avg_ret = float(np.mean(returns)) if returns else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    succ_rate = (succ_cnt / succ_total) if succ_total > 0 else None

    stats = {
        "avg_return": avg_ret,
        "avg_length": avg_len,
        "success_rate": succ_rate,
        "episodes": episodes
    }
    print("[Eval Summary]", stats)
    return stats

if __name__ == "__main__":
    cfg = PPOConfig(
        total_steps=200_000,
        rollout_steps= 8192, #original: 4096
        update_epochs=4, #original: 10
        mini_batch=512, #original: 256
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=1.0, #original: 0.5
        ent_coef_start=0.01,
        ent_coef_end=0.001,
        # original: ent_coef=0.01,
        lr=3e-4,
        max_grad_norm= 0.5, #original: 1.0
        use_lr_schedule=True, #added cong
        lr_min=1e-5,                # (added cong)
        target_kl=0.02,           # Early stopping threshold (added cong)
        normalize_rewards=True,   # Enable reward normalization (added cong)
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        hidden_size=256,
        num_layers=2,
        activation="tanh"   
    )
    net = ppo_train(cfg)

    # 训练完成后做一次评估（不渲染，跑 10 回合）
    print("\n Starting evaluation...")
    evaluate_policy(
        net,
        make_env_fn=lambda render=False: make_env(render),
        episodes=10,
        render=False,
        record_dir=None,   # 想录视频就给个文件夹，比如 "videos/eval_001"
        seed=123
    )
