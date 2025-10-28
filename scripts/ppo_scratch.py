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

# ---- 训练器 ----
@dataclass
class PPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 4096
    update_epochs: int = 10
    mini_batch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ===== 新增网络结构参数 =====
    hidden_size: int = 256          # 每层隐藏维度
    num_layers: int = 2             # 隐藏层层数
    activation: str = "tanh"        # 激活函数类型 ('tanh', 'relu', etc.)

def make_env(render=False):
    # 你可以在这里切换 single_object_demo 或其它 options
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

    buf = RolloutBuffer(cfg.rollout_steps, obs_dim, act_dim, cfg.device)

    ep_ret = 0.0
    ep_len = 0
    global_steps = 0

    # --- progress meters (NEW) ---
    start_time = time.time()
    last_report_time = start_time
    rollout_count = 0
    update_count = 0
    steps_since_last_report = 0

    # --- 训练日志 (NEW) ---
    loss_log = []
    return_log = []


    import matplotlib.pyplot as plt
    save_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Info] Plot dir: {save_dir}")

    while global_steps < cfg.total_steps:
        # ====== 收集一段 on-policy 轨迹 ======
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

            ep_ret += r
            ep_len += 1
            global_steps += 1

            obs = next_obs

            if done_flag > 0.5:
                # 终止/截断就 reset
                print(f"[Episode End] Return={ep_ret:.3f}, Length={ep_len}")
                ep_returns_this_rollout.append(ep_ret)   # (NEW) 只记录到“本轮次”的暂存里
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        # ====== GAE / Return ======
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            _, _, last_v = net.forward(obs_t)
        buf.compute_gae(last_v=last_v.squeeze(0), gamma=cfg.gamma, lam=cfg.gae_lambda)

        # 标准化优势（重要）
        adv = buf.adv[:buf.ptr if not buf.full else buf.size]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf.adv[:adv.shape[0]] = adv

        # ====== PPO 更新（多 epoch 小批次）======
        epoch_loss_list = []  # (NEW) 统计本轮次里各 epoch 的平均 loss
        for epoch in range(cfg.update_epochs):
            # 统计本 epoch 的平均指标
            kl_list, pol_list, val_list, ent_list, tot_list = [], [], [], [], []
            for mb_obs, mb_raw_a, _, mb_logp_old, _, mb_adv, mb_ret in buf.get(cfg.mini_batch):
                new_logp, entropy, v = net.evaluate_actions(mb_obs, mb_raw_a)

                # ratio & 损失
                ratio = torch.exp(new_logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (mb_ret - v).pow(2).mean()
                entropy_loss = -entropy.mean()  # 取负号，后面乘以 ent_coef

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

                # 统计本 minibatch 的 KL（近似）
                with torch.no_grad():
                    approx_kl_mb = (mb_logp_old - new_logp).mean().item()
                    kl_list.append(approx_kl_mb)
                    pol_list.append(policy_loss.item())
                    val_list.append(value_loss.item())
                    ent_list.append((-entropy_loss).item())  # 还原成正的 entropy
                    tot_list.append(loss.item())

            # 每个 epoch 打印一次
            print(
                f"[Update-Epoch {epoch+1}/{cfg.update_epochs}] "
                f"Steps={global_steps} | "
                f"Loss={np.mean(tot_list):.5f} | "
                f"Policy={np.mean(pol_list):.5f} | "
                f"Value={np.mean(val_list):.5f} | "
                f"Entropy={np.mean(ent_list):.5f} | "
                f"KL={np.mean(kl_list):.5f}"
            )
            # 记录平均 loss（每个 epoch）
            epoch_loss_list.append(np.mean(tot_list))   # (NEW) 暂存本 epoch 平均 loss

        # —— 这次 rollout 结束后，聚合“按轮次”的指标 ——
        # 1) loss：这一轮次跨所有 epoch 的平均
        rollout_loss = float(np.mean(epoch_loss_list)) if epoch_loss_list else float("nan")
        loss_log.append(rollout_loss)

        # 2) return：这一轮次里结束的多个 episode 的平均
        rollout_return = float(np.mean(ep_returns_this_rollout)) if ep_returns_this_rollout else float("nan")
        return_log.append(rollout_return)

        # --- progress summary (REFINED) ---
        rollout_count += 1
        update_count += cfg.update_epochs

        elapsed = time.time() - start_time
        steps_since_last_report += cfg.rollout_steps

        progress = min(1.0, global_steps / float(cfg.total_steps))
        eta = (elapsed / progress - elapsed) if progress > 0 else float('inf')

        # FPS：自上次打印以来
        now = time.time()
        fps = steps_since_last_report / max(1e-6, (now - last_report_time))
        last_report_time = now
        steps_since_last_report = 0

        # 文本进度条（单行刷新）
        bar_len = 40
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)

        line = (f"[Progress] [{bar}] {progress*100:6.2f}%  "
                f"steps={global_steps}/{cfg.total_steps}  "
                f"rollouts={rollout_count}  updates={update_count}  "
                f"fps={fps:6.1f}  "
                f"elapsed={_fmt_secs(elapsed)}  ETA={_fmt_secs(eta)}")

        print(line, end="\r", flush=True)  # 关键：同一行刷新


        # ====== 每 5 个 rollout 保存一次图 (NEW) ======
        if rollout_count % 5 == 0:
            # X 轴按“轮次”
            xs_loss = np.arange(1, len(loss_log) + 1)
            xs_ret  = np.arange(1, len(return_log) + 1)

            # Loss 曲线（随轮次）
            loss_path = os.path.join(save_dir, "loss_curve.png")
            plt.figure(figsize=(6,4))
            plt.plot(xs_loss, loss_log, linewidth=2)
            plt.xlabel("Rollout (iteration)")
            plt.ylabel("Loss")
            plt.title("PPO Training Loss vs Rollouts")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(loss_path, dpi=300)
            plt.close()

            # Return 曲线（随轮次）
            return_path = os.path.join(save_dir, "return_curve.png")
            plt.figure(figsize=(6,4))
            plt.plot(xs_ret, return_log, linewidth=2)
            plt.xlabel("Rollout (iteration)")
            plt.ylabel("Return (mean per rollout)")
            plt.title("Episode Return vs Rollouts")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(return_path, dpi=300)
            plt.close()

            print(f"[Saved] Plots updated @ rollout {rollout_count}:")
            print(f"        - {loss_path}")
            print(f"        - {return_path}")

    env.close()
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
        rollout_steps=4096,
        update_epochs=10,
        mini_batch=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        lr=3e-4,
        max_grad_norm=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        hidden_size=256,          # 每层隐藏维度
        num_layers=4,             # 隐藏层层数
        activation="tanh"    
    )
    net = ppo_train(cfg)

    # 训练完成后做一次评估（不渲染，跑 10 回合）
    evaluate_policy(
        net,
        make_env_fn=make_env,
        episodes=10,
        render=False,
        record_dir=None,   # 想录视频就给个文件夹，比如 "videos/eval_001"
        seed=123
    )
