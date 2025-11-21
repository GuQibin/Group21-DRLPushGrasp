# Run with "python -m scripts.ppo_scratch"

# ===================================================================
# PPO FROM SCRATCH — MODIFIED FOR INCREASED OBSERVABILITY
# ===================================================================

import os
import time
import math
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym


# ====== UTILITY ======
def _fmt_secs(s: float) -> str:
    s = max(0, float(s))
    h = int(s // 3600);
    s -= 3600 * h
    m = int(s // 60);
    s -= 60 * m
    return f"{h:02d}:{m:02d}:{int(s):02d}"


# ====== IMPORT YOUR CUSTOM ENVIRONMENT ======
from envs.strategic_env import StrategicPushAndGraspEnv


# (RewardNormalizer, ActorCritic, RolloutBuffer classes remain UNCHANGED from your upload)
# - To save space, I assume these classes are pasted here exactly as they were.
# If you copy-paste, ensure RewardNormalizer, ActorCritic, RolloutBuffer are present here.

# ... [Insert RewardNormalizer class here] ...
# ... [Insert ActorCritic class here] ...
# ... [Insert RolloutBuffer class here] ...

class RewardNormalizer:
    def __init__(self, gamma=0.99, epsilon=1e-8):
        self.mean = 0.0;
        self.var = 1.0;
        self.count = epsilon;
        self.gamma = gamma;
        self.returns = 0.0

    def update(self, reward):
        self.returns = reward + self.gamma * self.returns
        self.count += 1
        delta = self.returns - self.mean
        self.mean += delta / self.count
        delta2 = self.returns - self.mean
        self.var += delta * delta2

    def normalize(self, reward):
        return reward / (np.sqrt(self.var / self.count) + 1e-8)

    def reset_episode(self): self.returns = 0.0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, num_layers=4, activation="tanh"):
        super().__init__()

        def _act_factory(name):
            return nn.Tanh if name == "tanh" else nn.ReLU

        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size));
            layers.append(_act_factory(activation)())
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.v_head = nn.Linear(hidden_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, obs):
        h = self.backbone(obs)
        return self.policy_head(h), self.v_head(h)

    def act(self, obs):
        logits, v = self.forward(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a).unsqueeze(-1), v, a.float().unsqueeze(-1)

    def evaluate_actions(self, obs, raw_actions):
        logits, v = self.forward(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(raw_actions.squeeze(-1).long()).unsqueeze(-1)
        return logp, dist.entropy().unsqueeze(-1), v


class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.size = size;
        self.device = device;
        self.ptr = 0;
        self.full = False
        self.obs = torch.zeros((size, obs_dim), device=device)
        self.raw_a = torch.zeros((size, 1), device=device)
        self.logp = torch.zeros((size, 1), device=device)
        self.v = torch.zeros((size, 1), device=device)
        self.r = torch.zeros((size, 1), device=device)
        self.done = torch.zeros((size, 1), device=device)
        self.adv = torch.zeros((size, 1), device=device)
        self.ret = torch.zeros((size, 1), device=device)

    def add(self, obs, raw_a, logp, v, r, done):
        self.obs[self.ptr] = obs;
        self.raw_a[self.ptr] = raw_a
        self.logp[self.ptr] = logp;
        self.v[self.ptr] = v
        self.r[self.ptr] = r;
        self.done[self.ptr] = done
        self.ptr += 1
        if self.ptr >= self.size: self.full = True; self.ptr = 0

    def compute_gae(self, last_v, gamma=0.99, lam=0.95):
        T = self.size if self.full else self.ptr
        adv = torch.zeros((T, 1), device=self.device);
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.done[t]
            next_v = last_v if t == T - 1 else self.v[t + 1]
            delta = self.r[t] + gamma * next_v * nonterminal - self.v[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        self.adv[:T] = adv;
        self.ret[:T] = adv + self.v[:T]

    def get(self, batch_size):
        T = self.size if self.full else self.ptr
        idx = torch.randperm(T, device=self.device)
        for start in range(0, T, batch_size):
            mb = idx[start:start + batch_size]
            yield self.obs[mb], self.raw_a[mb], self.logp[mb], self.v[mb], self.adv[mb], self.ret[mb]


# ========================= CONFIGURATION ===========================
@dataclass
class PPOConfig:
    # ### MODIFIED: INCREASED STEPS ###
    total_steps: int = 500_000  # Increased from 100k/300k to ensure convergence
    checkpoint_interval: int = 5_000
    load_model_path: str = None

    rollout_steps: int = 4096
    update_epochs: int = 4
    mini_batch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 1.0
    ent_coef_start: float = 0.01
    ent_coef_end: float = 0.001
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    use_lr_schedule: bool = True
    lr_min: float = 1e-5
    target_kl: float = 0.02
    normalize_rewards: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    motion_scale: float = 0.5
    hidden_size: int = 256
    num_layers: int = 4
    activation: str = "tanh"
    save_dir: str = "checkpoints_run_rebalanced"  # New folder


def make_env(render=False, motion_scale=1.0):
    return StrategicPushAndGraspEnv(render_mode="human" if render else 'rgb_array', motion_scale=motion_scale)


def ppo_train(cfg: PPOConfig):
    run_name = f"ppo_rebalanced_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"📊 Logging to: runs/{run_name}")

    torch.manual_seed(cfg.seed);
    np.random.seed(cfg.seed);
    random.seed(cfg.seed)
    env = make_env(render=False, motion_scale=cfg.motion_scale)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n,
                      cfg.hidden_size, cfg.num_layers, cfg.activation).to(cfg.device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    reward_normalizer = RewardNormalizer(gamma=cfg.gamma) if cfg.normalize_rewards else None
    buf = RolloutBuffer(cfg.rollout_steps, env.observation_space.shape[0], env.action_space.n, cfg.device)

    global_steps = 0
    obs, _ = env.reset(seed=cfg.seed)

    # Metrics
    ep_ret, ep_raw_ret, ep_len = 0.0, 0.0, 0
    # ### MODIFIED: NEW METRICS ###
    ep_push_reward = 0.0
    ep_push_success = 0
    ep_rl_actions = 0

    scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=cfg.lr_min / cfg.lr,
                                            total_iters=cfg.total_steps // cfg.rollout_steps) if cfg.use_lr_schedule else None

    while global_steps < cfg.total_steps:
        net.eval()
        for step in range(cfg.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            action_idx, logp, v, raw_a = net.act(obs_t)

            obs_next, reward, terminated, truncated, info = env.step(action_idx)

            # Trackers
            global_steps += 1
            ep_raw_ret += reward
            ep_len += 1

            # ### MODIFIED: CAPTURE SPECIFIC METRICS ###
            if info.get("action_type") == "push":
                ep_rl_actions += 1
                # Capture raw pushing reward component
                r_breakdown = info.get("reward_breakdown", {})
                ep_push_reward += r_breakdown.get("occlusion_change", 0)
                if info.get("push_success", False):
                    ep_push_success += 1

            if reward_normalizer:
                reward_normalizer.update(reward)
                reward = reward_normalizer.normalize(reward)
            ep_ret += reward

            done = terminated or truncated
            buf.add(obs_t.squeeze(0), raw_a.squeeze(0), logp.squeeze(0), v.squeeze(0),
                    torch.tensor([reward], device=cfg.device), torch.tensor([float(done)], device=cfg.device))
            obs = obs_next

            if done:
                writer.add_scalar("Charts/Episode_Return_Raw", ep_raw_ret, global_steps)
                writer.add_scalar("Charts/Episode_Length", ep_len, global_steps)

                # ### MODIFIED: LOGGING NEW METRICS ###
                writer.add_scalar("Charts/Push_Reward_Total", ep_push_reward, global_steps)
                writer.add_scalar("Charts/Push_Success_Count", ep_push_success, global_steps)
                writer.add_scalar("Charts/RL_Action_Ratio", ep_rl_actions / ep_len, global_steps)

                print(
                    f"Step {global_steps} | RawRet: {ep_raw_ret:.1f} | PushRet: {ep_push_reward:.1f} | PushSucc: {ep_push_success}")

                ep_ret, ep_raw_ret, ep_len = 0.0, 0.0, 0
                ep_push_reward, ep_push_success, ep_rl_actions = 0.0, 0, 0
                if reward_normalizer: reward_normalizer.reset_episode()
                obs, _ = env.reset()

            if global_steps >= cfg.total_steps: break

        # Update Phase
        with torch.no_grad():
            _, last_v = net.forward(torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0))
        buf.compute_gae(last_v.squeeze())

        net.train()
        ent_coef = cfg.ent_coef_start + (cfg.ent_coef_end - cfg.ent_coef_start) * (global_steps / cfg.total_steps)

        for epoch in range(cfg.update_epochs):
            for mb_obs, mb_a, mb_logp, mb_v, mb_adv, mb_ret in buf.get(cfg.mini_batch):
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                new_logp, entropy, v = net.evaluate_actions(mb_obs, mb_a)
                ratio = torch.exp(new_logp - mb_logp)

                loss = -torch.min(ratio * mb_adv,
                                  torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv).mean() + \
                       cfg.vf_coef * 0.5 * (mb_ret - v).pow(2).mean() - ent_coef * entropy.mean()

                opt.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm);
                opt.step()

        if scheduler: scheduler.step()

        # Save logic
        if global_steps % cfg.checkpoint_interval == 0:
            torch.save(net.state_dict(), f"{cfg.save_dir}/model_{global_steps}.pt")

    env.close()
    writer.close()
    return net


if __name__ == "__main__":
    cfg = PPOConfig()
    ppo_train(cfg)