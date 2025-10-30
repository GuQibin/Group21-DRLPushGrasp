# Run with "python -m scripts.ppo_scratch"

# ===================================================================
# PPO FROM SCRATCH — ANNOTATED FOR ME5418 PROJECT (GROUP 21)
# Hierarchical RL for Multi-attribute Object Manipulation
# ===================================================================
# Project Goal: Learn joint-space control of push/grasp strategies
# Environment: StrategicPushAndGraspEnv (4-DOF arm, cluttered table)
# Action: Unified 4D vector A = (α_skill, α_x, α_y, α_θ) ∈ [-1,1]^4
# High-Level Primitives: execute_pick_and_place() or execute_push()
# Low-Level Control: Handled by pre-programmed IK + motion planners
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
from torch.distributions import Normal

import gymnasium as gym

# --------------------- UTILITY: Time Formatting ---------------------
def _fmt_secs(s: float) -> str:
    s = max(0, float(s))
    h = int(s // 3600); s -= 3600*h
    m = int(s // 60);   s -= 60*m
    return f"{h:02d}:{m:02d}:{int(s):02d}"


# ====== IMPORT YOUR CUSTOM ENVIRONMENT ==============================
# This is where your StrategicPushAndGraspEnv is defined
# Must implement:
#   - observation_space: full state (robot + objects + spatial relations)
#   - action_space: Box(-1,1, shape=(4,)) → (α_skill, α_x, α_y, α_θ)
#   - step(): executes high-level primitive via pre-programmed functions
#   - reset(): randomizes object positions, shapes, goal zones
from envs.strategic_env import StrategicPushAndGraspEnv

# ==================== REWARD NORMALIZATION =========================
# CRITICAL FOR STABILITY in sparse/reward-scale-varying tasks
# Your rewards: +5 per object, +25 bonus, small penalties
# Without normalization → PPO oscillates wildly
class RewardNormalizer:
    """
    Online running mean/std normalization of rewards.
    Uses Welford's algorithm + return tracking for stability.
    """
    def __init__(self, gamma=0.99, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.gamma = gamma
        self.returns = 0.0
        
    def update(self, reward):
        """Update running stats using current (normalized) reward."""
        self.returns = reward + self.gamma * self.returns
        self.count += 1
        delta = self.returns - self.mean
        self.mean += delta / self.count
        delta2 = self.returns - self.mean
        self.var += delta * delta2 #incremental variance
        
    def normalize(self, reward):
        """Z-score normalize reward using running stats."""
        std = np.sqrt(self.var / self.count)
        return reward / (std + 1e-8)
    
    def reset_episode(self):
        """Reset episode-level return tracking"""
        self.returns = 0.0


# ============ ACTION SQUASHING + LOG PROB CORRECTION ===============
# PPO needs correct log_prob under tanh squashing
# Without Jacobian correction → biased policy gradient

def squash_action_and_log_prob(mu, std, raw_action=None, eps=1e-4):  # FIXED: increased eps from 1e-6
    """
    Sample action ~ N(mu, std), squash to [-1,1] via tanh,
    and compute log_prob with change-of-variables correction.
    """
    
    dist = Normal(mu, std)
    if raw_action is None:
        raw_action = dist.rsample()   # reparameterization
    log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)
    
    # Apply tanh: a = tanh(x)
    action = torch.tanh(raw_action)
    
    # Jacobian: log|d(tanh(x))/dx| = log(1 - tanh(x)^2)
    # We subtract it → correct log density under transformation
    log_prob -= torch.log(1 - action.pow(2) + eps).sum(-1, keepdim=True)
    return action, log_prob, raw_action

# ===================== ACTOR-CRITIC NETWORK ========================
# Matches your proposal:
# - Object encoder → 64D per object
# - Robot encoder → 128D
# - Cross-attention fusion → 256D scene
# → Here simplified into shared MLP backbone (can be extended)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, num_layers=2, activation="tanh"):
        super().__init__()

        # Activation factory
        def _act_factory(name: str):
            name = name.lower()
            if name == "tanh": return nn.Tanh
            if name == "relu": return nn.ReLU
            if name == "leakyrelu": return lambda: nn.LeakyReLU(negative_slope=0.01)
            if name == "elu": return nn.ELU
            return nn.Tanh

        Act = _act_factory(activation)

        # Shared backbone (can later add per-object MLPs + attention)
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(Act())
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)

        # Policy head: mean of Gaussian (before tanh)
        self.mu_head = nn.Linear(hidden_size, act_dim)
        # Learnable log_std (shared across actions)
        self.logstd = nn.Parameter(torch.zeros(act_dim))
        # Value head: V(s)
        self.v_head = nn.Linear(hidden_size, 1)

        # Xavier init for stability
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
        """Sample action during rollout."""
        mu, std, v = self.forward(obs)
        a, logp, raw = squash_action_and_log_prob(mu, std)
        return a, logp, v, mu, std, raw

    def evaluate_actions(self, obs, raw_actions):
        """Recompute logp, entropy, v for PPO update (given old raw actions)."""
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        logp = dist.log_prob(raw_actions).sum(-1, keepdim=True)
        # tanh jacobian correction
        a = torch.tanh(raw_actions)
        logp -= torch.log(1 - a.pow(2) + 1e-4).sum(-1, keepdim=True)  # FIXED: increased eps
        entropy = dist.entropy().sum(-1, keepdim=True)
        return logp, entropy, v

# ====================== ROLLOUT BUFFER =============================
# On-policy buffer storing transitions for PPO update
class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False

        # Pre-allocate tensors
        self.obs   = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.raw_a = torch.zeros((size, act_dim), dtype=torch.float32, device=device) # before tanh
        self.a     = torch.zeros((size, act_dim), dtype=torch.float32, device=device) # after tanh
        self.logp  = torch.zeros((size, 1),       dtype=torch.float32, device=device)
        self.v     = torch.zeros((size, 1),       dtype=torch.float32, device=device)
        self.r     = torch.zeros((size, 1),       dtype=torch.float32, device=device)
        self.done  = torch.zeros((size, 1),       dtype=torch.float32, device=device)

        # Filled after GAE computation
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
        """Generalized Advantage Estimation (GAE-λ)"""
        T = self.size if self.full else self.ptr
        adv = torch.zeros((T, 1), device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.done[t]
            next_v = last_v if t == T-1 else self.v[t+1]
            delta = self.r[t] + gamma * next_v * nonterminal - self.v[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        ret = adv + self.v[:T]
        self.adv[:T] = adv
        self.ret[:T] = ret

    def get(self, batch_size):
        """Yield shuffled mini-batches for PPO update."""
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

# ========================= CONFIGURATION ===========================
@dataclass
class PPOConfig:
    total_steps: int = 200_000
    rollout_steps: int = 8192        # Steps per PPO update
    update_epochs: int = 4           # Epochs over buffer
    mini_batch: int = 512            # Mini-batch size
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 1.0             # Value loss weight
    ent_coef_start: float = 0.01     # High → explore
    ent_coef_end: float = 0.001      # Low → exploit
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    use_lr_schedule: bool = True
    lr_min: float = 1e-5
    target_kl: float = 0.02          # Early stopping if KL too high
    normalize_rewards: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Network settings (matches proposal: 256D, 2 layers, tanh)
    hidden_size: int = 256
    num_layers: int = 2
    activation: str = "tanh"


# ====== FUNCTION 1: get_entropy_coef ======
def get_entropy_coef(current_step, total_steps, start_coef, end_coef):
    """
    Linear decay: encourages exploration early, exploitation late.
    Critical for discovering push-before-grasp sequences.
    """
    progress = min(1.0, current_step / total_steps)
    return start_coef + (end_coef - start_coef) * progress


# ====== FUNCTION 2: make_env ======
def make_env(render=True):#cong add
    """
    Create custom StrategicPushAndGraspEnv.
    Must expose:
      - Full state observability (as in proposal)
      - Action: 4D continuous vector
      - Reward: shaped as defined
    """
    render_mode = "human" if render else 'rgb_array'
    env = StrategicPushAndGraspEnv(render_mode=render_mode)
    return env


# ====== FUNCTION 3: ppo_train ======
def ppo_train(cfg: PPOConfig):
    """
    Main PPO training loop — fully on-policy, GAE, clipped objective.
    """
    # ====== Setup ======
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Create environment
    env = make_env(render=True)#cong add
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print("="*70)
    print("PPO TRAINING CONFIGURATION")
    print("="*70)
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    print(f"Device: {cfg.device}")
    print(f"Total steps: {cfg.total_steps:,}")
    print(f"Rollout steps: {cfg.rollout_steps:,}")
    print(f"Update epochs: {cfg.update_epochs}")
    print(f"Mini batch: {cfg.mini_batch}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Network: {cfg.num_layers} layers x {cfg.hidden_size} units ({cfg.activation})")
    print("="*70 + "\n")

    # ====== Initialize network and optimizer ======
    net = ActorCritic(
        obs_dim, 
        act_dim, 
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        activation=cfg.activation
    ).to(cfg.device)
    
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    
    # Learning rate scheduler
    scheduler = None
    if cfg.use_lr_schedule:
        scheduler = optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1.0,
            end_factor=cfg.lr_min / cfg.lr,
            total_iters=cfg.total_steps // cfg.rollout_steps
        )

    # Initialize reward normalizer
    reward_normalizer = None
    if cfg.normalize_rewards:
        reward_normalizer = RewardNormalizer(gamma=cfg.gamma)

    # ====== Initialize buffer ======
    buf = RolloutBuffer(cfg.rollout_steps, obs_dim, act_dim, cfg.device)

    # ====== Training metrics ======
    global_steps = 0
    episode_returns = []
    episode_lengths = []
    ep_ret, ep_len = 0.0, 0
    
    # Reset environment
    obs, _ = env.reset(seed=cfg.seed)
    start_time = time.time()

    print("Starting training...\n")

    # ====== Main training loop ======
    while global_steps < cfg.total_steps:
        # ====== Collect rollout ======
        net.eval()
        with torch.inference_mode():
            for step in range(cfg.rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                a, logp, v, mu, std, raw_a = net.act(obs_t)
                
                action = a.squeeze(0).cpu().numpy()
                obs_next, reward, terminated, truncated, info = env.step(action)
                
                # Normalize reward if enabled
                if reward_normalizer is not None:
                    reward_normalizer.update(reward)
                    reward = reward_normalizer.normalize(reward)
                
                ep_ret += reward
                ep_len += 1
                global_steps += 1
                
                done = terminated or truncated
                
                # Store transition
                buf.add(
                    obs_t.squeeze(0),
                    raw_a.squeeze(0),
                    a.squeeze(0),
                    logp.squeeze(0),
                    v.squeeze(0),
                    torch.tensor([reward], dtype=torch.float32, device=cfg.device),
                    torch.tensor([float(done)], dtype=torch.float32, device=cfg.device)
                )
                
                obs = obs_next
                
                # Episode done
                if done:
                    episode_returns.append(ep_ret)
                    episode_lengths.append(ep_len)
                    
                    if reward_normalizer is not None:
                        reward_normalizer.reset_episode()
                    
                    # Print episode info
                    elapsed = time.time() - start_time
                    recent_returns = episode_returns[-10:] if len(episode_returns) >= 10 else episode_returns
                    avg_recent = np.mean(recent_returns)
                    
                    print(f"[Episode {len(episode_returns):4d}] Steps={global_steps:6d} | "
                          f"Return={ep_ret:7.2f} | Length={ep_len:3d} | "
                          f"Avg10={avg_recent:7.2f} | Time={_fmt_secs(elapsed)}")
                    
                    ep_ret, ep_len = 0.0, 0
                    obs, _ = env.reset()
                    
                if global_steps >= cfg.total_steps:
                    break
        
        # ====== Compute GAE ======
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            _, _, last_v = net.forward(obs_t)
            last_v = last_v.squeeze()
        
        buf.compute_gae(last_v, gamma=cfg.gamma, lam=cfg.gae_lambda)

        # ====== Get current entropy coefficient (decays over training) ======
        current_ent_coef = get_entropy_coef(
            global_steps, 
            cfg.total_steps,
            cfg.ent_coef_start,
            cfg.ent_coef_end
        )

        # ====== PPO update (multiple epochs over mini-batches) ======
        net.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0
        
        early_stop = False
        for epoch in range(cfg.update_epochs):
            for mb_obs, mb_raw_a, _, mb_logp_old, _, mb_adv, mb_ret in buf.get(cfg.mini_batch):
                # FIXED: Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                new_logp, entropy, v = net.evaluate_actions(mb_obs, mb_raw_a)

                ratio = torch.exp(new_logp - mb_logp_old)
                
                # Check KL divergence for early stopping
                with torch.no_grad():
                    approx_kl = (mb_logp_old - new_logp).mean().item()
                    total_kl += approx_kl
                    
                    # FIXED: Early stopping should break both loops
                    if approx_kl > cfg.target_kl:
                        print(f"  ⚠️ Early stopping at epoch {epoch+1}/{cfg.update_epochs}, KL={approx_kl:.4f} > {cfg.target_kl}")
                        early_stop = True
                        break
                
                # Policy loss (clipped)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (mb_ret - v).pow(2).mean()

                # Entropy regularization (encourage exploration)
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
            
            # FIXED: Break outer loop if early stopping triggered
            if early_stop:
                break

        # ====== Step learning rate scheduler ======
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg.lr

        # Compute average losses
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0
        avg_kl = total_kl / num_updates if num_updates > 0 else 0

        print(f"[Update] Steps={global_steps:6d} | "
              f"PolicyLoss={avg_policy_loss:.4f} | ValueLoss={avg_value_loss:.4f} | "
              f"Entropy={avg_entropy:.4f} | KL={avg_kl:.4f} | "
              f"EntCoef={current_ent_coef:.4f} | LR={current_lr:.2e}")

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


# ====== Evaluation: run deterministic policy for several episodes ======
def evaluate_policy(net, make_env_fn, episodes=10, render=True, record_dir=None, seed=123):#cong add
    """
    Args:
        net: trained ActorCritic
        make_env_fn: function that returns a new env instance
        episodes: number of evaluation episodes
        render: whether to render visually
        record_dir: if provided, record videos using RecordVideo
        seed: random seed for evaluation
    Returns:
        stats: dict containing avg_return, avg_length, success_rate
    """
    from collections import defaultdict

    # Create fresh environment for evaluation
    env = make_env_fn(render=render)

    # Optional: video recording
    if record_dir is not None:
        try:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=record_dir,
                episode_trigger=lambda i: True,
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
            obs, _ = env.reset(seed=int(rng.randint(0, 1e9)))
            ep_ret, ep_len = 0.0, 0

            while True:
                # Deterministic action: use mean mu, with tanh squashing
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=next(net.parameters()).device).unsqueeze(0)
                mu, _, _v = net.forward(obs_t)
                a = torch.tanh(mu)  # deterministic = no sampling, just use mu
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
                    # Track success if info contains it
                    if isinstance(info, dict):
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
        rollout_steps=8192,
        update_epochs=4,
        mini_batch=512,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=1.0,
        ent_coef_start=0.01,
        ent_coef_end=0.001,
        lr=3e-4,
        max_grad_norm=0.5,
        use_lr_schedule=True,
        lr_min=1e-5,
        target_kl=0.02,
        normalize_rewards=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        hidden_size=256,
        num_layers=2,
        activation="tanh"   
    )
    net = ppo_train(cfg)

    # Post-training evaluation (no rendering, 10 episodes)
    print("\nStarting evaluation...")
    evaluate_policy(
        net,
        make_env_fn=lambda render=True: make_env(render), #cong add
        episodes=10,
        render=True, #cong add
        record_dir=None,  # set to "videos/eval_001" to record
        seed=123
    )
