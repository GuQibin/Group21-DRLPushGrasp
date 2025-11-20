# Run with "python -m scripts.ppo_scratch"

# ===================================================================
# PPO FROM SCRATCH — ANNOTATED FOR ME5418 PROJECT (GROUP 21)
# HYBRID CONTROL: RL learns Discrete 8D Push Direction
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
from torch.distributions import Normal, Categorical  # MODIFIED: Added Categorical

import gymnasium as gym


# --------------------- UTILITY: Time Formatting ---------------------
def _fmt_secs(s: float) -> str:
    s = max(0, float(s))
    h = int(s // 3600);
    s -= 3600 * h
    m = int(s // 60);
    s -= 60 * m
    return f"{h:02d}:{m:02d}:{int(s):02d}"


def _print_training_progress(global_steps: int,
                             total_steps: int,
                             start_time: float,
                             current_episode: int,
                             episode_step: int,
                             episode_max_steps: int):
    """
    Utility printer for training progress (steps, ETA, and episode progress).
    """
    total_steps = max(total_steps, 1)
    progress = min(max(global_steps / total_steps, 0.0), 1.0)
    bar_width = 30
    filled = int(progress * bar_width)
    bar = "=" * filled + "." * (bar_width - filled)

    elapsed = max(time.time() - start_time, 1e-6)
    if global_steps > 0:
        steps_per_sec = global_steps / elapsed
        remaining_steps = max(total_steps - global_steps, 0)
        eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else float("inf")
    else:
        eta = float("inf")

    eta_text = "??:??:??" if not math.isfinite(eta) else _fmt_secs(eta)

    if episode_max_steps and episode_max_steps > 0:
        episode_progress = f"{episode_step}/{episode_max_steps}"
    else:
        episode_progress = f"{episode_step}"

    print(
        f"[Progress] [{bar}] {progress * 100:5.1f}% | "
        f"Step {global_steps}/{total_steps} | ETA {eta_text} | "
        f"Episode {current_episode} ({episode_progress})",
        flush=True
    )


# ====== IMPORT YOUR CUSTOM ENVIRONMENT ==============================
from envs.strategic_env import StrategicPushAndGraspEnv


# ==================== REWARD NORMALIZATION =========================
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
        self.var += delta * delta2  # incremental variance

    def normalize(self, reward):
        """Z-score normalize reward using running stats."""
        std = np.sqrt(self.var / self.count)
        return reward / (std + 1e-8)

    def reset_episode(self):
        """Reset episode-level return tracking"""
        self.returns = 0.0


# ============ ACTION SQUASHING + LOG PROB CORRECTION ===============
# REMOVED: No longer needed as action space is purely discrete.


# ===================== ACTOR-CRITIC NETWORK ========================
class ActorCritic(nn.Module):
    # act_dim is now 8 (for 8 discrete push directions)
    def __init__(self, obs_dim, act_dim, hidden_size=256, num_layers=2, activation="tanh"):
        super().__init__()

        # Activation factory (remains unchanged)
        def _act_factory(name: str):
            name = name.lower()
            if name == "tanh": return nn.Tanh
            if name == "relu": return nn.ReLU
            if name == "leakyrelu": return lambda: nn.LeakyReLU(negative_slope=0.01)
            if name == "elu": return nn.ELU
            return nn.Tanh

        Act = _act_factory(activation)

        # Shared backbone (remains unchanged)
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(Act())
            in_dim = hidden_size
        self.backbone = nn.Sequential(*layers)

        # MODIFIED: Policy head outputs logits for Categorical distribution
        self.policy_head = nn.Linear(hidden_size, act_dim)

        # Value head: V(s) (remains unchanged)
        self.v_head = nn.Linear(hidden_size, 1)

        # Xavier init for stability (remains unchanged)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        # MODIFIED: Returns logits and V
        h = self.backbone(obs)
        logits = self.policy_head(h)
        v = self.v_head(h)
        return logits, v

    def act(self, obs):
        """Sample discrete action (index 0-7) during rollout."""
        # MODIFIED: Uses Categorical distribution
        logits, v = self.forward(obs)

        dist = Categorical(logits=logits)
        a = dist.sample()  # a is the discrete action index tensor (e.g., tensor([3]))

        # logp shape: (batch_size, 1)
        logp = dist.log_prob(a).unsqueeze(-1)

        # raw_a: The action index stored as a float tensor (N, 1)
        raw_a = a.float().unsqueeze(-1)

        # Return the action index as a scalar integer (needed for env.step)
        # We don't need mu, std anymore
        return a.squeeze(0).cpu().numpy().item(), logp, v, raw_a

    def evaluate_actions(self, obs, raw_actions):
        """Recompute logp, entropy, v for PPO update (given old raw action indices)."""
        # MODIFIED: Uses Categorical distribution
        logits, v = self.forward(obs)

        # raw_actions is (N, 1) float tensor. Needs to be (N) long tensor for Categorical.
        actions_long = raw_actions.squeeze(-1).long()

        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions_long).unsqueeze(-1)

        # Entropy of the discrete distribution
        entropy = dist.entropy().unsqueeze(-1)

        # No Tanh correction is needed
        return logp, entropy, v


# ====================== ROLLOUT BUFFER =============================
class RolloutBuffer:
    # MODIFIED: act_dim for storage is fixed to 1 (for discrete index)
    def __init__(self, size, obs_dim, act_dim, device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False

        # Action storage dimension is 1 (to store the discrete index)
        act_storage_dim = 1

        # Pre-allocate tensors
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        # MODIFIED: Size is (size, 1)
        self.raw_a = torch.zeros((size, act_storage_dim), dtype=torch.float32, device=device)  # action index
        self.a = torch.zeros((size, act_storage_dim), dtype=torch.float32, device=device)  # action index copy
        self.logp = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.v = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.r = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.done = torch.zeros((size, 1), dtype=torch.float32, device=device)

        # Filled after GAE computation (remains unchanged)
        self.adv = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.ret = torch.zeros((size, 1), dtype=torch.float32, device=device)

    def add(self, obs, raw_a, a, logp, v, r, done):
        self.obs[self.ptr] = obs
        self.raw_a[self.ptr] = raw_a
        self.a[self.ptr] = a
        self.logp[self.ptr] = logp
        self.v[self.ptr] = v
        self.r[self.ptr] = r
        self.done[self.ptr] = done
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True
            self.ptr = 0

    # compute_gae (remains unchanged)
    def compute_gae(self, last_v, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation (GAE-λ)"""
        T = self.size if self.full else self.ptr
        adv = torch.zeros((T, 1), device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.done[t]
            next_v = last_v if t == T - 1 else self.v[t + 1]
            delta = self.r[t] + gamma * next_v * nonterminal - self.v[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        ret = adv + self.v[:T]
        self.adv[:T] = adv
        self.ret[:T] = ret

    # get (remains unchanged)
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
    # ... (All config parameters remain unchanged)
    total_steps: int = 200_000
    rollout_steps: int = 8192
    update_epochs: int = 4
    mini_batch: int = 512
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
    motion_scale: float = 1.0  # <1.0 speeds up env by shortening controller rollouts

    # Network settings (remains unchanged)
    hidden_size: int = 256
    num_layers: int = 2
    activation: str = "tanh"

    # Checkpoint settings (remains unchanged)
    save_dir: str = "checkpoints"
    checkpoint_interval: int = 5_000


# ====== FUNCTION 1: get_entropy_coef ======
def get_entropy_coef(current_step, total_steps, start_coef, end_coef):
    """Linear decay: encourages exploration early, exploitation late."""
    progress = min(1.0, current_step / total_steps)
    return start_coef + (end_coef - start_coef) * progress


# ====== FUNCTION 2: make_env ======
def make_env(render=True, motion_scale=1.0):
    """
    Create custom StrategicPushAndGraspEnv.
    """
    render_mode = "human" if render else 'rgb_array'
    env = StrategicPushAndGraspEnv(render_mode=render_mode, motion_scale=motion_scale)
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
    env = make_env(render=False, motion_scale=cfg.motion_scale)
    obs_dim = env.observation_space.shape[0]
    # MODIFIED: Action space is Discrete(8), so act_dim is 8 (for logits)
    act_dim = env.action_space.n

    print("=" * 70)
    print("PPO TRAINING CONFIGURATION (DISCRETE ACTION SPACE)")
    print("=" * 70)
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim (Logits Size): {act_dim}")
    print(f"Device: {cfg.device}")
    print(f"Motion scale: {cfg.motion_scale:.2f}")
    # ... (rest of print config remains unchanged)

    # ====== Initialize network and optimizer ======
    net = ActorCritic(
        obs_dim,
        act_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        activation=cfg.activation
    ).to(cfg.device)

    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    # Learning rate scheduler (remains unchanged)
    scheduler = None
    if cfg.use_lr_schedule:
        scheduler = optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1.0,
            end_factor=cfg.lr_min / cfg.lr,
            total_iters=cfg.total_steps // cfg.rollout_steps
        )

    # Initialize reward normalizer (remains unchanged)
    reward_normalizer = None
    if cfg.normalize_rewards:
        reward_normalizer = RewardNormalizer(gamma=cfg.gamma)

    # ====== Initialize buffer ======
    # Note: RolloutBuffer's internal action storage size is now 1D
    buf = RolloutBuffer(cfg.rollout_steps, obs_dim, act_dim, cfg.device)

    # ====== Training metrics (remains unchanged) ======
    global_steps = 0
    episode_returns = []
    episode_lengths = []
    ep_ret, ep_len = 0.0, 0

    # Reset environment
    obs, _ = env.reset(seed=cfg.seed)
    start_time = time.time()
    last_ckpt_step = 0
    progress_log_interval = max(100, cfg.rollout_steps // 4)
    last_progress_print = 0

    print("Starting training...\n")

    # ====== Main training loop ======
    while global_steps < cfg.total_steps:
        # ====== Collect rollout ======
        net.eval()
        with torch.inference_mode():
            for step in range(cfg.rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)

                # MODIFIED: net.act returns the integer index action_idx
                action_idx, logp, v, raw_a = net.act(obs_t)

                # action_idx is the discrete integer (0-7) expected by env.step()
                obs_next, reward, terminated, truncated, info = env.step(action_idx)

                # Normalize reward if enabled (remains unchanged)
                if reward_normalizer is not None:
                    reward_normalizer.update(reward)
                    reward = reward_normalizer.normalize(reward)

                ep_ret += reward
                ep_len += 1
                global_steps += 1

                done = terminated or truncated

                if (global_steps - last_progress_print) >= progress_log_interval or global_steps >= cfg.total_steps:
                    episode_max_steps = getattr(env, "max_episode_steps", 0)
                    _print_training_progress(
                        global_steps,
                        cfg.total_steps,
                        start_time,
                        len(episode_returns) + 1,
                        ep_len,
                        episode_max_steps
                    )
                    last_progress_print = global_steps

                # Store transition
                # Note: For discrete PPO, we store the action index tensor (raw_a)
                # in both raw_a and a slots in the buffer.
                buf.add(
                    obs_t.squeeze(0),
                    raw_a.squeeze(0),  # Action index as float tensor
                    raw_a.squeeze(0),  # Action index as float tensor (copy)
                    logp.squeeze(0),
                    v.squeeze(0),
                    torch.tensor([reward], dtype=torch.float32, device=cfg.device),
                    torch.tensor([float(done)], dtype=torch.float32, device=cfg.device)
                )

                obs = obs_next

                # ... (rest of rollout loop remains unchanged)
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

        # ... (Compute GAE and PPO update loops remain unchanged, using raw_a as the action index)

        # ====== Compute GAE ======
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            _, last_v = net.forward(obs_t)
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
                # Normalize advantages (remains unchanged)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_logp, entropy, v = net.evaluate_actions(mb_obs, mb_raw_a)

                ratio = torch.exp(new_logp - mb_logp_old)

                # Check KL divergence for early stopping (remains unchanged)
                with torch.no_grad():
                    # For discrete actions, use log(P_old / P_new) approximation
                    approx_kl = (mb_logp_old - new_logp).mean().item()
                    total_kl += approx_kl

                    if approx_kl > cfg.target_kl:
                        print(
                            f"  ⚠️ Early stopping at epoch {epoch + 1}/{cfg.update_epochs}, KL={approx_kl:.4f} > {cfg.target_kl}")
                        early_stop = True
                        break

                # Policy loss (clipped)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (remains unchanged)
                value_loss = 0.5 * (mb_ret - v).pow(2).mean()

                # Entropy regularization
                entropy_loss = -entropy.mean()

                # Total loss with current entropy coefficient
                loss = policy_loss + cfg.vf_coef * value_loss + current_ent_coef * entropy_loss

                # Optimization step (remains unchanged)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

                # Track metrics (remains unchanged)
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

            if early_stop:
                break

        # ... (rest of update tracking and checkpointing remains unchanged)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg.lr

        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0
        avg_kl = total_kl / num_updates if num_updates > 0 else 0

        print(f"[Update] Steps={global_steps:6d} | "
              f"PolicyLoss={avg_policy_loss:.4f} | ValueLoss={avg_value_loss:.4f} | "
              f"Entropy={avg_entropy:.4f} | KL={avg_kl:.4f} | "
              f"EntCoef={current_ent_coef:.4f} | LR={current_lr:.2e}")

        if cfg.save_dir and (global_steps - last_ckpt_step) >= cfg.checkpoint_interval:
            os.makedirs(cfg.save_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.save_dir, f"ppo_step_{global_steps}.pt")
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "global_steps": global_steps,
                    "config": cfg.__dict__,
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "learning_rate": current_lr,
                },
                ckpt_path,
            )
            torch.save(
                net.state_dict(),
                os.path.join(cfg.save_dir, "ppo_latest_weights.pt"),
            )
            print(f"[Checkpoint] Saved to {ckpt_path}")
            last_ckpt_step = global_steps

    env.close()

    # ... (Final statistics remain unchanged)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    if len(episode_returns) > 0:
        print(f"Total episodes: {len(episode_returns)}")
        print(f"Final 10 episodes avg return: {np.mean(episode_returns[-10:]):.2f}")
        print(f"Best episode return: {np.max(episode_returns):.2f}")
        print(f"Final 10 episodes avg length: {np.mean(episode_lengths[-10:]):.1f}")
    print("=" * 70 + "\n")

    return net


# ====== Evaluation: run deterministic policy for several episodes ======
def evaluate_policy(net, make_env_fn, episodes=10, render=True, record_dir=None, seed=123):
    """
    Args:
        net: trained ActorCritic
        ...
    """
    from collections import defaultdict

    env = make_env_fn(render=render)

    if record_dir is not None:
        try:
            # ... (video recording logic remains unchanged)
            pass
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
                # MODIFIED: Deterministic action uses the argmax (most likely) logit
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=next(net.parameters()).device).unsqueeze(0)
                logits, _v = net.forward(obs_t)
                # Deterministic action: Argmax of logits gives the discrete index
                action_idx = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().item()

                obs, r, terminated, truncated, info = env.step(action_idx)
                ep_ret += r
                ep_len += 1

                if render and hasattr(env, "render"):
                    try:
                        env.render()
                    except NotImplementedError:
                        pass

                done = terminated or truncated
                if done:
                    returns.append(ep_ret)
                    lengths.append(ep_len)
                    # ... (success tracking remains unchanged)
                    if isinstance(info, dict):
                        for key in ("success", "is_success"):
                            if key in info:
                                succ_total += 1
                                succ_cnt += 1 if (float(info[key]) > 0.5) else 0
                                break
                    print(f"[Eval Episode {ep + 1}/{episodes}] return={ep_ret:.2f}, len={ep_len}")
                    break

    env.close()
    # ... (rest of summary stats remains unchanged)

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
        total_steps=20_000,
        rollout_steps=1024,
        update_epochs=3,
        mini_batch=256,
        #device="cuda",
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
        num_layers=4,
        activation="tanh",
        save_dir="checkpoints_run1",
        checkpoint_interval=1_000,
        motion_scale=0.5,
    )
    net = ppo_train(cfg)

    # Post-training evaluation (no rendering, 10 episodes)
    print("\nStarting evaluation...")
    evaluate_policy(
        net,
        make_env_fn=lambda render=True: make_env(render, motion_scale=cfg.motion_scale),
        episodes=10,
        render=True,
        record_dir=None,
        seed=123
    )
