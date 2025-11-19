# scripts/multi_env_ppo.py
import os
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from envs.strategic_env import StrategicPushAndGraspEnv
from scripts.ppo_scratch import ActorCritic, RewardNormalizer, _fmt_secs, get_entropy_coef

class ParallelEnv:
    def __init__(self, num_envs, make_env_fn):
        self.envs = [make_env_fn(render=False) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obses, infos = [], []
        for idx, env in enumerate(self.envs):
            obs, info = env.reset(None if seed is None else seed + idx)
            obses.append(obs)
            infos.append(info)
        return np.stack(obses), infos

    def step(self, actions):
        obses, rews, terms, truncs, infos = [], [], [], [], []
        for idx, (env, act) in enumerate(zip(self.envs, actions)):
            try:
                ob, r, term, trunc, info = env.step(act)
            except Exception as e:
                # Reset env on error and treat as truncated episode
                print(f"[ParallelEnv] Env {idx} error: {e}. Resetting.")
                ob, info = env.reset()
                r, term, trunc = 0.0, True, True
                info = info or {}
                info["reset_due_to_error"] = True
            obses.append(ob)
            rews.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        return (np.stack(obses),
                np.asarray(rews, dtype=np.float32),
                np.asarray(terms, dtype=np.bool_),
                np.asarray(truncs, dtype=np.bool_),
                infos)

    def close(self):
        for env in self.envs:
            env.close()

class VectorRolloutBuffer:
    def __init__(self, rollout_steps, num_envs, obs_dim, act_dim, device):
        self.capacity = rollout_steps * num_envs
        self.device = device
        self.ptr = 0

        self.obs = torch.zeros((self.capacity, obs_dim), device=device)
        self.raw_a = torch.zeros((self.capacity, act_dim), device=device)
        self.a = torch.zeros((self.capacity, act_dim), device=device)
        self.logp = torch.zeros((self.capacity, 1), device=device)
        self.v = torch.zeros((self.capacity, 1), device=device)
        self.r = torch.zeros((self.capacity, 1), device=device)
        self.done = torch.zeros((self.capacity, 1), device=device)
        self.adv = torch.zeros((self.capacity, 1), device=device)
        self.ret = torch.zeros((self.capacity, 1), device=device)

    def add_batch(self, obs, raw_a, a, logp, v, reward, done):
        bsz = obs.shape[0]
        idx = slice(self.ptr, self.ptr + bsz)
        self.obs[idx] = obs
        self.raw_a[idx] = raw_a
        self.a[idx] = a
        self.logp[idx] = logp
        self.v[idx] = v
        self.r[idx] = reward
        self.done[idx] = done
        self.ptr += bsz

    def compute_gae(self, last_v, gamma, lam):
        T = self.ptr
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

    def get(self, mini_batch):
        T = self.ptr
        idx = torch.randperm(T, device=self.device)
        for start in range(0, T, mini_batch):
            mb = idx[start:start + mini_batch]
            yield (self.obs[mb], self.raw_a[mb], self.logp[mb],
                   self.v[mb], self.adv[mb], self.ret[mb])

    def reset(self):
        self.ptr = 0

@dataclass
class VecPPOConfig:
    total_steps: int = 50_000
    rollout_steps: int = 512
    num_envs: int = 64
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
    target_kl: float = 0.02
    normalize_rewards: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "checkpoints_vec"
    checkpoint_interval: int = 1_000
    seed: int = 42
    hidden_size: int = 256
    num_layers: int = 3
    activation: str = "tanh"

def make_env(render=False):
    render_mode = "human" if render else "rgb_array"
    return StrategicPushAndGraspEnv(render_mode=render_mode)

def ppo_train_vec(cfg: VecPPOConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    vec_env = ParallelEnv(cfg.num_envs, make_env)
    obs_dim = vec_env.single_observation_space.shape[0]
    act_dim = vec_env.single_action_space.shape[0]

    net = ActorCritic(obs_dim, act_dim,
                      hidden_size=cfg.hidden_size,
                      num_layers=cfg.num_layers,
                      activation=cfg.activation).to(cfg.device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    reward_norm = RewardNormalizer(gamma=cfg.gamma) if cfg.normalize_rewards else None
    buf = VectorRolloutBuffer(cfg.rollout_steps, cfg.num_envs, obs_dim, act_dim, cfg.device)

    obs, _ = vec_env.reset(cfg.seed)
    start_time = time.time()
    global_steps = 0
    last_ckpt = 0
    episode_returns = []

    while global_steps < cfg.total_steps:
        net.eval()
        buf.reset()

        for _ in range(cfg.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device)
            a, logp, v, _, _, raw_a = net.act(obs_t)
            actions = a.detach().cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)

            real_done = terminated.astype(np.float32)
            # 不管 truncated，采样步都丢进 buffer，但 done 只标记真实终止
            buf.add_batch(
                obs_t.detach(),
                raw_a.detach(),
                a.detach(),
                logp.detach(),
                v.detach(),
                torch.as_tensor(rewards[:, None], dtype=torch.float32, device=cfg.device),
                torch.as_tensor(real_done[:, None], dtype=torch.float32, device=cfg.device)
            )

            if reward_norm is not None:
                for r in rewards:
                    reward_norm.update(r)

            obs = next_obs
            global_steps += cfg.num_envs

            for env_idx, done in enumerate(terminated | truncated):
                if done:
                    if reward_norm is not None:
                        reward_norm.reset_episode()
                    ret = infos[env_idx].get("episode_return")
                    if ret is not None:
                        episode_returns.append(ret)
                    obs_single, _ = vec_env.envs[env_idx].reset()
                    obs[env_idx] = obs_single

        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device)
            _, _, last_v = net.forward(obs_t)
            # 用批量平均的 value 作为 bootstrap
            buf.compute_gae(last_v.mean(dim=0), cfg.gamma, cfg.gae_lambda)

        ent_coef = get_entropy_coef(global_steps, cfg.total_steps,
                                    cfg.ent_coef_start, cfg.ent_coef_end)

        net.train()
        total_pol, total_val, total_ent, total_kl, updates = 0, 0, 0, 0, 0
        for _ in range(cfg.update_epochs):
            for mb_obs, mb_raw, mb_logp_old, mb_v_old, mb_adv, mb_ret in buf.get(cfg.mini_batch):
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_logp, entropy, v_pred = net.evaluate_actions(mb_obs, mb_raw)
                ratio = torch.exp(new_logp - mb_logp_old)
                approx_kl = (mb_logp_old - new_logp).mean().item()
                total_kl += approx_kl
                if approx_kl > cfg.target_kl:
                    break

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb_ret - v_pred).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + ent_coef * entropy_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

                total_pol += policy_loss.item()
                total_val += value_loss.item()
                total_ent += entropy.mean().item()
                updates += 1
            else:
                continue
            break

        if cfg.save_dir and global_steps - last_ckpt >= cfg.checkpoint_interval:
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "global_steps": global_steps,
                    "config": cfg.__dict__,
                },
                os.path.join(cfg.save_dir, f"vec_ppo_step_{global_steps}.pt"),
            )
            torch.save(net.state_dict(),
                       os.path.join(cfg.save_dir, "vec_ppo_latest.pt"))
            last_ckpt = global_steps
            print(f"[Checkpoint] saved at step {global_steps}")

        avg_pol = total_pol / max(updates, 1)
        avg_val = total_val / max(updates, 1)
        avg_ent = total_ent / max(updates, 1)
        avg_kl = total_kl / max(updates, 1)
        print(f"[Update] steps={global_steps:6d} | Policy={avg_pol:.4f} "
              f"| Value={avg_val:.4f} | Ent={avg_ent:.4f} | KL={avg_kl:.4f} "
              f"| EntCoef={ent_coef:.4f} | time={_fmt_secs(time.time()-start_time)}")

    vec_env.close()
    return net

def evaluate_policy(net, episodes=5, render=True):
    env = make_env(render=render)
    net.eval()
    with torch.inference_mode():
        for ep in range(episodes):
            obs, _ = env.reset()
            ep_ret = 0.0
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=next(net.parameters()).device).unsqueeze(0)
                mu, _, _ = net.forward(obs_t)
                action = torch.tanh(mu).squeeze(0).cpu().numpy()
                obs, r, terminated, truncated, info = env.step(action)
                ep_ret += r
                if render:
                    env.render()
                if terminated or truncated:
                    print(f"[Eval] Episode {ep+1}: return={ep_ret:.2f}")
                    break
    env.close()

if __name__ == "__main__":
    cfg = VecPPOConfig()
    policy = ppo_train_vec(cfg)
    evaluate_policy(policy, episodes=3, render=True)
