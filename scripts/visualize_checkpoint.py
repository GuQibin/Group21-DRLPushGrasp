"""Load a trained PPO checkpoint and run a short rendered evaluation.

Example:
    python -m scripts.visualize_checkpoint
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import random

from .ppo_scratch import ActorCritic, PPOConfig, evaluate_policy, make_env


def _merge_config(saved_cfg) -> PPOConfig:
    """Apply saved config values over library defaults."""
    cfg = PPOConfig()
    data = cfg.__dict__.copy()
    if isinstance(saved_cfg, dict):
        data.update(saved_cfg)
    return PPOConfig(**data)


def load_policy(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint at {checkpoint} is missing model_state_dict")

    cfg = _merge_config(ckpt.get("config", {}))

    probe_env = make_env(render=False, motion_scale=cfg.motion_scale)
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.n
    probe_env.close()

    net = ActorCritic(
        obs_dim,
        act_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    return net, cfg


def main():
    parser = argparse.ArgumentParser(description="Render a single PPO checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_run2_lrdecay/ppo_step_10240.pt"),
        help="Path to the .pt checkpoint to load",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to run")
    parser.add_argument(
        "--render",
        type=lambda x: str(x).lower() not in {"0", "false", "no"},
        default=True,
        help="Open the PyBullet viewer window while evaluating",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the policy on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic eval")
    args = parser.parse_args()

    # Global seeding for reproducibility across Python, NumPy, and torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    net, cfg = load_policy(args.checkpoint, args.device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Using motion_scale={cfg.motion_scale}, hidden_size={cfg.hidden_size}, layers={cfg.num_layers}")
    print("Starting evaluation...")

    evaluate_policy(
        net,
        make_env_fn=lambda render=True: make_env(render, motion_scale=cfg.motion_scale),
        episodes=args.episodes,
        render=args.render,
        record_dir=None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
