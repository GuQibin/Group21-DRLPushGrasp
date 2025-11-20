"""Batch evaluation of PPO checkpoints.

Run with:
    python -m scripts.eval_checkpoints --checkpoint-dir checkpoints_run1
"""

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from .ppo_scratch import ActorCritic, PPOConfig, make_env, evaluate_policy


def _str2bool(val):
    if isinstance(val, bool):
        return val
    val = str(val).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from '{val}'")


def build_config(raw_cfg: Dict) -> PPOConfig:
    defaults = PPOConfig()
    cfg_data = defaults.__dict__.copy()
    if raw_cfg:
        cfg_data.update(raw_cfg)
    return PPOConfig(**cfg_data)


def _extract_step(path: Path) -> int:
    name = path.stem
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1


def collect_checkpoints(directory: Path) -> List[Path]:
    paths = []
    for path in directory.glob("*.pt"):
        if path.name.endswith("latest_weights.pt"):
            continue
        paths.append(path)
    paths.sort(key=_extract_step)
    return paths


def evaluate_checkpoint(path: Path, episodes: int, render: bool, device: str) -> Dict:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint {path} is missing the required state dict")

    cfg = build_config(ckpt.get("config", {}))
    current_lr = ckpt.get("learning_rate", cfg.lr)

    env = make_env(render=False, motion_scale=cfg.motion_scale)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    net = ActorCritic(
        obs_dim,
        act_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])

    stats = evaluate_policy(
        net,
        make_env_fn=lambda render=True: make_env(render, motion_scale=cfg.motion_scale),
        episodes=episodes,
        render=render,
        record_dir=None,
        seed=123,
    )

    return {
        "checkpoint": path.name,
        "global_steps": ckpt.get("global_steps", math.nan),
        "avg_return": stats.get("avg_return", math.nan),
        "avg_length": stats.get("avg_length", math.nan),
        "success_rate": stats.get("success_rate", math.nan),
        "episodes": stats.get("episodes", episodes),
        "motion_scale": cfg.motion_scale,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "learning_rate": current_lr,
    }


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(rows[-1])


def save_plot(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    steps = [row["global_steps"] for row in rows]
    returns = [row["avg_return"] for row in rows]
    success = [row["success_rate"] if row["success_rate"] is not None else math.nan for row in rows]

    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(steps, returns, marker="o")
    plt.ylabel("Average Return")
    plt.title("Checkpoint Evaluation")

    plt.subplot(2, 1, 2)
    plt.plot(steps, success, marker="o", color="green")
    plt.ylabel("Success Rate")
    plt.xlabel("Global Steps")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for PPO checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_run1"))
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per checkpoint")
    parser.add_argument(
        "--render",
        type=_str2bool,
        default=True,
        help="Whether to render the PyBullet window (default: True)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = Path(f"reports_{timestamp}")
    if args.output_csv is None:
        args.output_csv = default_dir / "checkpoint_eval.csv"
    if args.plot_path is None:
        args.plot_path = default_dir / "checkpoint_eval.png"

    checkpoints = collect_checkpoints(ckpt_dir)
    if not checkpoints:
        print("No checkpoint files were found")
        return

    results = []
    for ckpt_path in checkpoints:
        print(f"\n=== Evaluating {ckpt_path.name} ===")
        row = evaluate_checkpoint(
            ckpt_path,
            episodes=args.episodes,
            render=args.render,
            device=args.device,
        )
        print(
            f"steps={row['global_steps']} | avg_return={row['avg_return']:.2f} | avg_length={row['avg_length']:.1f} | success_rate={row['success_rate']}"
        )
        print(
            f"config: motion_scale={row['motion_scale']}, hidden={row['hidden_size']}, layers={row['num_layers']}"
        )
        results.append(row)
        save_csv(results, args.output_csv)

    results.sort(key=lambda r: r.get("global_steps") or 0)
    print(f"\nAppended results to {args.output_csv}")

    save_plot(results, args.plot_path)
    print(f"Plot saved to {args.plot_path}")


if __name__ == "__main__":
    main()
