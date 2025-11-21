import os
import glob
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from scripts.ppo_scratch import ActorCritic, make_env

# ==========================================
# CONFIGURATION (MUST MATCH TRAINING SCRIPT)
# ==========================================
CHECKPOINT_DIR = "checkpoints_run1"  # 队友设置的保存路径
EVAL_EPISODES = 5  # 每个点测5局取平均
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [CRITICAL] Network Architecture
HIDDEN_SIZE = 256
NUM_LAYERS = 4  # <--- 必须是 4，匹配队友的训练配置
ACTIVATION = "tanh"

# [CRITICAL] Environment Physics
MOTION_SCALE = 0.5  # <--- 必须是 0.5，匹配队友的训练配置


# ==========================================

def extract_step(filename):
    """从文件名 ppo_step_1024.pt 中提取数字 1024"""
    match = re.search(r"step_(\d+)", filename)
    return int(match.group(1)) if match else -1


def evaluate_ckpt(ckpt_path, env):
    """加载模型并评估"""
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

        # 1. 初始化网络 (使用硬编码的正确参数)
        obs_dim = env.observation_space.shape[0]
        # 处理 Discrete 空间 (act_dim = n)
        if hasattr(env.action_space, 'n'):
            act_dim = env.action_space.n
        else:
            act_dim = env.action_space.shape[0]

        net = ActorCritic(
            obs_dim,
            act_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,  # 关键：这里传入 4
            activation=ACTIVATION
        ).to(DEVICE)

        # 2. 加载权重
        # 兼容完整 Checkpoint 或 纯权重字典
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint)

        net.eval()

        # 3. 运行测试
        returns = []
        for _ in range(EVAL_EPISODES):
            obs, _ = env.reset()
            done = False
            ep_ret = 0
            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    # 获取 Logits
                    logits, _ = net(obs_t)
                    # 确定性选择 (Argmax)
                    action = torch.argmax(logits, dim=1).item()

                obs, r, terminated, truncated, _ = env.step(action)
                ep_ret += r
                done = terminated or truncated
            returns.append(ep_ret)

        return np.mean(returns), np.std(returns)

    except Exception as e:
        print(f"❌ Error evaluating {ckpt_path}: {e}")
        return None, None


def main():
    # 1. 找到所有 .pt 文件
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_step_*.pt"))
    if not files:
        print(f"No checkpoints found in {CHECKPOINT_DIR}!")
        return

    # 按步数排序
    files.sort(key=extract_step)

    steps = []
    means = []
    stds = []

    print(f"Found {len(files)} checkpoints. Starting evaluation on {DEVICE}...")
    print(f"Config: Layers={NUM_LAYERS}, MotionScale={MOTION_SCALE}")

    # 创建环境 (关闭渲染以加速，传入正确的 motion_scale)
    env = make_env(render=False, motion_scale=MOTION_SCALE)

    for f in files:
        step = extract_step(f)
        print(f"Testing step {step:<8}...", end="", flush=True)

        mean_ret, std_ret = evaluate_ckpt(f, env)

        if mean_ret is not None:
            print(f" Return: {mean_ret:.2f} (+/- {std_ret:.2f})")
            steps.append(step)
            means.append(mean_ret)
            stds.append(std_ret)
        else:
            print(" Skipped.")

    env.close()

    # 2. 绘图
    if not steps:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, marker='o', label='Mean Return', color='b')

    # 绘制标准差阴影
    means = np.array(means)
    stds = np.array(stds)
    plt.fill_between(steps, means - stds, means + stds, alpha=0.2, color='b', label='Std Dev')

    plt.xlabel('Training Steps')
    plt.ylabel('Episode Return (Raw)')
    plt.title(f'Training Progress (Layers={NUM_LAYERS}, Scale={MOTION_SCALE})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_path = "evaluation_curve.png"
    plt.savefig(save_path)
    print(f"\n✅ Curve saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()