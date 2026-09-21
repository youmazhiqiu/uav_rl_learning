"""Plot training metrics from an existing CSV without running training.

Example:
    python3 visualize_training.py \
        --input results/training_metrics_500_episodes_seed_0.csv \
        --output-dir results
"""

import argparse
import csv
import math
from pathlib import Path


WINDOW_SIZE = 50


def load_training_metrics(input_path):
    """Read one training run, keeping every episode in its original order."""
    episodes = []
    rewards = []
    steps = []
    successes = []

    with Path(input_path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"episode", "episode_reward", "episode_steps", "success"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing_columns)))

        for row_number, row in enumerate(reader, start=2):
            try:
                episode = int(row["episode"])
                reward = float(row["episode_reward"])
                episode_steps = int(row["episode_steps"])
                success_text = (row["success"] or "").strip().lower()

                if episode < 1 or episode_steps < 1 or not math.isfinite(reward):
                    raise ValueError("episode/steps must be positive and reward finite")
                if episodes and episode != episodes[-1] + 1:
                    raise ValueError("episode numbers must be consecutive and increasing")
                if success_text not in ("true", "false"):
                    raise ValueError("success must be True or False")
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV row {row_number}: {error}") from error

            episodes.append(episode)
            rewards.append(reward)
            steps.append(episode_steps)
            # 字符串 "False" 也会被 bool() 转成 True，因此要显式比较。
            successes.append(1 if success_text == "true" else 0)

    if not episodes:
        raise ValueError("The CSV contains no training episodes")

    return episodes, rewards, steps, successes


def moving_average(values, window_size=WINDOW_SIZE):
    """Average the current and preceding values, using no future episodes."""
    averages = []
    for index in range(len(values)):
        # 前 49 个点使用已有样本，不补零，也不丢弃这些 episode。
        start = max(0, index - window_size + 1)
        window = values[start:index + 1]
        averages.append(sum(window) / len(window))
    return averages


def save_curve(episodes, values, title, y_label, output_path, success_rate=False):
    """Save one figure; reward and steps also include the raw observations."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    if success_rate:
        trend_label = f"{WINDOW_SIZE}-episode rolling success rate"
        plt.ylim(0, 1)
    else:
        plt.plot(
            episodes, values, color="#9AA5B1", linewidth=1, alpha=0.65,
            label="Raw episodes",
        )
        trend_label = f"{WINDOW_SIZE}-episode moving average"

    plt.plot(
        episodes, moving_average(values), color="#1963A3", linewidth=2,
        label=trend_label, zorder=3, clip_on=False,
    )
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    if len(episodes) > 1:
        plt.xlim(episodes[0], episodes[-1])
    plt.grid(True, color="#D6D6D6", alpha=0.5)
    plt.legend(loc="best")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.figtext(
        0.5, 0.02,
        f"Trailing {WINDOW_SIZE}-episode window (includes current episode); "
        f"first {WINDOW_SIZE - 1} points use available history.",
        ha="center", fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved:", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot reward, steps and rolling success rate from a training CSV"
    )
    parser.add_argument("--input", required=True, type=Path, help="Training metrics CSV")
    parser.add_argument("--output-dir", default=Path("results"), type=Path)
    args = parser.parse_args()

    try:
        episodes, rewards, steps, successes = load_training_metrics(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    try:
        import matplotlib
    except ImportError:
        parser.error(
            "matplotlib is required. Use a Python environment with matplotlib, "
            "or install it there with: python3 -m pip install matplotlib"
        )
    # Agg 只保存图片，适用于没有图形界面的终端。
    matplotlib.use("Agg")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 保留原 CSV 文件名中的实验标识，例如 500_episodes_seed_0。
    suffix = args.input.stem.removeprefix("training_metrics_")
    run_label = suffix.replace("_", " ")
    curves = [
        ("reward", rewards, "Reward", "Episode reward", False),
        ("steps", steps, "Steps", "Episode steps", False),
        ("success_rate", successes, "Success Rate", "Success rate (0 to 1)", True),
    ]

    for name, values, title, y_label, success_rate in curves:
        save_curve(
            episodes, values,
            title=f"Q-learning Training {title}\n{run_label}",
            y_label=y_label,
            output_path=args.output_dir / f"{name}_curve_{suffix}.png",
            success_rate=success_rate,
        )


if __name__ == "__main__":
    main()
