"""Animate two saved greedy-policy trajectories, without running training.

Example (from the project directory):
    MPLCONFIGDIR=.venv/.matplotlib .venv/bin/python animate_trajectories.py \
        --left results/trajectory_100_episodes_seed_0.csv \
        --right results/trajectory_500_episodes_seed_0.csv \
        --output results/trajectory_comparison_seed_0.gif

Requires matplotlib and Pillow. Also saves <output_stem>_final.png.
"""

import argparse
import csv
import math
from pathlib import Path


def load_trajectory(file_path):
    """Read the step-0 initial state and each subsequent state in CSV order."""
    trajectory = []
    coordinates = ("x", "y", "goal_x", "goal_y")

    with file_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"step", *coordinates}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{file_path}: missing columns {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            try:
                state = {"step": int(row["step"])}
                for name in coordinates:
                    state[name] = float(row[name])
                    if not math.isfinite(state[name]):
                        raise ValueError(f"{name} must be finite")

                if state["step"] != len(trajectory):
                    raise ValueError("steps must start at 0 and increase by 1")
                if trajectory and (
                    state["goal_x"] != trajectory[0]["goal_x"]
                    or state["goal_y"] != trajectory[0]["goal_y"]
                ):
                    raise ValueError("the goal must stay fixed throughout the trajectory")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{file_path}, row {row_number}: {error}") from error

            trajectory.append(state)

    if not trajectory:
        raise ValueError(f"{file_path}: no trajectory rows")
    return trajectory


def final_result(trajectory):
    """Use the same success criterion as test_policy.py: distance < 1."""
    last = trajectory[-1]
    distance = math.hypot(last["x"] - last["goal_x"], last["y"] - last["goal_y"])
    return "Success" if distance < 1 else "Fail"


def save_comparison(left, right, output_path, fps):
    """Advance both panels by one step per frame, holding completed panels."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    trajectories = [left, right]
    titles = ["100 episodes", "500 episodes"]
    colors = ["#C65A27", "#166EAD"]

    # 两侧统一坐标范围，并包含所有轨迹点和目标点。
    all_states = left + right
    all_x = [state[name] for state in all_states for name in ("x", "goal_x")]
    all_y = [state[name] for state in all_states for name in ("y", "goal_y")]
    x_limits = (min(all_x) - 1, max(all_x) + 1)
    y_limits = (min(all_y) - 1, max(all_y) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.25, top=0.81, wspace=0.18)
    fig.suptitle("UAV greedy-policy trajectory comparison", fontsize=17, y=0.97)
    fig.text(
        0.5, 0.91, "One recorded step per frame | Completed UAVs hold their final position",
        ha="center", fontsize=10, color="#555555",
    )
    panels = []

    for ax, trajectory, title, color in zip(axes, trajectories, titles, colors):
        start = trajectory[0]
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)

        # 起点用空心方框，目标用星形，避免与当前 UAV 圆点混淆。
        ax.plot(
            start["x"], start["y"], marker="s", markersize=13,
            markerfacecolor="none", markeredgecolor="#333333",
            markeredgewidth=2, linestyle="none", zorder=4,
        )
        ax.annotate("Start", (start["x"], start["y"]), xytext=(9, -18),
                    textcoords="offset points", fontsize=10)
        ax.plot(
            start["goal_x"], start["goal_y"], marker="*", markersize=23,
            color="#E8B923", markeredgecolor="#826600", linestyle="none", zorder=4,
        )
        ax.annotate("Goal", (start["goal_x"], start["goal_y"]), xytext=(9, 10),
                    textcoords="offset points", fontsize=10)

        line, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.75, zorder=2)
        uav, = ax.plot(
            [], [], marker="o", markersize=10, color=color,
            markeredgecolor="white", markeredgewidth=1.5, linestyle="none", zorder=5,
        )
        info = ax.text(0, -0.22, "", transform=ax.transAxes, fontsize=11, linespacing=1.6)
        result = ax.text(
            0, -0.34, "", transform=ax.transAxes, fontsize=12, fontweight="bold",
        )
        panels.append({
            "trajectory": trajectory,
            "x": [state["x"] for state in trajectory],
            "y": [state["y"] for state in trajectory],
            "line": line,
            "uav": uav,
            "info": info,
            "result": result,
            "outcome": final_result(trajectory),
        })

    def update(frame):
        for panel in panels:
            trajectory = panel["trajectory"]
            # 较短的轨迹结束后，索引一直停在最后一个状态。
            index = min(frame, len(trajectory) - 1)
            state = trajectory[index]
            panel["line"].set_data(panel["x"][:index + 1], panel["y"][:index + 1])
            panel["uav"].set_data([state["x"]], [state["y"]])
            panel["info"].set_text(
                f"Step: {state['step']} / {trajectory[-1]['step']}\n"
                f"UAV position: ({state['x']:g}, {state['y']:g})"
            )
            if index == len(trajectory) - 1:
                panel["result"].set_text(
                    f"{panel['outcome']} | total steps: {state['step']}"
                )
                panel["result"].set_color(
                    "#26733D" if panel["outcome"] == "Success" else "#B13D2D"
                )
            else:
                panel["result"].set_text("Running")
                panel["result"].set_color("#555555")

    last_step = max(len(left), len(right)) - 1
    # 最终状态共展示 2 秒（包括第一次到达最终状态的那一帧）。
    frame_count = last_step + 2 * fps
    animation = FuncAnimation(
        fig, update, frames=range(frame_count), interval=1000 / fps,
        repeat=True, blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image = output_path.with_name(output_path.stem + "_final.png")
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
        update(last_step)
        fig.savefig(final_image, dpi=150)
    finally:
        plt.close(fig)

    print("GIF saved to:", output_path)
    print("Final comparison saved to:", final_image)
    for title, trajectory in zip(titles, trajectories):
        print(f"{title}: {final_result(trajectory)}, total steps: {trajectory[-1]['step']}")


def main():
    parser = argparse.ArgumentParser(description="Animate 100 vs 500 episode trajectories")
    parser.add_argument(
        "--left", type=Path, default=Path("results/trajectory_100_episodes_seed_0.csv"),
        help="CSV for the left panel (100 episodes)",
    )
    parser.add_argument(
        "--right", type=Path, default=Path("results/trajectory_500_episodes_seed_0.csv"),
        help="CSV for the right panel (500 episodes)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/trajectory_comparison_seed_0.gif"),
        help="Output GIF; also saves a sibling *_final.png",
    )
    parser.add_argument("--fps", type=int, default=5, help="Playback steps per second (1-50)")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".gif":
        parser.error("--output must have a .gif extension")
    if not 1 <= args.fps <= 50:
        parser.error("--fps must be between 1 and 50")

    try:
        left = load_trajectory(args.left)
        right = load_trajectory(args.right)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    try:
        import matplotlib
        from PIL import Image  # Pillow is required by matplotlib's GIF writer.
    except ImportError:
        parser.error("matplotlib and Pillow are required; use a Python environment with both")
    # Agg 适用于 WSL/无桌面的终端，保存动画时无需弹出窗口。
    matplotlib.use("Agg")
    save_comparison(left, right, args.output, args.fps)


if __name__ == "__main__":
    main()
