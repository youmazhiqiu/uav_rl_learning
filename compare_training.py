import argparse
import csv
from pathlib import Path

from env.uav_env import UAVEnv
from test_policy import evaluate_policy
from train import save_q_table, save_training_metrics, train_agent


EPISODE_COUNTS = [100, 500]


def save_trajectory(trajectory, file_path):
    """Save a state trajectory in a simple row-per-step CSV file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "x", "y", "goal_x", "goal_y"])

        for step, state in enumerate(trajectory):
            writer.writerow([step] + state)


def save_evaluation_summary(results, file_path):
    """Save one evaluation row for each training length."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "training_episodes",
        "seed",
        "success",
        "steps",
        "total_reward",
        "final_distance",
        "training_metrics_file",
        "trajectory_file",
        "q_table_file",
    ]

    with file_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_comparison(seed, output_directory):
    """Compare independent 100- and 500-episode runs using one seed."""
    output_directory = Path(output_directory)
    results = []
    training_runs = {}

    for training_episodes in EPISODE_COUNTS:
        # Each run starts from a new agent with the same configured seed.
        agent, training_metrics = train_agent(training_episodes, seed)
        training_runs[training_episodes] = training_metrics

        file_suffix = f"{training_episodes}_episodes_seed_{seed}"
        metrics_file = output_directory / f"training_metrics_{file_suffix}.csv"
        trajectory_file = output_directory / f"trajectory_{file_suffix}.csv"
        q_table_file = output_directory / f"q_table_{file_suffix}.pkl"

        save_training_metrics(training_metrics, metrics_file)
        save_q_table(agent, q_table_file)

        # Evaluation uses a fresh environment, no exploration, and no learning.
        evaluation = evaluate_policy(UAVEnv(), agent)
        save_trajectory(evaluation["trajectory"], trajectory_file)

        results.append({
            "training_episodes": training_episodes,
            "seed": seed,
            "success": evaluation["success"],
            "steps": evaluation["steps"],
            "total_reward": evaluation["total_reward"],
            "final_distance": evaluation["final_distance"],
            "training_metrics_file": str(metrics_file),
            "trajectory_file": str(trajectory_file),
            "q_table_file": str(q_table_file),
        })

    summary_file = output_directory / f"evaluation_summary_seed_{seed}.csv"
    save_evaluation_summary(results, summary_file)

    # With the same seed, the first 100 episodes of both independent runs
    # should be identical. This is a direct reproducibility check.
    first_100_match = (
        training_runs[100] == training_runs[500][:100]
    )

    return results, summary_file, first_100_match


def main():
    parser = argparse.ArgumentParser(
        description="Compare 100 and 500 Q-learning training episodes"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    results, summary_file, first_100_match = run_comparison(
        seed=args.seed,
        output_directory=args.output_dir,
    )

    print("Training-length comparison")
    print("Seed:", args.seed)
    print("First 100 training episodes match:", first_100_match)

    for result in results:
        print(
            "Episodes:", result["training_episodes"],
            "| Success:", result["success"],
            "| Steps:", result["steps"],
            "| Total reward:", round(result["total_reward"], 2),
            "| Final distance:", round(result["final_distance"], 2),
        )

    print("Summary saved to:", summary_file)


if __name__ == "__main__":
    main()
