import argparse
import csv
import math
import pickle
from pathlib import Path

from agent.q_learning import QLearningAgent
from env.uav_env import UAVEnv


def distance_from_goal(state):
    """Calculate the UAV's Euclidean distance from its goal."""
    x, y, goal_x, goal_y = state
    return math.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)


def train_agent(episodes, seed):
    """Train one Q-learning agent and return it with episode metrics."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")

    env = UAVEnv()
    agent = QLearningAgent(seed=seed)
    metrics = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        episode_steps = 0

        while True:
            # Training uses epsilon-greedy exploration.
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # Learning happens only in the training function.
            agent.learn(state, action, reward, next_state)

            total_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        final_distance = distance_from_goal(state)
        success = final_distance < 1

        metrics.append({
            "seed": seed,
            "episode": episode,
            "episode_reward": total_reward,
            "episode_steps": episode_steps,
            "success": success,
            "final_distance": final_distance,
        })

    return agent, metrics


def save_training_metrics(metrics, file_path):
    """Save per-episode training measurements as CSV."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "episode",
        "episode_reward",
        "episode_steps",
        "success",
        "final_distance",
    ]

    with file_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def save_q_table(agent, file_path):
    """Save a trained Q-table without changing the learning logic."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as q_table_file:
        pickle.dump(agent.q_table, q_table_file)


def main():
    parser = argparse.ArgumentParser(description="Train the tabular UAV controller")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics", default="training_metrics.csv")
    parser.add_argument("--q-table", default="q_table.pkl")
    args = parser.parse_args()

    agent, metrics = train_agent(args.episodes, args.seed)
    save_training_metrics(metrics, args.metrics)
    save_q_table(agent, args.q_table)

    final_metrics = metrics[-1]
    print("Training complete")
    print("Episodes:", args.episodes)
    print("Seed:", args.seed)
    print("Final episode reward:", round(final_metrics["episode_reward"], 2))
    print("Metrics saved to:", args.metrics)
    print("Q-table saved to:", args.q_table)


if __name__ == "__main__":
    main()
