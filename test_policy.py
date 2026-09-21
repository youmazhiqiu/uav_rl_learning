import argparse
import math
import pickle
from pathlib import Path

from agent.q_learning import QLearningAgent
from env.uav_env import UAVEnv


def evaluate_policy(env, agent):
    """Run one greedy episode without learning and return evaluation results."""
    # Evaluation has no exploration.
    agent.epsilon = 0

    state = env.reset()
    trajectory = [state.copy()]
    total_reward = 0
    steps = 0

    while True:
        # This method is deterministic and does not modify the Q-table.
        action = agent.choose_greedy_action(state)
        next_state, reward, done = env.step(action)

        total_reward += reward
        steps += 1
        state = next_state
        trajectory.append(state.copy())

        if done:
            break

    x, y, goal_x, goal_y = state
    final_distance = math.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    success = final_distance < 1

    return {
        "success": success,
        "steps": steps,
        "total_reward": total_reward,
        "final_distance": final_distance,
        "trajectory": trajectory,
    }


def load_q_table(agent, file_path):
    """Load a saved Q-table into an agent."""
    with Path(file_path).open("rb") as q_table_file:
        agent.q_table = pickle.load(q_table_file)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a learned UAV policy")
    parser.add_argument("--q-table", default="q_table.pkl")
    args = parser.parse_args()

    env = UAVEnv()
    agent = QLearningAgent(epsilon=0)
    load_q_table(agent, args.q_table)

    result = evaluate_policy(env, agent)

    print("Evaluation complete")
    print("Success:", result["success"])
    print("Steps:", result["steps"])
    print("Total reward:", round(result["total_reward"], 2))
    print("Final distance:", round(result["final_distance"], 2))
    print("Trajectory:", result["trajectory"])


if __name__ == "__main__":
    main()
