from env.uav_env import UAVEnv
from agent.q_learning import QLearningAgent
import pickle



env = UAVEnv()

agent = QLearningAgent()


episodes = 100



for episode in range(episodes):


    state = env.reset()

    total_reward = 0


    while True:


        action = agent.choose_action(state)


        next_state, reward, done = env.step(action)


        agent.learn(
            state,
            action,
            reward,
            next_state
        )


        total_reward += reward


        state = next_state


        if done:

            break



    print(
        "Episode:",
        episode,
        "Reward:",
        round(total_reward,2)
    )



# 保存Q表

with open("q_table.pkl","wb") as f:

    pickle.dump(agent.q_table,f)


print("Q table saved!")
