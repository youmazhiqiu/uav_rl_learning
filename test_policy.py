from env.uav_env import UAVEnv
from agent.q_learning import QLearningAgent
import pickle



env = UAVEnv()


agent = QLearningAgent()



# 加载训练好的Q表

with open("q_table.pkl","rb") as f:

    agent.q_table = pickle.load(f)



# 测试时不探索

agent.epsilon = 0



state = env.reset()


print("start:", state)



total_reward = 0



while True:


    action = agent.choose_action(state)


    next_state, reward, done = env.step(action)



    print(
        "state:",
        state,
        "action:",
        action,
        "reward:",
        reward
    )



    total_reward += reward


    state = next_state


    if done:

        break



print("final reward:", total_reward)
