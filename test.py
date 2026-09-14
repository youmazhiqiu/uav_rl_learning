from env.uav_env import UAVEnv


env = UAVEnv()


state = env.reset()

print("初始状态:")
print(state)


for i in range(10):

    state, reward, done = env.step("right")

    print(
        "状态:",
        state,
        "奖励:",
        reward
    )

    if done:
        print("到达目标")
        break
