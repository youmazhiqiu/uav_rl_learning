import math


class UAVEnv:

    def __init__(self):

        # UAV位置
        self.uav_x = 0
        self.uav_y = 0

        # 目标位置
        self.target_x = 5
        self.target_y = 5

        # 最大步数
        self.steps = 0
        self.max_steps = 100


    def reset(self):

        # 重置无人机

        self.uav_x = 0
        self.uav_y = 0

        self.steps = 0


        state = [
            self.uav_x,
            self.uav_y,
            self.target_x,
            self.target_y
        ]

        return state



    def step(self, action):

        # 记录步数

        self.steps += 1


        # 执行动作

        if action == "up":

            self.uav_y += 1


        elif action == "down":

            self.uav_y -= 1


        elif action == "left":

            self.uav_x -= 1


        elif action == "right":

            self.uav_x += 1



        # 计算距离

        distance = math.sqrt(
            (self.uav_x-self.target_x)**2 +
            (self.uav_y-self.target_y)**2
        )



        # 默认奖励

        reward = -distance



        # 终止条件

        done = False



        # 到达目标

        if distance < 1:

            reward = 100

            done = True



        # 超过最大步数

        elif self.steps >= self.max_steps:

            done = True



        # 新状态

        state = [
            self.uav_x,
            self.uav_y,
            self.target_x,
            self.target_y
        ]


        return state, reward, done
