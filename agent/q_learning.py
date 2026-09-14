import random


class QLearningAgent:


    def __init__(self):

        # Q表
        self.q_table = {}


        # 动作空间

        self.actions = [

            "up",
            "down",
            "left",
            "right"

        ]


        # 探索率

        self.epsilon = 0.3


        # 学习率

        self.alpha = 0.1


        # 折扣因子

        self.gamma = 0.9



    def choose_action(self, state):


        # list -> tuple

        state = tuple(state)



        # 第一次遇到状态

        if state not in self.q_table:


            self.q_table[state] = {}


            for action in self.actions:

                self.q_table[state][action] = 0



        # 探索

        if random.random() < self.epsilon:


            return random.choice(self.actions)



        # 利用

        else:


            return max(

                self.q_table[state],

                key=self.q_table[state].get

            )



    def learn(self, state, action, reward, next_state):


        # list -> tuple

        state = tuple(state)

        next_state = tuple(next_state)



        # 初始化新状态


        if next_state not in self.q_table:


            self.q_table[next_state] = {}


            for action_name in self.actions:

                self.q_table[next_state][action_name] = 0



        # 当前Q值


        old_q = self.q_table[state][action]



        # 下一状态最大Q


        max_next_q = max(

            self.q_table[next_state].values()

        )



        # Q学习目标


        target = reward + self.gamma * max_next_q



        # 更新


        self.q_table[state][action] = (

            old_q +

            self.alpha * (target - old_q)

        )
