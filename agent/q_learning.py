import random


class QLearningAgent:
    """Tabular Q-learning controller for the discrete UAV environment."""

    def __init__(self, epsilon=0.3, alpha=0.1, gamma=0.9, seed=0):
        # Q-table: Q(state, action)
        self.q_table = {}

        # Discrete control inputs
        self.actions = ["up", "down", "left", "right"]

        # Reinforcement learning parameters
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # This generator controls epsilon decisions and random actions.
        self.random_generator = random.Random(seed)

    def _initialize_state(self, state):
        """Add an unseen state to the Q-table with zero action values."""
        state = tuple(state)

        if state not in self.q_table:
            self.q_table[state] = {}

            for action in self.actions:
                self.q_table[state][action] = 0

        return state

    def choose_action(self, state):
        """Choose an epsilon-greedy action during training."""
        state = self._initialize_state(state)

        # Exploration: try a random control input.
        if self.random_generator.random() < self.epsilon:
            return self.random_generator.choice(self.actions)

        # Exploitation: use the action with the largest learned Q-value.
        return max(
            self.q_table[state],
            key=self.q_table[state].get
        )

    def choose_greedy_action(self, state):
        """Choose the learned greedy action without exploration or table updates."""
        state = tuple(state)

        # An unseen state has four equal zero values. Return the first action in
        # the same deterministic order used when the Q-table is initialized.
        if state not in self.q_table:
            return self.actions[0]

        return max(
            self.q_table[state],
            key=self.q_table[state].get
        )

    def learn(self, state, action, reward, next_state):
        """Apply the existing one-step Q-learning update."""
        state = self._initialize_state(state)
        next_state = self._initialize_state(next_state)

        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        target = reward + self.gamma * max_next_q

        self.q_table[state][action] = (
            old_q + self.alpha * (target - old_q)
        )
