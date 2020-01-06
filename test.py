import matplotlib.pyplot as plt
import numpy as np
import time
import math
import gym

import matplotlib.pyplot as plt
import numpy as np
import math
import gym

MAX_LEARNING_STEPS = 100000
LEARNING_INTERVAL_SIZE = 10000  # Interval for learning before eval
GAMMA = 0.95
LAMBDA = 0.8
STATE_NUM = 500
ACTION_NUM = 6 # TODO: change
TRAIN_SESSION_NUM = 10
EVALUATE_STEP_NUM = 100
EVALUATE_TIMES: int = 10

#TODO: add feature extraction class
class EGreedy:
    Q = None
    E = None
    epsilon_denominator = 0
    pick_e_greedy_times = 0

    # TODO: Need to store w only , no tables
    def __init__(self):
        self.Q = np.zeros((STATE_NUM, ACTION_NUM))
        self.E = np.zeros((STATE_NUM, ACTION_NUM))
        self.epsilon_denominator = 3  #TODO: change ?
        self.pick_e_greedy_times = 0   #TODO: redundant

    # TODO:Remove table , add Q evaluation
    def pick_greedy_action(self, state):
        # if there is more than one best action, chooses randomly from the best actions
        max_action = [0]
        max_value = self.Q[state, 0]
        for act in range(1, 6):
            if self.Q[state, act] == max_value:
                max_action.append(act)
            elif self.Q[state, act] > max_value:
                max_action = [act]
                max_value = self.Q[state, act]
        action_index = np.random.randint(max_action.__len__())
        return max_action[action_index]

    def should_explore(self):
        return np.random.random() < (1 / self.epsilon_denominator)

    # TODO:Less actions
    def pick_action(self, state: int):
        self.epsilon_denominator += 0.00001
        if self.should_explore():
            return np.random.randint(6)
        else:
            return self.pick_greedy_action(state)
    # TODO:Redundant
    def increase_E(self, state, action):
        self.E[state, action] += 1

    # TODO:Redundant
    def update_Qs(self, funct):
        # funct argument recieves (Q(s,a) , E(s,a))
        self.Q = funct(self.Q, self.E)

    # TODO:Change to w update
    def update_Es(self, funct):

        self.E = funct(self.E)

    def reset_E(self):
        self.E = np.zeros((STATE_NUM, ACTION_NUM))

    # TODO:Change to approximation
    def get_Q_value(self, state: int, action: int):
        return self.Q[state, action]


class SarsaLambda:
    policy: EGreedy

    def __init__(self):
        self.policy = EGreedy()

    def train(self, steps_num: int):
        # Continue training for LEARNING_INTERVAL_SIZE steps from training step steps_num

        #TODO: observation is tuple
        done = True
        for step in range(LEARNING_INTERVAL_SIZE):
            if done:
                observation = env.reset()
                self.policy.reset_E()
                action = self.policy.pick_action(observation)
                action_value = self.policy.get_Q_value(observation, action)
            else:
                action = action_next
                observation = observation_next
                action_value = action_next_value

            # Get R_t+1, O_t+1, A_t+1 and Q(O_t+1,A_t+1)
            observation_next, reward, done, _ = env.step(action)
            action_next = self.policy.pick_action(observation_next)
            action_next_value = self.policy.get_Q_value(observation_next, action_next)

            delta = reward + GAMMA * action_next_value - action_value
            alpha = 1 / math.log(steps_num + step + 2)
            steps_num += 1

            # Update policy
            self.policy.increase_E(observation, action)  #TODO: Redundant
            self.policy.update_Qs(lambda q, e: q + alpha * delta * e) #TODO: Redundant
            # TODO :Compute delta_w  - do it in Policy class ? Here ?

            self.policy.update_Es(lambda e: GAMMA * LAMBDA * e)  # TODO: Update w instead -

    def evaluate(self) -> float:
        #TODO: No need to change

        # Return : average EVALUATE_TIMES discounted returns over EVALUATE_STEP_NUM steps
        returns = 0.0
        for _ in range(EVALUATE_TIMES):
            gamma_factor = 1
            observation = env.reset()
            for _ in range(EVALUATE_STEP_NUM):
                action = self.policy.pick_greedy_action(observation)
                observation, reward, done, _ = env.step(action)
                returns += gamma_factor * reward
                gamma_factor *= GAMMA
                if done:
                    observation = env.reset()
        return returns / EVALUATE_TIMES


def main_loop():
    # TODO: No need to change
    train_interval_num = MAX_LEARNING_STEPS // LEARNING_INTERVAL_SIZE
    returns = np.zeros((train_interval_num, TRAIN_SESSION_NUM))
    intervals = np.arange(0, stop=MAX_LEARNING_STEPS + 1, step=LEARNING_INTERVAL_SIZE)
    for train_session in range(TRAIN_SESSION_NUM):
        rl = SarsaLambda()
        for train_interval, step in zip(range(train_interval_num), intervals):
            rl.train(step)
            returns[train_interval, train_session] = rl.evaluate()
    plot_errorbar(intervals, returns)


def plot_errorbar(intervals, returns):
    # TODO: No need to change
    # Compute mean and variance and plot
    means = []
    var = []
    for returns_arr in returns:
        means.append(np.mean(returns_arr))
        var.append(np.var(returns_arr))
    # print("mean", means[-1])
    # print("var", var[-1])
    plt.errorbar(intervals[1:], means, var, linestyle='None', marker='s')
    plt.xticks(intervals[1:])
    plt.show()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    main_loop()
    env.close()
