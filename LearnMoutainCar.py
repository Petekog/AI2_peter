import matplotlib.pyplot as plt
import numpy as np
import math
import gym

# TODO: adjust parameters
MAX_LEARNING_STEPS = 100000
LEARNING_INTERVAL_SIZE = 10000  # Interval for learning before eval
ACTION_NUM = 3
GAMMA = 1
LAMBDA = 0.8
TRAIN_SESSION_NUM = 10
EVALUATE_STEP_NUM = 100
EVALUATE_TIMES: int = 10


class Tiling:
    dim = 2

    def __init__(self, bin_sizes, lows, highs, offsets):
        self.bin_sizes = bin_sizes
        self.lows = np.add(lows, offsets)
        self.highs = np.add(highs, offsets)
        # holds the size of the bins per dim
        self.bin_nums = np.zeros(Tiling.dim, dtype=np.int8)
        for index in range(Tiling.dim):
            self.bin_nums[index] = int(np.ceil((self.highs[index] - self.lows[index]) / self.bin_sizes[index]))
        self.number_of_tiles_per_action = np.prod(self.bin_nums, dtype=int)
        self.number_of_tiles = ACTION_NUM * self.number_of_tiles_per_action

    def get_tile_feature(self, state, action):
        # Returns a feature vector of zeroes except the state,action feature which is 1
        tile = self.get_tile_number(state)
        action_tile = action * self.number_of_tiles_per_action + tile
        features = np.zeros(self.number_of_tiles)
        features[action_tile] = 1
        return features

    def get_tile_number(self, state):
        # Returns the tile number of the state according to its 2 dimensions
        bins = []
        for low, bin_size, s in zip(self.lows, self.bin_sizes, state):
            bins.append(int(np.floor((s - low) / bin_size)))
        tile = bins[0] + (bins[1] * self.bin_nums[0])
        return tile

    # return vector size 3 of vectors. every vector of size number_of_tiles_per_action*3
    def get_features_per_action_for_state(self, state):
        tile = self.get_tile_number(state)
        features_per_action = [[], [], []]
        for action in range(ACTION_NUM):
            action_tile = action * self.number_of_tiles_per_action + tile
            action_features = np.zeros(self.number_of_tiles)
            action_features[action_tile] = 1
            features_per_action[action] = action_features
        return features_per_action


class Approximator:
    def __init__(self):
        self.min_pos = -1.2
        self.max_pos = 0.6
        self.min_speed = -0.07
        self.max_speed = 0.07
        # TODO: calc number
        self.number_of_features = 10

    # TODO: implement
    def get_features(self, state, action):
        return [state, action]

    def get_feature_number(self):
        return self.number_of_features

    # TODO: implement
    # return vector size 3 of vectors. every vector of size number of tiles*3*number_of_offsets
    def get_features_per_action_for_state(self, state):
        return [[1], [2], [3]]


class EGreedyApproximatedValue:
    W = None
    E = None
    approximator = None
    epsilon_denominator = 0

    def __init__(self):
        self.approximator = Approximator()
        self.feature_number = self.approximator.get_feature_number()
        # TODO: maybe smart init
        self.W = np.zeros(self.feature_number)
        self.E = np.zeros(self.feature_number)
        self.epsilon_denominator = 1

    def pick_greedy_action_and_get_value_and_features(self, state):
        actions_features = self.approximator.get_features_per_action_for_state(state)
        action_values = []
        for action_features in actions_features:
            action_value = np.dot(self.W, action_features)
            action_values.add(action_value)
        best_action = np.argmax(action_values)
        best_action_value = action_values[best_action]
        best_action_features = actions_features[best_action]
        return best_action, best_action_value, best_action_features

    def should_explore(self):
        return np.random.random() < (1 / self.epsilon_denominator)

    def pick_action_and_get_value_and_features(self, state: int):
        self.epsilon_denominator += 0.00001
        if self.should_explore():
            action = np.random.randint(3)
            action_features = self.approximator.get_features(state, action)
            action_value = np.dot(self.W, action_features)
            return action, action_value, action_features
        else:
            return self.pick_greedy_action_and_get_value_and_features(state)

    def update_weights(self, funct):
        # funct argument receives (Q(s,a) , E(s,a))
        self.W = funct(self.W, self.E)

    def update_Es(self, funct):
        # funct argument receives (E(s,a))
        self.E = funct(self.E)

    def reset_E(self):
        self.E = np.zeros(self.feature_number)


class SarsaLambda:
    policy: EGreedyApproximatedValue

    def __init__(self):
        self.policy = EGreedyApproximatedValue()

    def train(self, steps_num: int):
        # Continue training for LEARNING_INTERVAL_SIZE steps from training step steps_num
        done = True
        for step in range(LEARNING_INTERVAL_SIZE):
            if done:
                observation = env.reset()
                self.policy.reset_E()
                action, action_value, _ = self.policy.pick_action_and_get_value_and_features(observation)
            else:
                action = action_next
                # observation = observation_next
                action_value = action_next_value

            # Get R_t+1, O_t+1, A_t+1 and x(O_t+1,A_t+1)
            observation_next, reward, done, _ = env.step(action)
            action_next, action_next_value, features = self.policy.pick_action_and_get_value_and_features(observation_next)

            steps_num += 1

            # Update policy according to TD(lambda) formula
            delta = reward + GAMMA * action_next_value - action_value
            alpha = 1 / math.log(steps_num + step + 2)
            self.policy.update_weights(lambda w, e: w + alpha * delta * e)
            self.policy.update_Es(lambda e: GAMMA * LAMBDA * e + features)

    def evaluate(self) -> float:
        # TODO: reimplement according to assignment
        # Return : average EVALUATE_TIMES discounted returns over EVALUATE_STEP_NUM steps
        returns = 0.0
        for _ in range(EVALUATE_TIMES):
            gamma_factor = 1
            observation = env.reset()
            for _ in range(EVALUATE_STEP_NUM):
                action = self.policy.pick_greedy_action_and_get_value_and_features(observation)
                observation, reward, done, _ = env.step(action)
                returns += gamma_factor * reward
                gamma_factor *= GAMMA
                if done:
                    observation = env.reset()
        return returns / EVALUATE_TIMES


def main_loop():
    train_interval_num = MAX_LEARNING_STEPS // LEARNING_INTERVAL_SIZE
    returns = np.zeros((train_interval_num, TRAIN_SESSION_NUM))
    intervals = np.arange(0, stop=MAX_LEARNING_STEPS+1, step=LEARNING_INTERVAL_SIZE)
    for train_session in range(TRAIN_SESSION_NUM):
        rl = SarsaLambda()
        for train_interval, step in zip(range(train_interval_num), intervals):
            rl.train(step)
            returns[train_interval, train_session] = rl.evaluate()
    plot_errorbar(intervals, returns)


def plot_errorbar(intervals, returns):
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
