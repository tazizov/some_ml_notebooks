from tictactoe_env import TicTacToe
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import product

class RandomPlayer:
    def strategy(self, state):
        possible_steps = state[1]
        return possible_steps[np.random.randint(possible_steps.shape[0])]

class TicTacToeGame:
    def __init__(self, env):
        self.env = env

    def run_episode(self, player_x, player_o, return_history=False):
        state, _, is_done, _ = self.env.reset()
        states_x, states_o = [], []
        rewards_x, rewards_o = [], []
        actions_x, actions_o = [], []
        players = [player_x, player_o]
        cur_player = 0
        while not is_done:
            step = players[cur_player].strategy(state)
            state, reward, is_done, _ = self.env.step(step)
            if cur_player == 0:
                rewards_x.append(reward)
                states_x.append(state)
                actions_x.append(step)
            else:
                if reward == -10:
                    rewards_o.append(reward)
                else:
                    rewards_o.append(-reward)
                states_o.append(state)
                actions_o.append(step)
            cur_player = (cur_player + 1) % 2
        if (len(rewards_x) > len(rewards_o)) and (rewards_x[-1] == 1):
            rewards_o[-1] = -1
        if (rewards_o[-1] == 1):
            rewards_x[-1] = -1
        if return_history:
            history_x = [(state, action, reward) for (state, action, reward) in zip(states_x, actions_x, rewards_x)]
            history_o = [(state, action, reward) for (state, action, reward) in zip(states_o, actions_o, rewards_o)] 
            return history_x, history_o
        return rewards_x[-1], rewards_o[-1]

    def check_mean_reward(self, player_x, player_o, n_iter):
        rewards_x = []
        for _ in tqdm(range(n_iter)):
            reward_x, _ = self.run_episode(player_x, player_o)
            rewards_x.append(reward_x)
        return np.mean(rewards_x)


class QLearningPlayer:
    def __init__(self, eps, alpha, gamma, n_rows=3, n_cols=3):
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(lambda: np.zeros(n_rows * n_cols))
        self.possible_actions = list(product(range(n_rows), range(n_cols)))
        self.set_training()
        self.total_episodes_training = 0
    
    def set_training(self):
        self.training = True

    def set_evaluating(self):
        self.training = False

    def update_q(self, episode_history):
        # for t in range(len(episode_history)):
        #     state, action, reward = episode_history[t] # here r[t + 1], s[t], a[t]
        #     state_next, reward_next, _ = episode_history[t + 1] 
        #     # here we need reward[t + 1], s[t + 1]
        #     self.q[state][action] += self.alpha * (reward + self.gamma * np.max(self.q[state_next][action])) - self.q[state][action]
        self.total_episodes_training += 1
        for episode_step in episode_history:
            state, _, _ = episode_step
            state_hash = state[0]
            impossible_actions = np.where(np.array([*state_hash]) != '1')
            self.q_table[state_hash][impossible_actions] = -np.inf
        for t in range(1, len(episode_history)):
            # to r[t - 1], s[t - 1]
            state_cur, action_cur, _ = episode_history[t - 1]
            state_next, _, reward_next = episode_history[t]
            state_cur_hash, state_next_hash = state_cur[0], state_next[0]
            
            action_id = self.possible_actions.index(action_cur)
            # here we need reward[t + 1], s[t + 1]
            self.q_table[state_cur_hash][action_id] += self.alpha * (reward_next +\
                                 self.gamma * np.max(self.q_table[state_next_hash][action_id]) - self.q_table[state_cur_hash][action_id])

    def strategy(self, state):
        state_hash = state[0]
        coin = np.random.rand()
        greedy_action = self.greedy_step(state_hash)
        random_action = self.random_step()
        if self.training:
            coin = np.random.rand() < self.eps
            if coin:
                action = random_action
            else:
                action = greedy_action
        else:
            action = greedy_action
        return action

    def greedy_step(self, state_hash):
        return self.possible_actions[self.q_table[state_hash].argmax()]
    
    def random_step(self):
        return self.possible_actions[np.random.randint(0, len(self.possible_actions))]

if __name__ == "__main__":
    env = TicTacToe()
    game = TicTacToeGame(env)
    qplayer_params = {'eps': 0.1, 'alpha': 0.01, 'gamma': 1}
    qplayer_x , qplayer_o = QLearningPlayer(**qplayer_params), QLearningPlayer(**qplayer_params)
    for i in tqdm(range(100000)):
        history_x, history_o = game.run_episode(qplayer_x, qplayer_o, True)
        qplayer_x.update_q(history_x)
        qplayer_o.update_q(history_o)
    history_x, history_o = game.run_episode(qplayer_x, qplayer_o, True)
    qplayer_x.update_q(history_x)