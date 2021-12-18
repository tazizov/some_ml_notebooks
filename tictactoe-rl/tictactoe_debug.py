import re
from tictactoe_env import TicTacToe
import numpy as np
import torch
from tqdm.notebook import tqdm
from collections import defaultdict
from itertools import product
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import random

class BasePlayer(ABC):
    def set_training(self):
        self.training = True
    
    def set_evaluating(self):
        self.training = False

    @abstractmethod
    def strategy(self, state):
        raise NotImplementedError('Implement me in subclass')

    @abstractmethod
    def reset(self):
        raise NotImplementedError('Implement me in subclass')


class RandomPlayer(BasePlayer):
    def strategy(self, state):
        possible_steps = state[1]
        return possible_steps[np.random.randint(possible_steps.shape[0])]
    
    def reset(self):
        pass

class TicTacToeGame:
    def __init__(self, env):
        self.env = env

    def run_episode(self, player_x, player_o, return_history=False, print_board=False):
        state, _, is_done, _ = self.env.reset()
        if print_board:
            self.env.printBoard()
        rewards = [[], []]
        states = [[], []]
        actions = [[], []]
        players = [player_x, player_o]
        cur_player = 0
        while not is_done:
            states[cur_player].append(state)
            action = players[cur_player].strategy(state)
            actions[cur_player].append(action)
            state, reward, is_done, _ = self.env.step(action)
            if print_board:
                self.env.printBoard()
            if cur_player == 0:
                rewards[cur_player].append(reward)
            else:
                if reward == -10:
                    rewards[cur_player].append(reward)
                else:
                    rewards[cur_player].append(-reward)
            cur_player = (cur_player + 1) % 2
        if rewards[1][-1] == 1:
            rewards[0][-1] = -1
        if rewards[0][-1] == 1:
            rewards[1][-1] = -1
        if return_history:
            return states, actions, rewards
        else:
            return rewards[0][-1], rewards[1][-1]

    def check_rewards(self, player_x, player_o, n_iter=10000, use_tqdm=True):
        player_x.set_evaluating()
        player_o.set_evaluating()
        wins_x_cnt = 0
        wins_o_cnt = 0
        draws_cnt = 0
        miss_x = 0
        miss_o = 0
        for _ in tqdm(range(n_iter), disable=(not use_tqdm)):
            reward_x, reward_o = self.run_episode(player_x, player_o, return_history=False)
            if reward_x == -10:
                miss_x += 1
            if reward_o == -10:
                miss_o += 1
            if reward_x == 1:
                wins_x_cnt += 1
            if reward_o == 1:
                wins_o_cnt += 1
            if (reward_x == 0) and (reward_o == 0):
                draws_cnt += 1
        return {
            'wins_x': wins_x_cnt,
            'wins_o': wins_o_cnt,
            'draws': draws_cnt,
            'missx': miss_x,
            'misso': miss_o
        }


class QLearningPlayer:
    def __init__(self, eps, alpha, gamma, n_rows=3, n_cols=3):
        self.size = (n_rows, n_cols)
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(lambda: np.zeros(n_rows * n_cols))
        self.all_actions = list(product(range(n_rows), range(n_cols)))
        self.set_training()
        self.total_episodes_training = 0
    
    def reset(self):
        self.q_table = defaultdict(lambda: np.zeros(self.size[0] * self.size[1]))
        self.total_episodes_training = 0

    def update_q(self, episode_history):
        states, actions, rewards = episode_history
        self.total_episodes_training += 1
        for t in range(len(states)):
            state_cur, action_cur = states[t], actions[t]
            state_cur_hash = state_cur[0]
            reward_next = rewards[t]
            if t == len(states) - 1:
                state_next = None
            else:
                state_next = states[t + 1]
            action_id = self.all_actions.index(action_cur)
            if state_next is not None:
                state_next_hash = state_next[0]
                self.q_table[state_cur_hash][action_id] += self.alpha * (reward_next + self.gamma * np.max(self.q_table[state_next_hash]) - self.q_table[state_cur_hash][action_id])
            else:
                self.q_table[state_cur_hash][action_id] += self.alpha * (reward_next - self.q_table[state_cur_hash][action_id])

    def strategy(self, state):
        state_hash = state[0]
        coin = np.random.rand()
        greedy_action = self.greedy_step(state_hash)
        random_action = self.random_step(state_hash)
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
        return self.all_actions[self.q_table[state_hash].argmax()]
    
    def random_step(self, state_hash):
        return self.all_actions[np.random.randint(0, len(self.all_actions) - 1)]

    def __str__(self):
        return f"QLearningPlayer(eps={self.eps}, alpha={self.alpha}, gamma={self.gamma})"

class DQNPlayer(BasePlayer):
    def __init__(self, eps, alpha, gamma, n_rows, n_cols, model=None) -> None:
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.size = (n_rows, n_cols)
        if model is not None:
            self.model = model
        else:
            self.model = DQNet(hidden_size=128, board_size=self.size)
        self.all_actions = list(product(range(n_rows), range(n_cols)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.training = True
        
    def strategy(self, state):
        state_hash = state[0]
        coin = np.random.rand()
        greedy_action = self.greedy_step(state_hash)
        random_action = self.random_step(state_hash)
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
        return self.all_actions[self.model(self._state_to_tensor(state_hash)).argmax().item()]
    
    def random_step(self, state_hash):
        return self.all_actions[np.random.randint(0, len(self.all_actions) - 1)]

    def update_q(self, experience):
        states, actions, rewards, states_next = experience
        states = torch.cat([self._state_to_tensor(state) for state in states])
        states_next = torch.cat([self._state_to_tensor(state) for state in states_next])
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float64)
        q = self.model(states).gather(1, actions)
        qmax = self.model(states_next).detach().max(1)[0]
        qnext = rewards + (self.gamma * qmax)
        loss = torch.nn.functional.mse_loss(q.flatten().float(), qnext.flatten().float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def _state_to_tensor(self, state_hash):
        return torch.tensor(np.array(list(state_hash), dtype=np.float64)).reshape(1, 1, self.size[0], self.size[1]).float()
    
    def reset(self):
        pass
    
class DQNet(torch.nn.Module):
    def __init__(self, hidden_size, board_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.board_size = board_size
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=board_size)
        self.l1 = torch.nn.Linear(hidden_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, board_size[0] * board_size[1])
    def forward(self, x):
        x = self.conv(x).view(-1, self.hidden_size)
        x = torch.nn.functional.relu(self.l1(x))
        x = self.l2(x)
        return x

class Trainer:
    def __init__(self, game):
        self.game = game

    def train_one(self, training_side, player, sparring_partner=RandomPlayer(), n_iter=100000, use_tqdm=True):
        if not training_side in ('x', 'o'):
            raise ValueError('training_side must be "x" or "o"')
        player.set_training()
        sparring_partner.set_evaluating()
        if training_side == 'x':
            player_x = player
            player_o = sparring_partner
        else:
            player_x = sparring_partner
            player_o = player
        for i in tqdm(range(n_iter), desc=f'Training {player.__str__()}', disable=(not use_tqdm)):
            states, actions, rewards = self.game.run_episode(player_x, player_o, return_history=True)
            (states_x, states_o), (actions_x, actions_o), (rewards_x, rewards_o) = states, actions, rewards
            history_x = (states_x, actions_x, rewards_x)
            history_o = (states_o, actions_o, rewards_o)
            if training_side == 'x':
                player_x.update_q(history_x)
            else:
                player_o.update_q(history_o)

    def train_two(self, player_x, player_o, n_iter, use_tqdm=True):
        player_x.set_training()
        player_o.set_training()
        desc = f'Training x: {player_x.__str__()} vs o: {player_o.__str__()}'
        for i in tqdm(range(n_iter), desc=desc, disable=(not use_tqdm)):
            states, actions, rewards = self.game.run_episode(player_x, player_o, return_history=True)
            (states_x, states_o), (actions_x, actions_o), (rewards_x, rewards_o) = states, actions, rewards
            history_x = (states_x, actions_x, rewards_x)
            history_o = (states_o, actions_o, rewards_o)
            player_x.update_q(history_x)
            player_o.update_q(history_o)

    def plot_learning_one(self, training_side, player, sparring_partner=RandomPlayer(),
                         max_iter=500000, iter_step=10000, iter_check=100000, use_tqdm=True, figsize=(16, 9)):
        player.reset()
        n_steps = max_iter // iter_step
        wins_percentages = []
        lose_percentages = []
        draws_percentages = []
        misses_percentages = []
        if training_side == 'x':
            player_x = player
            player_o = sparring_partner
        else:
            player_x = sparring_partner
            player_o = player
        for i in tqdm(range(n_steps), desc='Collecting plot data', disable=(not use_tqdm)):                 
            self.train_one(training_side, player, sparring_partner, iter_step, use_tqdm=False)
            counts = self.game.check_rewards(player_x, player_o, n_iter=iter_check, use_tqdm=False)
            wins_sum = counts['wins_x'] if training_side == 'x' else counts['wins_o']
            loses_sum = counts['wins_o'] if training_side == 'x' else counts['wins_x']
            misses_sum = counts['missx'] if training_side == 'x' else counts['misso']
            wins_percentages.append(wins_sum / sum(counts.values()))
            lose_percentages.append(loses_sum / sum(counts.values()))
            draws_percentages.append(counts['draws'] / sum(counts.values()))
            misses_percentages.append(misses_sum / sum(counts.values()))
            # print(f'Wins percentage: {wins_sum / sum(counts.values())}')
            clear_output()
            print(f'{int(i + 1)} / {int(n_steps)} finished')
            x_plot = np.arange(iter_step, (i + 2) * iter_step, iter_step)
            plt.figure(figsize=figsize)
            sns.lineplot(x=x_plot, y=wins_percentages, alpha=0.7, marker='o', label='player wins')
            sns.lineplot(x=x_plot, y=lose_percentages, alpha=0.7, marker='o', label='player loses')
            sns.lineplot(x=x_plot, y=draws_percentages, alpha=0.7, marker='o', label='draws')
            sns.lineplot(x=x_plot, y=misses_percentages, alpha=0.7, marker='o', label='player misses')
            plt.xlabel('n_iter_train')
            plt.ylabel(f'Percentage off')
            plt.title('Зависимость исходов игры от числа итераций обучения')
            plt.show()


    def plot_learning_two(self, player_x, player_o, max_iter=500000, iter_step=10000, iter_check=100000,
                          use_tqdm=True, figsize=(16, 9)):
        n_steps = max_iter // iter_step
        winsx_percentages = []
        winso_percentages = []
        draws_percentages = []
        player_x.reset()
        player_o.reset()
        for i in tqdm(range(n_steps), desc='Collecting plot data', disable=(not use_tqdm)):                 
            self.train_two(player_x, player_o, iter_step, use_tqdm=False)
            counts = self.game.check_rewards(player_x, player_o, n_iter=iter_check, use_tqdm=False)
            winsx_sum = counts['wins_x']
            winso_sum = counts['wins_o']
            winsx_percentages.append(winsx_sum / sum(counts.values()))
            winso_percentages.append(winso_sum / sum(counts.values()))
            draws_percentages.append(counts['draws'] / sum(counts.values()))
            print(f'Winsx percentage: {winsx_sum / sum(counts.values())}')
        x_plot = np.arange(iter_step, max_iter + iter_step, iter_step)
        plt.figure(figsize=figsize)
        sns.lineplot(x=x_plot, y=winsx_percentages, alpha=0.7, marker='o', label='x wins')
        sns.lineplot(x=x_plot, y=winso_percentages, alpha=0.7, marker='o', label='o wins')
        sns.lineplot(x=x_plot, y=draws_percentages, alpha=0.7, marker='o', label='draws')
        plt.xlabel('n_iter_train')
        plt.ylabel(f'Percentage off')
        plt.title('Зависимость исходов игры от числа итераций обучения')

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, exptuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exptuple
        self.position = (self.position + 1) % self.capacity
       
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# def check_rewards(self, player_x, player_o, n_iter=10000, use_tqdm=True):


if __name__ == "__main__":
    env = TicTacToe()
    game = TicTacToeGame(env)
    qplayer_params = {'eps': 0.1, 'alpha': 0.01, 'gamma': 1}
    qplayer_x , qplayer_o = QLearningPlayer(**qplayer_params), QLearningPlayer(**qplayer_params)
    for i in tqdm(range(10)):
        (states_x, states_o), (actions_x, actions_o), (rewards_x, rewards_o) = game.run_episode(qplayer_x, qplayer_o, True)
        history_x = (states_x, actions_x, rewards_x)
        history_o = (states_o, actions_o, rewards_o)
        qplayer_x.update_q(history_x)
        qplayer_o.update_q(history_o)
    (states_x, states_o), (actions_x, actions_o), (rewards_x, rewards_o) = game.run_episode(qplayer_x, qplayer_o, True)
    qplayer_x.update_q(history_x)