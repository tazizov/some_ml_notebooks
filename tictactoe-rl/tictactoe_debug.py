from tictactoe_env import TicTacToe
import numpy as np

class RandomPlayer:
    def strategy(self, state):
        possible_steps = state[1]
        return possible_steps[np.random.randint(possible_steps.shape[0])]

class TicTacToeGame:
    def __init__(self, env):
        self.env = env

    def run_episode(self, player_x, player_o):
        state, _, is_done, _ = self.env.reset()
        self.env.printBoard()
        rewards_x, rewards_o = [], []
        while not is_done:
            step_x = player_x.strategy(state)
            state, reward_x, is_done, _ = self.env.step(step_x)
            step_o = player_o.strategy(state)
            self.env.printBoard()
            state, reward_o, is_done, _ = self.env.step(step_o)
            rewards_x.append(reward_x)
            rewards_o.append(reward_o)
            self.env.printBoard()
            print(reward_x, reward_o)
        return rewards_x, rewards_o

if __name__ == '__main__':    
    env = TicTacToe()
    game = TicTacToeGame(env)

    player_x = RandomPlayer()
    player_o = RandomPlayer()

    game.run_episode(player_x, player_o)