import numpy as np
import random
import time
import math
import pickle

"""
    BOARD : BOARD_ROWS * BOARD_COLUMNS -> [ [ 1 2 3 ] [ 4 5 6 ] [ 7 8 9] ]
    ACTIONS : 1 2 3  4 5 6  7 8 9 
    SYMBOL : '1' or '2'
    STATE : '012001001' -> 0 : available position
                           1 : symbol of player 1
                           2 : symbol of player 2 
    STATES-HISTORY : [('012001001', 4), ('0121200121', 1) ...]  history of states during single game
    Q-TABLE : {'[012001001]': {'4': -2.78, '1': 6.0}, ....} 
    AVAILABLE-POSITIONS = [(1,1), (2,4) ...]
    G-TIME  : Maximum training iterations
    MCS-TIME : Maximum time (ms) for Monte Carlo Search
    A : Parameter for E-greedy E=[A,B]
    B : Parameter for E-greedy E=[A,B]
    DISCOUNT-RATE : Parameter for Q-learning updating Q-Table
    ALPHA : Parameter for Q-learning updating Q-Table
"""

MCS_TIME = 50
A = 0.5
B = 0
DISCOUNT_RATE = 0.9
ALPHA = 0.1

REWARD_P1_WIN = 10
REWARD_P1_LOOSE = -10
REWARD_P1_DRAW = 0

REWARD_P2_WIN = 10
REWARD_P2_LOOSE = -10
REWARD_P2_DRAW = 0


class Player:
    def __init__(self, symbol):
        self.symbol = symbol
        self.QTable = {}
        self.available_positions = []
        self.board = None
        self.states_history = []

    def update_available_positions(self):
        self.available_positions = []
        rows, columns = self.board.shape
        for i in range(rows):
            for j in range(columns):
                if self.board[i, j] == 0:
                    self.available_positions.append((i, j))

    def choose_action(self, board, e_greedy):
        # update available positions
        self.board = board
        self.update_available_positions()

        # check e-greedy if is enabled
        if e_greedy:
            action = self.random_action()
        else:
            action = self.GBest_action()

        # save current state in history
        self.add_state_history(action)

        return action

    def random_action(self):
        # select best position to win if exists otherwise select random action from available positions
        action = self.position_choice()

        # add new states in QTable
        self.append_states_QTable()

        return action

    def position_choice(self):
        # check if next position will lead us to win
        rows, columns = self.board.shape

        if self.symbol == 1:
            win_positions = ['[0 1 1]', '[1 1 0]', '[1 0 1]']
        else:
            win_positions = ['[0 2 2]', '[2 2 0]', '[2 0 2]']

        best_positions = []

        # check rows
        for i in range(rows):
            row = str(self.board[i, :])

            if row == win_positions[0]:
                best_positions.append((i, 0))
            if row == win_positions[1]:
                best_positions.append((i, 2))
            if row == win_positions[2]:
                best_positions.append((i, 1))

        # check columns
        for i in range(columns):
            column = str(self.board[:, i])

            if column == win_positions[0]:
                best_positions.append((0, i))
            if column == win_positions[1]:
                best_positions.append((2, i))
            if column == win_positions[2]:
                best_positions.append((1, i))

        # check diagonals
        diagonal1 = str([self.board[i, i] for i in range(columns)]).replace(',', '')
        diagonal2 = str([self.board[i, columns - i - 1] for i in range(columns)]).replace(',', '')

        # descendant diagonal check
        if diagonal1 == win_positions[0]:
            best_positions.append((0, 0))
        if diagonal1 == win_positions[1]:
            best_positions.append((2, 2))

        # ascendant diagonal check
        if diagonal2 == win_positions[0]:
            best_positions.append((0, 2))
        if diagonal2 == win_positions[1]:
            best_positions.append((2, 0))

        # ascendant & descendant diagonal check
        if diagonal1 == win_positions[2] or diagonal2 == win_positions[2]:
            best_positions.append((1, 1))

        # check if best position list is empty then return random choice from available positions
        if len(best_positions) > 0:
            return random.choice(best_positions)
        else:
            return random.choice(self.available_positions)

    def monte_carlo_action(self):
        mcs = MonteCarloSearch(player_symbol=self.symbol, board=self.board)
        action = mcs.search

        # add new states in QTable
        self.append_states_QTable()
        return action

    def append_states_QTable(self):
        rows, columns = self.board.shape

        # add new empty row for hashed board inside QTable
        new_state = self.hash_board()
        if not self.QTable.keys().__contains__(new_state):
            for i in range(1, (columns * rows) + 1):
                self.QTable[new_state] = {i: 0 for i in range(1, (columns * rows) + 1)}

    def get_available_actions(self):
        available_actions = []
        rows, columns = self.board.shape
        action = 0
        for i in range(rows):
            for j in range(columns):
                action += 1
                if self.board[i, j] == 0:
                    available_actions.append(action)
        return available_actions

    def GBest_action(self):
        # get current game state
        state = self.hash_board()

        # check if state already exists in QTable & return Best action
        if self.QTable.keys().__contains__(state):

            # get available possible actions
            available_actions = self.get_available_actions()

            # get best actions list sorted reverse
            best_actions = sorted(self.QTable[state], key=self.QTable[state].get, reverse=True)

            # select best action that exists in possible actions
            best_action = None
            for action in best_actions:
                if action in available_actions:
                    best_action = action
                    break

            # transform action to position inside Board
            rows, columns = self.board.shape
            for i in range(rows):
                for j in range(columns):
                    best_action -= 1
                    if best_action == 0:
                        return i, j
        else:
            # state not exists return MCS
            return self.monte_carlo_action()

    def add_state_history(self, action):
        # add hashed state of board with action toked in history of states
        # get action ID
        rows, columns = self.board.shape

        action_id = 0
        broke = False
        for i in range(0, rows):
            for j in range(0, columns):
                action_id += 1
                if (i, j) == action:
                    broke = True
                    break
            if broke:
                break

        self.states_history.append((self.hash_board(), action_id))

    def hash_board(self):
        rows, columns = self.board.shape
        return str(self.board.reshape(rows * columns))

    def win(self):
        rows, columns = self.board.shape

        # row winner
        for i in range(rows):
            if not any(value != self.symbol for value in self.board[i, :]):
                return True

        # column winner
        for i in range(columns):
            if not any(value != self.symbol for value in self.board[:, i]):
                return True

        # diagonal winner
        diagonal1 = [self.board[i, i] for i in range(columns)]
        diagonal2 = [self.board[i, columns - i - 1] for i in range(columns)]
        if not any(value != self.symbol for value in diagonal1) or not any(value != self.symbol for value in diagonal2):
            return True

        return False

    def update_QTable(self, reward):

        # loop in reverse of (states, actions) history
        reverse_states_history = self.states_history[::-1]
        for i in range(len(reverse_states_history)):
            state, action = reverse_states_history[i]  # current state, action

            # get next state (if exists)
            next_state = None
            if i != 0:
                next_state, _ = reverse_states_history[i - 1]

            # get best action for next state
            if next_state is not None:
                next_action = max(self.QTable[next_state], key=self.QTable[next_state].get)
                next_value = self.QTable[next_state][next_action]
            else:
                next_value = 0

            # update QTable
            last_value = (1 - ALPHA) * self.QTable[state][action]
            self.QTable[state][action] = last_value + ALPHA * (reward + DISCOUNT_RATE * next_value)

            # reward is 0 for all actions except the last action
            reward = 0

        # delete all old history of states, action played
        self.states_history.clear()

    def savePolicy(self):
        fw = open('policy_' + str(self.symbol), 'wb')
        pickle.dump(self.QTable, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.QTable = pickle.load(fr)
        fr.close()


class MonteCarloSearch:
    def __init__(self, player_symbol, board):
        self.board = board
        if player_symbol == 1:
            self.player_1 = 1
            self.player_2 = 2
        else:
            self.player_1 = 2
            self.player_2 = 1

    @staticmethod
    def player_win(board, player_symbol):
        rows, columns = board.shape

        # row winner
        for i in range(rows):
            if not any(value != player_symbol for value in board[i, :]):
                return True

        # column winner
        for i in range(columns):
            if not any(value != player_symbol for value in board[:, i]):
                return True

        # diagonal winner
        diagonal1 = [board[i, i] for i in range(columns)]
        diagonal2 = [board[i, columns - i - 1] for i in range(columns)]

        # diagonal conditions to win
        diagonal1_condition = not any(value != player_symbol for value in diagonal1)
        diagonal2_condition = not any(value != player_symbol for value in diagonal2)

        if diagonal1_condition or diagonal2_condition:
            return True

        return False

    @staticmethod
    def player_draw(board):
        if len(np.where(board == 0)[0]) == 0:
            return True
        else:
            return False

    @staticmethod
    def hash_board(board):
        rows, columns = board.shape
        return str(board.reshape(rows * columns))

    @staticmethod
    def get_available_positions(board):
        available_positions = []
        rows, columns = board.shape
        for i in range(rows):
            for j in range(columns):
                if board[i, j] == 0:
                    available_positions.append((i, j))
        return available_positions

    def random_action(self, board):
        # create a list of available positions
        available_positions = self.get_available_positions(board)
        return random.choice(available_positions)

    def game_end(self, board):
        if not self.player_win(board, 1) and not self.player_win(board, 2) and not self.player_draw(board):
            return False
        else:
            return True

    def random_actions(self, board, action):
        # make first action inside
        i, j = action
        board[i][j] = self.player_1

        # perform random actions
        turn = self.player_2

        # loop while not end of game
        while not self.game_end(board):

            i, j = self.random_action(board)

            # player 1 choose action & update game board
            if turn == self.player_1:
                board[i, j] = self.player_1
            else:
                board[i, j] = self.player_2

            # change player turn
            turn = turn % 2 + 1

        # return specific reward
        if self.player_draw(board):
            return REWARD_P1_DRAW
        if self.player_win(board, self.player_1):
            return REWARD_P1_WIN
        if self.player_win(board, self.player_2):
            return REWARD_P1_LOOSE

    @property
    def search(self):
        # init available positions
        available_positions = self.get_available_positions(self.board)

        if len(available_positions) > 1:
            # init scores, visits
            scores = [0] * len(available_positions)
            visits = [0] * len(available_positions)

            # loop inside available positions until MCS_TIME is out
            timeout = time.time() + MCS_TIME/100  # 50 ms from now
            i = 0
            while time.time() < timeout:
                reward = self.random_actions(np.copy(self.board), available_positions[i])
                scores[i] += reward
                visits[i] += 1
                i = (i+1) % len(available_positions)

            # loop inside available positions to calculate best action
            expected_scores = [0] * len(available_positions)
            for i in range(len(available_positions)):
                if visits[i] > 0:
                    expected_scores[i] = scores[i] / visits[i]

            # return best action from expected_scores
            return available_positions[np.argmax(expected_scores)]
        else:
            return available_positions[0]


class Game:
    def __init__(self, player_1, player_2, rows, columns, time_limit=0, begin_iteration=0):
        # init players & game params
        self.player_1 = player_1
        self.player_2 = player_2
        self.G_time = time_limit
        self.begin_iteration = begin_iteration

        # init game board
        self.board = np.zeros((rows, columns)).astype(int)
        self.player_1.board = self.board
        self.player_2.board = self.board

    def e_greedy(self, m):
        e = A * math.cos((m / (2 * self.G_time)) * math.pi) + B
        num = random.random()
        if num < e:
            return True
        else:
            return False

    def isEnd(self):
        if not self.player_1.win() and not self.player_2.win() and not self.isDraw():
            return False
        else:
            if self.player_1.win():
                self.player_1.update_QTable(REWARD_P1_WIN)
                self.player_2.update_QTable(REWARD_P2_LOOSE)
            if self.player_2.win():
                self.player_2.update_QTable(REWARD_P2_WIN)
                self.player_1.update_QTable(REWARD_P1_LOOSE)
            if self.isDraw():
                self.player_1.update_QTable(REWARD_P1_DRAW)
                self.player_2.update_QTable(REWARD_P2_DRAW)
            return True

    def isDraw(self):
        if len(np.where(self.board == 0)[0]) == 0:
            return True
        else:
            return False

    def play(self):
        # random choice for player who start the match
        turn = random.choice([1, 2])
        counter = 0

        # loop while not end of game
        while not self.isEnd():
            # calculate e_greedy for the current game
            e_greedy = self.e_greedy(counter + self.begin_iteration + 1)

            # player choose action & update game board
            if turn == 1:
                action = self.player_1.choose_action(self.board, e_greedy)
                self.update_board(action, self.player_1.symbol)
            else:
                action = self.player_2.choose_action(self.board, e_greedy)
                self.update_board(action, self.player_2.symbol)

            # change player turn
            turn = turn % 2 + 1
            counter += 1

        return self.begin_iteration + counter

    def update_board(self, action, player_symbol):
        i, j = action
        self.board[i][j] = player_symbol

        # update board also for 2 players
        self.player_1.board = self.board
        self.player_2.board = self.board


class Training:
    def __init__(self, rows, columns, time_limit):
        self.rows = rows
        self.columns = columns
        self.G_time = time_limit
        self.player_1 = Player(symbol=1)
        self.player_2 = Player(symbol=2)

    def e_greedy(self, m):
        e = A * math.cos((m / (2 * self.G_time)) * math.pi) + B
        num = random.random()
        if num < e:
            return True
        else:
            return False

    def train(self):
        i = 0
        while i < self.G_time:
            # play new game between player_1 Vs player_2
            print(' --- epochs ', i, ' --- ')
            game = Game(player_1=self.player_1, player_2=self.player_2,
                        time_limit=self.G_time, begin_iteration=i,
                        rows=self.rows, columns=self.columns)
            iterations = game.play()
            i = iterations


if __name__ == "__main__":
    # training Tic Tac Toe Game ( 3 x 3 )
    training = Training(rows=3, columns=3, time_limit=50000)

    # train reinforcement learning model
    training.train()

    # save QTable of 2 players after training
    training.player_1.savePolicy()
    training.player_2.savePolicy()
