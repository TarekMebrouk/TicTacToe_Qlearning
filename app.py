from training import *
import pygame
import sys

WIDTH = 600
HEIGHT = 600
LINE_WIDTH = 8
WIN_LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = 200
CIRCLE_RADIUS = 60
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = 55
# rgb: red green blue
BG_COLOR = (51, 125, 239)
LINE_COLOR = (255, 255, 255)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (252, 199, 41)


class App:
    def __init__(self):
        # init game board & players
        self.player_1 = Player(symbol=1)
        self.player_2 = Player(symbol=2)

        self.board = np.zeros((BOARD_ROWS, BOARD_COLS)).astype(int)
        self.player_1.board = self.board
        self.player_2.board = self.board

        # load policy for player_1 "machine"
        self.player_1.loadPolicy('policy_1')

        # load game ui pygame
        pygame.init()

        # init screen ui
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('')
        self.screen.fill(BG_COLOR)
        self.draw_lines()

    def draw_lines(self):
        # 1 horizontal
        pygame.draw.line(self.screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
        # 2 horizontal
        pygame.draw.line(self.screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH)

        # 1 vertical
        pygame.draw.line(self.screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
        # 2 vertical
        pygame.draw.line(self.screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

    def draw_figures(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 1:
                    pygame.draw.circle(self.screen, CIRCLE_COLOR, (
                        int(col * SQUARE_SIZE + SQUARE_SIZE // 2), int(row * SQUARE_SIZE + SQUARE_SIZE // 2)),
                                       CIRCLE_RADIUS, CIRCLE_WIDTH)
                elif self.board[row][col] == 2:
                    pygame.draw.line(self.screen, CROSS_COLOR,
                                     (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                     (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
                    pygame.draw.line(self.screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE),
                                     (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                     CROSS_WIDTH)

    def mark_square(self, row, col, player_symbol):
        self.board[row][col] = player_symbol

    def available_square(self, row, col):
        return self.board[row][col] == 0

    def is_board_full(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    return False

        return True

    def draw_vertical_winning_line(self, col, player_symbol):
        pos_x = col * SQUARE_SIZE + SQUARE_SIZE // 2

        color = 1
        if player_symbol == 1:
            color = CIRCLE_COLOR
        elif player_symbol == 2:
            color = CROSS_COLOR

        pygame.draw.line(self.screen, color, (pos_x, 15), (pos_x, HEIGHT - 15), LINE_WIDTH)

    def draw_horizontal_winning_line(self, row, player_symbol):
        pos_y = row * SQUARE_SIZE + SQUARE_SIZE // 2

        color = 1
        if player_symbol == 1:
            color = CIRCLE_COLOR
        elif player_symbol == 2:
            color = CROSS_COLOR

        pygame.draw.line(self.screen, color, (15, pos_y), (WIDTH - 15, pos_y), WIN_LINE_WIDTH)

    def draw_asc_diagonal(self, player_symbol):
        color = 1
        if player_symbol == 1:
            color = CIRCLE_COLOR
        elif player_symbol == 2:
            color = CROSS_COLOR

        pygame.draw.line(self.screen, color, (15, HEIGHT - 15), (WIDTH - 15, 15), WIN_LINE_WIDTH)

    def draw_desc_diagonal(self, player_symbol):
        color = 1
        if player_symbol == 1:
            color = CIRCLE_COLOR
        elif player_symbol == 2:
            color = CROSS_COLOR

        pygame.draw.line(self.screen, color, (15, 15), (WIDTH - 15, HEIGHT - 15), WIN_LINE_WIDTH)

    def restart(self):
        # init screen
        self.screen.fill(BG_COLOR)
        self.draw_lines()
        pygame.display.set_caption('')

        # init board to restart new game
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                self.board[row][col] = 0

        # update board for players
        self.player_1.board = self.board
        self.player_2.board = self.board

    def isEnd(self):
        if not self.player_1.win() and not self.player_2.win() and not self.isDraw():
            return False
        else:
            return True

    def draw_winning_line(self):
        winner = -1

        # get winner player symbol
        if self.player_1.win():
            winner = 1
        if self.player_2.win():
            winner = 2

        # if match is not a draw
        if winner != -1:
            rows, columns = self.board.shape

            # row winner
            for i in range(rows):
                if all(value == winner for value in self.board[i, :]):
                    self.draw_horizontal_winning_line(i, winner)  # draw horizontal winning line
                    return

            # column winner
            for i in range(columns):
                if all(value == winner for value in self.board[:, i]):
                    self.draw_vertical_winning_line(i, winner)  # draw vertical winning line
                    return

            # diagonal winner
            diagonal1 = [self.board[i, i] for i in range(columns)]
            diagonal2 = [self.board[i, columns - i - 1] for i in range(columns)]

            if all(value == winner for value in diagonal1):
                self.draw_desc_diagonal(winner)  # draw descendant diagonal winning line
                return

            if all(value == winner for value in diagonal2):
                self.draw_asc_diagonal(winner)  # draw ascendant diagonal winning line
                return

    def isDraw(self):
        if len(np.where(self.board == 0)[0]) == 0:
            return True
        else:
            return False

    def update_board(self, action, player_symbol):
        i, j = action
        self.board[i][j] = player_symbol

        # update board also for 2 players
        self.player_1.board = self.board
        self.player_2.board = self.board

        # mark square selected
        self.mark_square(i, j, player_symbol)
        self.draw_figures()

    def display_game_result(self):
        # display state of game
        # 1- winning
        if self.player_1.win():
            pygame.display.set_caption('Machine win ! ')
        if self.player_2.win():
            pygame.display.set_caption('Human win !')
        # 2- draw
        if self.isDraw():
            pygame.display.set_caption('Draw')

    def play(self):

        # take random turn : Machine or player 2 : Human
        turn = random.choice([1, 2])

        while True:
            # listen for mouse click event
            for event in pygame.event.get():

                # exit click 'QUIT'
                if event.type == pygame.QUIT:
                    sys.exit()

                # check if end of game
                if not self.isEnd():
                    # player choose action & update game board
                    if turn == 1:
                        self.player_1.update_available_positions()
                        action = self.player_1.GBest_action()
                        self.update_board(action, self.player_1.symbol)
                        turn = turn % 2 + 1  # change player turn
                    else:
                        # take human action
                        if event.type == pygame.MOUSEBUTTONDOWN:

                            # get row, column selected square
                            mouse_x = event.pos[0]  # x
                            mouse_y = event.pos[1]  # y

                            clicked_row = int(mouse_y // SQUARE_SIZE)
                            clicked_col = int(mouse_x // SQUARE_SIZE)

                            if self.available_square(clicked_row, clicked_col):
                                self.update_board((clicked_row, clicked_col), self.player_2.symbol)
                                turn = turn % 2 + 1  # change player turn
                else:
                    self.draw_winning_line()  # display winning line
                    self.display_game_result()  # display match result

                # restart new game
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.restart()
                        turn = random.choice([1, 2])

            pygame.display.update()


if __name__ == "__main__":
    application = App()
    application.play()
