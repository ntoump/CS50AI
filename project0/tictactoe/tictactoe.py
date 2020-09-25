"""
Tic Tac Toe Player
"""

import math
import copy
from time import perf_counter

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    # Return None if the board is terminal
    if terminal(board):
        return None

    # X gets the first move
    if board == initial_state():
        return X

    # These two variables are counters of the total number of X and O, respectively, found in the given board
    totalx = 0
    totalo = 0
    for row in board:
        totalx += row.count(X)
        totalo += row.count(O)
    # This returns O for O's turn and X for X's
    if totalx == totalo + 1:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    # Return None if the board is terminal
    if terminal(board):
        return None

    # The possible actions for each provided board are stored in this set
    possible_actions = set()

    # Returns in tuples the possible actions, i.e. wherever there is "EMPTY" on the board
    rows = 0
    for row in board:
        counter = 0
        for col in row:
            if col is None:
                possible_actions.add((rows, counter))
            counter += 1
        rows += 1
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # This raises Exception for invalid action
    if action not in actions(board):
        print("actions(board):", actions(board), "action:", action)
        raise Exception

    # Else: temp becomes a deep copy of the given board, the player(temp) is called and the respective symbol is marked
    else:
        temp = copy.deepcopy(board)
        p = player(temp)
        temp[action[0]][action[1]] = p
        return temp


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    # Loops that check for win conditions
    candidates = [X, O]
    for c in candidates:
        for i in range(0, 3):

            # Checking horizontally
            if c == board[i][0] and c == board[i][1] and c == board[i][2]:
                return c

            # Checking vertically
            elif c == board[0][i] and c == board[1][i] and c == board[2][i]:
                return c

            # Checking diagonally #1
            elif c == board[0][0] and c == board[1][1] and c == board[2][2]:
                return c

            # Checking diagonally #2
            elif c == board[0][2] and c == board[1][1] and c == board[2][0]:
                return c
    # If no winner (in total or yet), return None
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    # This checks in each row for None, which is equal to "EMPTY" in this instance, if there is not a winner
    count = 0
    if winner(board) is not None:
        return True
    else:
        for i in board:
            count += 1
            # If "EMPTY" is in a row, the game is not yet over
            if None in i:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win is X:
        return 1
    elif win is O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # If board is terminal, return None
    t1 = perf_counter()

    # If it is X's turn
    if player(board) == X:
        v = -math.inf

        # Examine all possible actions, calling the min_value
        for action in actions(board):
            optimal = min_value(result(board, action))

            # Find best move, based on biggest evaluation of resulting board
            if optimal > v:
                v = optimal
                best_move = action

    # If it is O's turn
    elif player(board) == O:
        v = math.inf

        # Examine all possible actions, calling the max_value
        for action in actions(board):
            optimal = max_value(result(board, action))

            # Find best move, based on smallest evaluation of resulting board
            if optimal < v:
                v = optimal
                best_move = action
    t2 = perf_counter()
    print("time (sec):", t2-t1)  # This is a counter of how much time it took Minimax to calculate the best move
    return best_move


# The max_value function
def max_value(board):
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


# The min_value function
def min_value(board):
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
