"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

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
    xCount = 0
    oCount = 0

    # Counting moves made by each player to determine current turn
    for row in board:
        xCount += row.count(X)
        oCount += row.count(O)
    
    if xCount > oCount:
        return O
    return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == EMPTY:
                actions.add((i, j))
    
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i = action[0]
    j = action[1]

    # Error handling for negative indexes, out of bound moves, and taken spaces
    if i < 0 or j < 0: raise Exception
    if terminal(board) or board[i][j] != EMPTY: raise Exception
    try: board[i][j]
    except: raise Exception

    # Using deepcopy() to ensure completely new objects are created
    newBoard = deepcopy(board)
    newBoard[i][j] = player(board)

    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winConditions = [[(0, 0), (0, 1), (0, 2)],
                    [(1, 0), (1, 1), (1, 2)],
                    [(2, 0), (2, 1), (2, 2)],
                    [(0, 0), (1, 0), (2, 0)],
                    [(0, 1), (1, 1), (2, 1)],
                    [(0, 2), (1, 2), (2, 2)],
                    [(0, 0), (1, 1), (2, 2)],
                    [(0, 2), (1, 1), (2, 0)]]

    xCoords = []
    oCoords = []

    # Populating lists with coordinates of moves made by X and O
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == X:
                xCoords.append((i, j))
            elif cell == O:
                oCoords.append((i, j))

    # Evaluating if any coordinates supplied meet win conditions
    for moveset in winConditions:
        if all(coords in xCoords for coords in moveset):
            return X
        elif all(coords in oCoords for coords in moveset):
            return O  
    return None
        

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None or len(actions(board)) == 0:
        return True
    
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else: return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board,
    making use of Alpha-Beta Pruning to optimise the algorithm.
    """
    
    alpha = -math.inf
    beta = math.inf

    if terminal(board):
        return None
    elif player(board) == X:
        return maxValue(board, alpha, beta, init=True)
    elif player(board) == O:
        return minValue(board, alpha, beta, init=True)
    

def maxValue(board, alpha, beta, init=False):
    """
    Recursively compares the utility of every possible action that can 
    be performed on the board to the lowest bound, updating as higher 
    utility actions are found. Returning the optimal action to perform 
    when called outside of this function.
    """
    if terminal(board):
        return utility(board)
    
    for action in actions(board):
        actionValue = minValue(result(board, action), alpha, beta)
        if actionValue > alpha:
            alpha = actionValue
            optimalAction = action

        # If utility is less than or equal to the best possible utility for 
        # max player, exits loop as optimal action for max player has been found
        if alpha >= beta:
            break
    if init:
        return optimalAction
    return alpha


def minValue(board, alpha, beta, init=False):
    """
    Recursively compares the utility of every possible action that can 
    be performed on the board to the highest bound, updating as lower 
    utility actions are found. Returning the optimal action to perform 
    when called outside of this function.
    """
    if terminal(board):
        return utility(board)

    for action in actions(board):
        actionValue = maxValue(result(board, action), alpha, beta)
        if actionValue < beta:
            beta = actionValue
            optimalAction = action

        # If utility is greater than or equal to the best possible utility for 
        # min player, exits loop as optimal action for min player has been found
        if beta <= alpha:
            break
    if init:
        return optimalAction
    return beta