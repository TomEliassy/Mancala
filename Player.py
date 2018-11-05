from random import *
from decimal import *
from copy import *
import time

from MancalaBoard import *
import EvaluationFunctions


class Player:
    HUMAN = 0
    RANDOM = 1
    MINIMAX = 2
    ABPRUNE = 3
    CUSTOM = 4
    MIX = 5

    MAX_PLAYER = 1
    MIN_PLAYER = 2

    """ A basic AI (or human) player """
    def __init__(self, playerNum, playerType, opp, depth=1,
                 scoreFunc=EvaluationFunctions.points_diff_heuristic, agent=None):
        """Initialize a Player with a playerNum (1 or 2), playerType (one of
        the constants such as HUMAN), and a ply (default is 0)."""
        self.num = playerNum
        self.opp = opp
        self.opp_num = 2 - playerNum + 1
        self.type = playerType
        self.depth = depth
        self.scoreFunc = scoreFunc
        self.agent = agent


    def __repr__(self):
        """Returns a string representation of the Player."""
        if self.type == Player.HUMAN:
            return("Human")
        elif self.type == Player.RANDOM:
            return ("Random")
        elif self.type == Player.MINIMAX:
            return ("Minimax")
        elif self.type == Player.ABPRUNE:
            return ("ab Pruning")
        elif self.type == Player.CUSTOM:
            return "Q-Learner"
        elif self.type == Player.MIX:
            return "MIX"

    def set_learning_values(self, epsilon, alpha, gamma):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def minimax_move(self, board, depth, turn):
        if board.gameOver() or depth == 0:
            return (self.scoreFunc(self.num, board), )
        scores = dict()
        # make a new player to play the other side
        opponent = Player(self.opp_num, self.type, self.MINIMAX,
                          depth - 1, self.scoreFunc)
        if turn == Player.MAX_PLAYER:
            for action in board.legalMoves(self):
                # make a new board
                nb = deepcopy(board)
                # try the move
                nb.makeMove(self, action)
                scores[action] = (opponent.minimax_move(nb, depth - 1,
                                                        Player.MIN_PLAYER))[0]
            chosen_action = max(scores, key=scores.get)
            return max(scores.values()), chosen_action
        else:
            for action in board.legalMoves(self):
                # make a new board
                nb = deepcopy(board)
                # try the move
                nb.makeMove(self, action)
                scores[action] = (opponent.minimax_move(nb, depth - 1,
                                                        Player.MAX_PLAYER))[0]
            chosen_action = min(scores, key=scores.get)
            return min(scores.values()), chosen_action

    def alpha_beta_move(self, board, depth, alpha, beta, turn):
        if board.gameOver() or depth == 0:
            return (self.scoreFunc(self.num, board), )
        chosen_action = None
        opponent = Player(self.opp_num, self.type, self.ABPRUNE,
                          depth - 1, self.scoreFunc)
        if turn == Player.MAX_PLAYER:
            value = float('-inf')
            for action in board.legalMoves(self):
                # make a new board
                nb = deepcopy(board)
                # try the move
                nb.makeMove(self, action)
                cand_value = (opponent.alpha_beta_move(nb, depth - 1, alpha,
                                                       beta, Player.MIN_PLAYER))[0]
                if (value < cand_value):
                    chosen_action = action
                    value = cand_value
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, chosen_action
        else:
            value = float('inf')
            for action in board.legalMoves(self):
                # make a new board
                nb = deepcopy(board)
                # try the move
                nb.makeMove(self, action)
                cand_value = (opponent.alpha_beta_move(nb, depth - 1, alpha,
                                                       beta, Player.MAX_PLAYER))[0]
                if (value > cand_value):
                    chosen_action = action
                    value = cand_value
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, chosen_action

    def mixed_move(player, board):
        if player.num == 1:
            cups = board.P1Cups
        else:
            cups = board.P2Cups
        for i in range(6):
            if cups[5 - i] == i + 1:
                return 6 - i
        for i in range(6):
            if cups[5 - i] != 0:
                return (6 - i)
        return 0

    def choose_move(self, board):
        """ Returns the next move that this player wants to make """
        if self.opp == Player.HUMAN:
            time.sleep(4)
        if self.type == Player.HUMAN:
            move = input("Please enter your move:")
            while not board.legalMove(self, move):
                print(move, "is not valid")
                move = input("Please enter your move")
            return move
        elif self.type == Player.RANDOM:
            move = choice(board.legalMoves(self))
            return move
        elif self.type == Player.MINIMAX:
            val, move = self.minimax_move(board, self.depth * 2,
                                         Player.MAX_PLAYER)
            board.last_move = move
            return move
        elif self.type == Player.ABPRUNE:
            val, move = self.alpha_beta_move(board, self.depth * 2,
                                             float('-inf'), float('inf'),
                                             Player.MAX_PLAYER)
            return move
        elif self.type == Player.CUSTOM:
            move = self.agent.getAction(board)
            self.agent.update_current_state(board, move)
            return move
        elif self.type == Player.MIX:
            return self.mixed_move(board)

        else:
            print("Unknown player type")
            return -1