import util

# Heuristic n.0: The null heuristic - return 0 for every board position
def null_heuristic(player, board):
    return 0

# Heuristic n.1: This heuristic uses the formula: My_marbles - Opp_marbles
def points_diff_heuristic(player, board):
    if board is not None:
        if player == 1:
            return (board.scoreCups[player-1] - board.scoreCups[1])
        else: # playerNum == 2
            return (board.scoreCups[player-1] - board.scoreCups[0])
    return 0

# Heuristic n.2: This heuristic will prefer a move which would increase the
# number of player's captured marbles
def max_my_points(player, board):
    if board is not None:
        return board.scoreCups[player-1]
    return 0

# Heuristic n.3: This heuristic will prefer a move which would decrease the
# number of player's opponent captured marbles
def min_opp_points (player, board):
    if board is not None:
        if player == 1:
            return (48 - board.scoreCups[1])
            # 48 is the sum of all the marbles on the board
        else: # playerNum == 2
            return (48 - board.scoreCups[0])
    return 0


# Heuristic n.4: Returns 1 if this move earned the player with extra turn
def max_repeat_turns(player, board):
    if board.repeated_turn:
        return 1
    return 0

# Heuristic n.5: Returns the number of marbles in the player's cups
def max_my_playing_marbles(player, board):
    for i in range(6):
        if player == 1:
            return sum(s for s in board.P1Cups)
        else: # playerNum == 2
            return sum(s for s in board.P2Cups)
    return 1

# Heuristic n.6:
def mixed_heuristic(player, board):
    if player == 1:
        cups = board.P1Cups
    else:
        cups = board.P2Cups
    for i in range(6):
        if cups[5-i] == i + 1:
            return 12 - i
    for i in range(6):
        if cups[5-i] != 0:
            return (6 - i)
    return 0

class FeatureExtractor:
    def __init__(self, player):
        self.player = player
        self.playerNum = player.num

    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats

class oneExtractor(FeatureExtractor):

    def __init__(self, player):
        self.player = player
        self.playerNum = player.num
        self.func = player.scoreFunc

    def getFeatures(self, state, action):
        board = state.clone()
        board.makeMoveHelp(self.player, action)
        features = util.Counter()
        features["bias"] = 1.0
        features["value-func"] = self.func(self.playerNum, board) / 10
        features.divideAll(10.0)
        return features

class SimpleExtractor(FeatureExtractor):

    def getFeatures(self, state, action):
        board = state.clone()
        board.makeMoveHelp(self.player, action)
        features = util.Counter()
        features["bias"] = 1.0
        features["max-my-marbles"] = max_my_points(self.playerNum, board)
        features["min-opp-marbles"] = min_opp_points(self.playerNum, board)
        features.divideAll(10.0)
        return features