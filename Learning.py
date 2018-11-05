import random as rand
import util


class QLearningAgent:
    """
    Q-Learning Agent

    Abstract agent which assigns values to (state,action)
    Q-Values for an environment. As well as a value to a
    state and a policy given respectively by,

    V(s) = max_{a in actions} Q(s,a)
    policy(s) = arg_max_{a in actions} Q(s,a)

    The QLearningAgent estimates
    Q-Values while acting in the environment.
    """
    def __init__(self, player, featExtractor, numTraining=1000, epsilon=0.05, alpha=1, gamma=0.8):
        """
        actionFn: Function which takes a state and returns the list of legal actions
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.episodesSoFar = 0
        self.numTraining = numTraining
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = gamma
        self.player = player
        self.Q = {}
        self.w = util.Counter()
        self.featExtractor = featExtractor

    def getQValue(self, state, action):
        return sum([self.featExtractor.getFeatures(state, action)[feature] * self.w[feature]
                    for feature in self.featExtractor.getFeatures(state, action)])

    def getValue(self, board):
        max_val = {}
        for action in board.legalMoves(self.player):
            max_val[action] = self.getQValue(board, action)
        return max_val[max(max_val)]

    def getPolicy(self, board):
        best = []
        best_val = - float('inf')
        for action in board.legalMoves(self.player):
            if self.getQValue(board, action) > best_val:
                best = [action]
                best_val = self.getQValue(board, action)
            elif self.getQValue(board, action) == best_val:
                best.append(action)
        if len(best) == 0:
            return None
        else:
            return rand.choice(best)

    def getAction(self, board):
        # Observe the board
        self.observationFunction(board)
        # Pick Action
        legalActions = board.legalMoves(self.player)
        action = None
        if len(legalActions) > 0:
            eps = rand.random()
            if ( eps < self.epsilon):
                action = rand.choice(legalActions)
            else:
                action = self.getPolicy(board)
        return action

    def update(self, state, action, nextState, reward):
        next_step = self.getValue(nextState)
        # update w
        correction = reward + (self.discount * next_step) - self.getQValue(state, action)
        for feature in self.featExtractor.getFeatures(state, action):
            self.w[feature] += self.featExtractor.getFeatures(state, action)[feature] * correction * self.alpha

    def observationFunction(self, state):
        if not self.lastState is None:
            reward = state.getScore(self.player.num) - self.lastState.getScore(self.player.num)
            if reward == 0:
                reward = -1
            self.update(self.lastState, self.lastAction, state, reward)

    def update_current_state(self, board, move):
        self.lastState = board.clone()
        self.lastAction = move

    def startEpisode(self):
        self.lastState = None
        self.lastAction = None

    def stopEpisode(self):
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining




