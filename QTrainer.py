import MancalaGUI
import util

class QTrainer:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    def __init__(self, agent, opp):
        self.agent = agent
        self.opp = opp
        self.gameOver = False

    def train(self):
        """
        Main control loop for game play.
        """
        wins = util.Counter()
        i = 0
        while self.agent.agent.isInTraining():
            self.agent.agent.startEpisode()
            wins[MancalaGUI.startGameNoGUI(self.agent, self.opp, i)] += 1
            i += 1
            self.agent.agent.stopEpisode()


