# @author Metro
# @time 2021/10/29

class Agent(object):
    """
    Define a basic reinforcement learning agent
    """

    NAME = "Abstract Agent"

    def __init__(self):


    def act(self, state):
        """
        Determines which action to take when given state

        :param state:
        :return:
        """
        raise NotImplementedError

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps):
        """
        Performs a learning step given a (s, a, r, s', a') transition

        :param state: previous observed state (s)
        :param action: action taken in previous state (a)
        :param reward: reward for the transition (r)
        :param next_state: the resulting observed state (s')
        :param next_action: action taken in next state (a')
        :param terminal: whether the episode is over
        :param time_steps: number of time steps the action took to execute (default=1)
        :return:
        """
        raise NotImplementedError

    def start_episode(self):
        """
        Perform any initialisation for the start of an episode.
        :return:
        """
        raise NotImplementedError

    def end_episode(self):
        """
        Performs any cleanup before the next episode.
        :return:
        """
        raise NotImplementedError

    def __str__(self):
        desc = self.NAME
        return desc
