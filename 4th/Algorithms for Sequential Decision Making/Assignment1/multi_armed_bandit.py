import numpy as np


class bernoulli_arm:
    """A Bernoulli arm. The reward is 1 with probability p, and 0 with probability 1-p."""

    def __init__(self, p, index):
        self.__p = p
        self.index = index

    def pull(self):
        return np.random.binomial(1, self.__p)

    def _get_true_mean(self):
        print(
            "WARNING: Calling get_true_mean() is not allowed in your final submission"
        )
        return self.__p

    def __repr__(self):
        return "Bernoulli arm with index " + str(self.index)


class gaussian_arm:
    """A Gaussian arm. The reward is drawn from a normal distribution with mean mu and standard deviation sigma."""

    def __init__(self, mu, sigma, index):
        assert (
            mu - 4 * sigma >= 0
        ), "The mean mu should be at least 4 times the standard deviation sigma."
        self.__mu = mu
        self.__sigma = sigma
        self.index = index

    def pull(self):
        reward = np.random.normal(self.__mu, self.__sigma)
        return max(reward, 0)

    def _get_true_mean(self):
        print(
            "WARNING: Calling get_true_mean() is not allowed in your final submission"
        )
        return self.__mu

    def __repr__(self):
        return "Gaussian arm with index " + str(self.index)


class uniform_arm_discrete:
    """A uniform arm with discrete rewards. The reward is an integer between low and high."""

    def __init__(self, low, high, index):
        assert low < high, "The lower bound should be less than the upper bound."
        assert low >= 0, "The lower bound should be non-negative."
        self.__low = low
        self.__high = high
        self.index = index

    def pull(self):
        return np.random.randint(
            self.__low, self.__high + 1
        )  # By default, numpy.random.randint generates random integers from the “discrete uniform” distribution.

    def _get_true_mean(self):
        print(
            "WARNING: Calling get_true_mean() is not allowed in your final submission"
        )
        return (self.__low + self.__high) / 2

    def __repr__(self):
        return "Uniform discrete arm with index " + str(self.index)


class uniform_arm_continuous:
    """A uniform arm with continuous rewards. The reward is a float between low and high."""

    def __init__(self, low, high, index):
        assert low < high, "The lower bound should be less than the upper bound."
        assert low >= 0, "The lower bound should be non-negative."
        self.__low = low
        self.__high = high
        self.index = index

    def pull(self):
        return np.random.uniform(self.__low, self.__high)

    def _get_true_mean(self):
        print(
            "WARNING: Calling get_true_mean() is not allowed in your final submission"
        )
        return (self.__low + self.__high) / 2

    def __repr__(self):
        return "Uniform continuous arm with index " + str(self.index)
