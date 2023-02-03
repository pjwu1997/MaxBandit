# -*- coding: utf-8 -*-
r""" An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.

- Reference: [On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems, by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)

- It uses an additional :math:`\mathcal{O}(\tau)` memory but do not cost anything else in terms of time complexity (the average is done with a sliding window, and costs :math:`\mathcal{O}(1)` at every time step).

.. warning:: This is very experimental!
.. note:: This is similar to :class:`SlidingWindowRestart.SWR_UCB` but slightly different: :class:`SlidingWindowRestart.SWR_UCB` uses a window of size :math:`T_0=100` to keep in memory the last 100 *draws* of *each* arm, and restart the index if the small history mean is too far away from the whole mean, while this :class:`SWUCB` uses a fixed-size window of size :math:`\tau=1000` to keep in memory the last 1000 *steps*.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

from math import log, sqrt
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy

from scipy.stats import norm,uniform
import scipy.stats as ss
import copy
from bisect import bisect, insort

#: Size of the sliding window. (DEFAULT = 2)
TAU = 2

#: Default value for the constant :math:`\alpha`.
ALPHA = 1.0

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)

def mahalanobis_distance(model_1, model_2):
    """
    Return the mahalanobis distance of two normal distributions.
    """
    mean_1 = model_1.mean
    mean_2 = model_2.mean
    std_1 = model_1.std
    std_2 = model_2.std
    distance = 1 / 8 * ((mean_1 - mean_2) ** 2) * (2 / (std_1 + std_2)) + 2 * np.log((std_1 + std_2) / (2 * np.sqrt(std_1 * std_2)))

class MaxMedian(IndexPolicy):
    """ An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.
    """

    def __init__(self, nbArms, budget,
                 *args, **kwargs):
        super(MaxMedian, self).__init__(nbArms, *args, **kwargs)
        # New parameters
        # assert 1 <= tau, "Error: parameter 'tau' for class SWUCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        # self.tau = int(tau)  #: Size :math:`\tau` of the sliding window.
        # assert alpha > 0, "Error: parameter 'alpha' for class SWUCB has to be > 0, but was {}.".format(alpha)  # DEBUG
        # self.alpha = alpha  #: Constant :math:`\alpha` in the square-root in the computation for the index.
        # Internal memory
        # self.last_rewards = [] ## np.zeros((nbArms,2 * tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.nbArms = nbArms
        self.t = 0
        self.T = budget
        self.Na = np.zeros(self.nbArms, dtype='int')
        # self.maxima = dict(zip(np.arange(self.nbArms), [{} for _ in range(self.nbArms)]))
        self.n = np.zeros(self.nbArms, dtype=np.int32)
        self.sorted_rewards_arm = [[] for _ in range(self.nbArms)]
    
    def explo_func(self, t):
        return 1 / t

    def __str__(self):
        return f"Maxmedian"

    def getReward(self, arm, reward):
        self.Na[arm] += 1
        self.t += 1
        insort(self.sorted_rewards_arm[arm], reward)
    
    def computeIndex(self, arm):
        return 0

    def choice(self):
        if self.t < self.nbArms:
            return self.t % self.nbArms
        else:
            if np.random.binomial(1, self.explo_func(self.t)) == 1:
                k = np.random.randint(self.nbArms)
            else:
                m = self.Na.min()
                orders = np.ceil(self.Na/m).astype(np.int32)
                idx = [self.sorted_rewards_arm[i][-orders[i]] for i in range(self.nbArms)]
                k = rd_argmax(np.array(idx))
            return k
