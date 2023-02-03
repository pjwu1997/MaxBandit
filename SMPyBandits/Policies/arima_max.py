# -*- coding: utf-8 -*-
r""" An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.

- Reference: [On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems, by A.Garivier & E.Moulines, ALT 2011](https://arxiv.org/pdf/0805.3415.pdf)

- It uses an additional :math:`\mathcal{O}(\tau)` memory but do not cost anything else in terms of time complexity (the average is done with a sliding window, and costs :math:`\mathcal{O}(1)` at every time step).

.. warning:: This is very experimental!
.. note:: This is similar to :class:`SlidingWindowRestart.SWR_UCB` but slightly different: :class:`SlidingWindowRestart.SWR_UCB` uses a window of size :math:`T_0=100` to keep in memory the last 100 *draws* of *each* arm, and restart the index if the small history mean is too far away from the whole mean, while this :class:`SWUCB` uses a fixed-size window of size :math:`\tau=1000` to keep in memory the last 1000 *steps*.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "PJ"
__version__ = "0.9"

from math import log, sqrt
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy
from statsmodels.tsa.arima.model import ARIMA

#: Size of the sliding window.
TAU = 20

#: Default value for the constant :math:`\alpha`.
ALPHA = 1.0


# --- Manually written

class Arima_MaxBandit(IndexPolicy):
    r""" An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.
    """

    def __init__(self, nbArms,
                 tau=TAU, alpha=ALPHA,
                 *args, **kwargs):
        super(MaxBandit, self).__init__(nbArms, *args, **kwargs)
        # New parameters
        assert 1 <= tau, "Error: parameter 'tau' for class SWUCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        self.tau = int(tau)  #: Size :math:`\tau` of the sliding window.
        assert alpha > 0, "Error: parameter 'alpha' for class SWUCB has to be > 0, but was {}.".format(alpha)  # DEBUG
        self.alpha = alpha  #: Constant :math:`\alpha` in the square-root in the computation for the index.
        # Internal memory
        self.last_rewards = [] ## np.zeros((nbArms,2 * tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        for _ in range(nbArms):
            self.last_rewards.append([])
        ## self.last_choices = np.full((tau, -1)  #: Keep in memory the times where each arm was last seen.
        self.speed_record = np.zeros((nbArms))

    def __str__(self):
        return f"MaxBandit with tau={self.tau}"

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).
        """

        reward = (reward - self.lower) / self.amplitude
        if len(self.last_rewards[arm]) == 2 * self.tau:
            del self.last_rewards[arm][0]
        self.last_rewards[arm].append(reward)
        ## self.t += 1

    def computeIndex(self, arm):
        r""" Compute the current index, at time :math:`t` and after :math:`N_{k,\tau}(t)` pulls of arm :math:`k`:
        .. math::

            I_k(t) &= \frac{X_{k,\tau}(t)}{N_{k,\tau}(t)} + c_{k,\tau}(t),\\
            \text{where}\;\; c_{k,\tau}(t) &:= \sqrt{\alpha \frac{\log(\min(t,\tau))}{N_{k,\tau}(t)}},\\
            \text{and}\;\; X_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} X_k(s) \mathbb{1}(A(t) = k),\\
            \text{and}\;\; N_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} \mathbb{1}(A(t) = k).
        """
        ## last_pulls_of_this_arm = np.count_nonzero(self.last_choices == arm)
        last_pulls_of_this_arm = len(self.last_rewards[arm])
        if last_pulls_of_this_arm < 2 * self.tau:
            return float('+inf')
        else:
            max_value = max(self.last_rewards[arm])
            speed = max(self.last_rewards[arm][self.tau:]) - max(self.last_rewards[arm][:self.tau])
            deviation = np.std(self.last_rewards[arm])
            return max_value
# --- Horizon dependent version


