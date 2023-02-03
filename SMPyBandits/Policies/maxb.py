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

#: Size of the sliding window. (DEFAULT = 2)
TAU = 2

#: Default value for the constant :math:`\alpha`.
ALPHA = 1.0


def mahalanobis_distance(model_1, model_2):
    """
    Return the mahalanobis distance of two normal distributions.
    """
    mean_1 = model_1.mean
    mean_2 = model_2.mean
    std_1 = model_1.std
    std_2 = model_2.std
    distance = 1 / 8 * ((mean_1 - mean_2) ** 2) * (2 / (std_1 + std_2)) + 2 * np.log((std_1 + std_2) / (2 * np.sqrt(std_1 * std_2)))
    return distance

class model:
    def __init__(self, budget, window_length=2):
        self.mean = 0
        self.std = 1
        self.distribution = 'normal'
        self.window_length = window_length
        self.record = []
        self.sliding_record = []
        self.std_numer = 1
        self.std_denom = 1
        self.t = 0
        self.budget = budget
        self.time_decay = True
    
    def prediction(self, predict_length, alpha=1):
        '''
        Create a max distribution based on model and sample from it.
        '''
        # print(self.mean + self.std * (np.sqrt((self.window_length - 1))))
        if self.time_decay:
            # print(predict_length)
            # print(self.budget)
            # pre_value = np.exp(-alpha * (self.budget - predict_length) / (self.budget))
            # print(pre_value)
            
            value = self.mean + np.exp(-alpha * (self.budget - predict_length) / (self.budget)) * self.std * np.sqrt(predict_length - 1)
        else:
            value = self.mean + self.std * np.sqrt(predict_length - 1)
        try:
            if len(value) > 0:
                return value[0]
        except:
            return value
        
    def getReward(self, value):
        """
        update mean/variance according to input value.
        """
        self.record.append(value)
        self.mean = max(self.record[-self.window_length:])
        if len(self.record) > self.window_length:
            std = np.std(self.record[-self.window_length:])
            self.std_numer = self.std_numer * 0.9 + std
            self.std_denom = self.std_denom * 0.9 + 1
            self.std = self.std_numer / self.std_denom
        else:
            self.std = np.std(self.record[-self.window_length:])

class MaximumBandit(IndexPolicy):
    """ An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.
    """

    def __init__(self, nbArms, budget,
                 *args, **kwargs):
        super(MaximumBandit, self).__init__(nbArms, *args, **kwargs)
        # New parameters
        # assert 1 <= tau, "Error: parameter 'tau' for class SWUCB has to be >= 1, but was {}.".format(tau)  # DEBUG
        # self.tau = int(tau)  #: Size :math:`\tau` of the sliding window.
        # assert alpha > 0, "Error: parameter 'alpha' for class SWUCB has to be > 0, but was {}.".format(alpha)  # DEBUG
        # self.alpha = alpha  #: Constant :math:`\alpha` in the square-root in the computation for the index.
        # Internal memory
        # self.last_rewards = [] ## np.zeros((nbArms,2 * tau))  #: Keep in memory all the rewards obtained in the last :math:`\tau` steps.
        self.models = []
        self.t = 0
        self.budget = budget
        self.elimination_list = []
        self.leader = None
        self.do_elimination = False
        for _ in range(nbArms):
            print(nbArms)
            self.models.append(model(budget))

    def __str__(self):
        return f"MaxBandit"

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).
        """
        # reward = (reward - self.lower) / self.amplitude
        self.t += 1
        self.models[arm].getReward(reward)
        if self.do_elimination:
            if self.t > self.budget / 2:
                if self.leader is None:
                    self.leader = self.calculate_leader()
                    self.elimination()
                cur_leader = self.calculate_leader()
                if self.leader != cur_leader:
                    self.leader = cur_leader
                    self.elimination_list = []
                    self.elimination()
        # if len(self.elimination_list) > 5:
        #     print(len(self.elimination_list))
        #     print(f"'leader':{self.leader}")

    def softmax(self, array):
        """ Turn an ordered array -> softmax'd array.
        """
        array = array - array.max()
        transformation = np.exp(array)
        total_sum = sum([np.exp(x) for x in transformation])
        return transformation / total_sum


    def roulette_selection(self):
        # print(self.index)
        for ind, val in enumerate(self.index):
            if val == np.inf:
                return ind
        filtered_index = np.delete(np.array(list(range(len(self.index)))), self.elimination_list)
        # print(f'filtered index: {filtered_index}')
        filtered_value = self.index[filtered_index]
        # print(f'filtered value: {filtered_value}')
        order = np.argsort(filtered_value)
        # print(f'order: {order}')
        # print(f'index: {self.index}')
        minimum = self.index[filtered_index[order[0]]]
        maximum = self.index[filtered_index[order[-1]]]
        # order = np.argsort(self.index)
        # minimum = self.index[order[0]]
        # maximum = self.index[order[-1]]

        # ordered_value = (copy.deepcopy(self.index)[order] - minimum) / (maximum - minimum)
        # print(self.index)
        # ordered_value = self.softmax(copy.deepcopy(self.index)[order])
        ordered_value = self.softmax(copy.deepcopy(filtered_value)[order])
        random_value = np.random.rand()
        # with open('record.txt', 'a') as f:
        #     for ind, value in enumerate(ordered_value):
        #         f.write(f'{self.index} \n')
        #         # f.write(f'arm {order[ind]}: {value}\n')
        #     # f.write('\n')
        for ind, value in enumerate(ordered_value):
            if random_value < value:
                return filtered_index[order[ind]]
        return filtered_index[order[-1]]

    def elimination(self):
        order = np.argsort(self.index)
        best_model = self.models[order[-1]]
        second_best_model= self.models[order[-2]]
        distance = mahalanobis_distance(best_model, second_best_model)
        for model_ind in order[:-2]:
            model = self.models[model_ind]
            distance_2 = mahalanobis_distance(best_model, model)
            if distance_2 / distance < 1.5:
                if model_ind not in self.elimination_list:
                    self.elimination_list.append(model_ind)

    def calculate_leader(self):
        order = np.argsort(self.index)
        return order[-1]
    
    def computeIndex(self, arm):
        r""" Compute the current index, at time :math:`t` and after :math:`N_{k,\tau}(t)` pulls of arm :math:`k`:
        .. math::

            I_k(t) &= \frac{X_{k,\tau}(t)}{N_{k,\tau}(t)} + c_{k,\tau}(t),\\
            \text{where}\;\; c_{k,\tau}(t) &:= \sqrt{\alpha \frac{\log(\min(t,\tau))}{N_{k,\tau}(t)}},\\
            \text{and}\;\; X_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} X_k(s) \mathbb{1}(A(t) = k),\\
            \text{and}\;\; N_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} \mathbb{1}(A(t) = k).
        """
        ## last_pulls_of_this_arm = np.count_nonzero(self.last_choices == arm)
        last_pulls_of_this_arm = len(self.models[arm].record)
        if last_pulls_of_this_arm < 2:
            return np.inf
        else:
            value = self.models[arm].prediction(predict_length=self.budget - self.t)
            # print(value)
            return value
            # max_value = max(self.last_rewards[arm])
            # speed = max(self.last_rewards[arm][self.tau:]) - max(self.last_rewards[arm][:self.tau])
            # deviation = np.std(self.last_rewards[arm])
            # return max_value
    def choice(self):
        """Roulette selection strategy.
        """
        self.computeAllIndex()
        # if self.t > self.budget / 2:
        #     try:
        #         return np.random.choice(np.nonzero(self.index == np.max(self.index))[0])
        #     except ValueError:
        #         # print("Warning: unknown error in IndexPolicy.choice(): the indexes were {} but couldn't be used to select an arm.".format(self.index))
        #         return np.random.randint(self.nbArms)
        return self.roulette_selection()
        # return 1

# --- Horizon dependent version


