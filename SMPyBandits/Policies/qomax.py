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
from bisect import bisect

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
# --- Manually written

# class model(ss.rv_continuous):
#     def __init__(self, window_length=5):
#         super().__init__()
#         self.mean = 0
#         self.std = 1
#         self.distribution = 'normal'
#         self.window_length = window_length
#         self.record = []
#         self.sliding_record = []
#         self.std_numer = 1
#         self.std_denom = 1


#     def _pdf(self, x):
#         if self.distribution == 'normal':
#             return (self.window_length) * (norm.cdf(x, loc=self.mean, scale=self.std) ** (self.window_length - 1)) * norm.pdf(x, loc=self.mean, scale=self.std)
    
#     def prediction(self, window_length):
#         '''
#         Create a max distribution based on model and sample from it.
#         '''
#         self.window_length = window_length
#         print(self.std)
#         print(self.mean)
#         return self.rvs()

#     def getReward(self, value):
#         """
#         update mean/variance according to input value.
#         """
#         self.record.append(value)
#         if len(self.record) > self.window_length:
#             self.mean = max(self.record[-self.window_length:])
#             std = np.std(self.record[-self.window_length:])
#             self.std_numer = self.std_numer * 0.99 + std
#             self.std_denom = self.std_denom * 0.99 + 1
#             self.std = self.std_numer / self.std_denom

def mahalanobis_distance(model_1, model_2):
    """
    Return the mahalanobis distance of two normal distributions.
    """
    mean_1 = model_1.mean
    mean_2 = model_2.mean
    std_1 = model_1.std
    std_2 = model_2.std
    distance = 1 / 8 * ((mean_1 - mean_2) ** 2) * (2 / (std_1 + std_2)) + 2 * np.log((std_1 + std_2) / (2 * np.sqrt(std_1 * std_2)))

class model:
    def __init__(self, budget, window_length=20):
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
    
    def prediction(self, predict_length):
        '''
        Create a max distribution based on model and sample from it.
        '''
        # print(self.mean + self.std * (np.sqrt((self.window_length - 1))))
        return self.mean + self.std * np.sqrt(predict_length - 1)# Expect Upbound for extreme value

    def getReward(self, value):
        """
        update mean/variance according to input value.
        """
        self.record.append(value)
        if len(self.record) > self.window_length:
            self.mean = max(self.record[-self.window_length:])
            std = np.std(self.record[-self.window_length:])
            self.std_numer = self.std_numer * 0.7 + std
            self.std_denom = self.std_denom * 0.7 + 1
            self.std = self.std_numer / self.std_denom

class Qomax(IndexPolicy):
    """ An experimental policy, using only a sliding window (of for instance :math:`\tau=1000` *steps*, not counting draws of each arms) instead of using the full-size history.
    """

    def __init__(self, nbArms, budget,
                 *args, **kwargs):
        super(Qomax, self).__init__(nbArms, *args, **kwargs)
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
        self.maxima = dict(zip(np.arange(self.nbArms), [{} for _ in range(self.nbArms)]))
        self.n = np.zeros(self.nbArms, dtype=np.int32)
        self.nb_batch = np.zeros(self.nbArms, dtype=np.int32)
        self.qomax = np.inf * np.ones(self.nbArms)
        self.arm_rewards = [[] for _ in range(self.nbArms)]
        self.current_max = -np.inf
        self.max_arms = np.zeros(self.nbArms)

    def __str__(self):
        return f"Qomax"

    def getReward(self, arm, reward):
        """Give a reward: increase t, pulls, and update cumulated sum of rewards and update small history (sliding window) for that arm (normalized in [0, 1]).
        """
        # reward = (reward - self.lower) / self.amplitude
        self.Na[arm] += 1
        self.t += 1
        self.arm_rewards[arm].getReward(reward)

    # def softmax(self, array):
    #     """ Turn an ordered array -> softmax'd array.
    #     """
    #     transformation = np.exp(array)
    #     total_sum = sum([np.exp(x) for x in transformation])
    #     return transformation / total_sum


    # def roulette_selection(self):
    #     # print(self.index)
    #     order = np.argsort(self.index)
    #     minimum = self.index[order[0]]
    #     maximum = self.index[order[-1]]
    #     # ordered_value = (copy.deepcopy(self.index)[order] - minimum) / (maximum - minimum)
    #     ordered_value = self.softmax(copy.deepcopy(self.index)[order])
    #     random_value = np.random.rand()
    #     with open('record.txt', 'a') as f:
    #         for ind, value in enumerate(ordered_value):
    #             f.write(f'{self.index} \n')
    #             # f.write(f'arm {order[ind]}: {value}\n')
    #         # f.write('\n')
    #     for ind, value in enumerate(ordered_value):
    #         if random_value < value:
    #             return order[ind]
    #     return order[-1]

    # def computeIndex(self, arm):
    #     r""" Compute the current index, at time :math:`t` and after :math:`N_{k,\tau}(t)` pulls of arm :math:`k`:
    #     .. math::

    #         I_k(t) &= \frac{X_{k,\tau}(t)}{N_{k,\tau}(t)} + c_{k,\tau}(t),\\
    #         \text{where}\;\; c_{k,\tau}(t) &:= \sqrt{\alpha \frac{\log(\min(t,\tau))}{N_{k,\tau}(t)}},\\
    #         \text{and}\;\; X_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} X_k(s) \mathbb{1}(A(t) = k),\\
    #         \text{and}\;\; N_{k,\tau}(t) &:= \sum_{s=t-\tau+1}^{t} \mathbb{1}(A(t) = k).
    #     """
        # ## last_pulls_of_this_arm = np.count_nonzero(self.last_choices == arm)
        # last_pulls_of_this_arm = len(self.models[arm].record)
        # if last_pulls_of_this_arm < 3:
        #     return float('+inf')
        # else:
        #     value = self.models[arm].prediction(predict_length=self.budget - self.t)
        #     return value
        #     # max_value = max(self.last_rewards[arm])
        #     # speed = max(self.last_rewards[arm][self.tau:]) - max(self.last_rewards[arm][:self.tau])
        #     # deviation = np.std(self.last_rewards[arm])
        #     # return max_value
    # def choice(self):
    #     """Roulette selection strategy.
    #     """
    #     self.computeAllIndex()
    #     return self.roulette_selection()
    #     # return 1

# --- Horizon dependent version

class Qomax_ETC(Qomax):
    def __init__(self, nbArms, budget, size, q,
                 *args, **kwargs):
        super(Qomax_ETC, self).__init__(nbArms, budget, *args, **kwargs)
        self.size = size
        self.q = q
        self.arm_final = None
    def __str__(self):
        return f'Qomax_ETC with prob {self.q}'
    def f(self):
        qomax = np.zeros(self.nbArms)
        for k in range(self.nbArms):
            samples = np.array(self.arm_rewards[k]).reshape(self.size)
            M = np.max(samples, axis=1)
            qomax[k] = np.quantile(M, self.q)
        return int(rd_argmax(qomax))
    def getReward(self, arm, reward):
        self.Na[arm] += 1
        self.t += 1
        self.arm_rewards[arm].append(reward)
    def computeIndex(self, arm):
        if arm != self.arm_final:
            return 0
        else:
            return 1

    def choice(self):
        # print(int(min(self.nbArms * self.size[0] * self.size[1], self.T)))
        if self.t < int(min(self.nbArms * self.size[0] * self.size[1], self.T)):
            return self.t % self.nbArms
        if self.t >= self.nbArms * self.size[0] * self.size[1]:
            if not self.arm_final:
                self.arm_final = self.f()
            return self.arm_final

class Qomax_SDA(Qomax):
    """
    self.n -> number of batches
    self.
    """
    def __init__(self, nbArms, budget, fe=None, batch_size=None, q=0.5, *args, **kwargs):
        super(Qomax_SDA, self).__init__(nbArms, budget, *args, **kwargs)
        self.q = q
        if not fe:
            self.fe = self.default_fe
        else:
            self.fe = fe
        if not batch_size:
            self.batch_size = self.default_batch_size
        else:
            self.batch_size = batch_size
        self.chosen_arms = [-2]
        self.l_prev = -1
        self.round = 1
        self.current_update_arm = None
        self.num_new_batches = None
        self.round_done_flag = True
        self.finished = False

    def __str__(self):
        return f'Qomax_SDA with prob {self.q}'
    
    def default_fe(self, x):
        return max(5, np.log(x) ** (3/2))
    
    def default_batch_size(self, n):
        return n ** (2/3)
    
    def computeIndex(self, arm):
        return 0

    def getReward(self, arm, reward):
        """
        self.type = '1' -> 只更新該arm每個batch多抽一個
        self.type = '2' -> 該arm每個batch多抽一個，且新增self.num_new_batches個新的batch
        self.type = '3' -> 該arm新增self.num_new_batches個新的batch, 但每個batch的抽樣次數相同(= self.n[arm]).
        """
        if self.finished:
            return None
        if self.type == '1':
            for batch in range(len(reward)):
                r = reward[batch]
                self.maxima[arm][int(batch)] = {key: val for key, val in self.maxima[arm][batch].items()
                                                if val > r}
                self.maxima[arm][int(batch)][self.n[arm]] = r
        elif self.type == '2':
            # print(arm)
            # print(reward)
            # print(self.nb_batch[arm])
            # print(self.num_new_batches)
            for batch in range(self.nb_batch[arm] - self.num_new_batches):
                r = reward[batch]
                self.maxima[arm][int(batch)] = {key: val for key, val in self.maxima[arm][batch].items()
                                                if val > r}
                self.maxima[arm][int(batch)][self.n[arm]] = r
            left_rewards = np.array(reward[self.nb_batch[arm] - self.num_new_batches:]).reshape(self.num_new_batches, self.n[arm])
            for i, rewards in enumerate(left_rewards):
                rwd = [(j, rewards[j]) for j in range(rewards.shape[0])]
                batch = []
                while len(rwd) > 0:
                    batch = [rwd[-1]] + batch
                    rwd = [x for x in rwd if x[1] > rwd[-1][1]]
                self.maxima[arm][int(self.nb_batch[arm] - self.num_new_batches + i)] = dict(batch)
        elif self.type == '3':
            left_rewards = np.array(reward).reshape(self.num_new_batches, self.n[arm])
            for i, rewards in enumerate(left_rewards):
                rwd = [(j, rewards[j]) for j in range(rewards.shape[0])]
                batch = []
                while len(rwd) > 0:
                    batch = [rwd[-1]] + batch
                    rwd = [x for x in rwd if x[1] > rwd[-1][1]]
                self.maxima[arm][int(self.nb_batch[arm] - self.num_new_batches + i)] = dict(batch)
            
    def get_leader_qomax(self, n, qomax):
        m = np.amax(n)
        n_argmax = np.nonzero(n == m)[0]
        if n_argmax.shape[0] == 1:
            return n_argmax[0]
        else:
            maximomax = qomax[n_argmax].max()
            s_argmax = np.nonzero(qomax[n_argmax] == maximomax)[0]
        return n_argmax[np.random.choice(s_argmax)]

    
    def get_pulls(self):
        """
        先加在chosen arm上
        當chosen arm為空(都已新增了) 且 self.nb_batch.max() > self.nb_batch[self.l]
        需要補足leader的batch數
        return (arm, num): arm要抽num次

        同時更新self.n, self.num_batch
        """
        num = 0
        if len(self.current_running_arms) > 0:
            np.random.shuffle(self.current_running_arms)
            arm = self.current_running_arms[0]
            self.update_arm = arm
            self.n[arm] += 1 ## 該arm每個batch要多抽一個
            num += self.nb_batch[arm] ## 總共要抽nb_batch[arm]個
            self.type = '1' ## update arm, with n, no batch updated
            if self.batch_size(self.n[arm]) > self.nb_batch[arm] and not (self.num_round_arms == 1 and self.current_running_arms == [self.l]):
                num_new_batches = int(np.ceil(self.batch_size(self.n[arm])) - self.nb_batch[arm])
                self.num_new_batches = num_new_batches
                self.nb_batch[arm] += self.num_new_batches
                num += (self.num_new_batches * self.n[arm]) ## 再加這麼多個
                self.type = '2' ## update arm, with batch also updated
            self.current_running_arms.remove(arm) ## 移除該arm
            max_num = self.T - self.t
            self.t += num
            if self.t >= self.T:
                self.finished = True
                num = max_num
            return (arm, num)
        else:
            if self.nb_batch.max() > self.nb_batch[self.l]:
                ## leader應該要有一樣多的batch數
                arm = self.l
                num_new_batches = int(self.nb_batch.max() - self.nb_batch[arm])
                self.num_new_batches = num_new_batches
                self.nb_batch[self.l] += self.num_new_batches
                num += (self.num_new_batches) * self.n[arm]
                self.round_done_flag = True
                self.round += 1
                self.type = '3' ## update leader, with only batch number updated
                max_num = self.T - self.t
                self.t += num
                if self.t >= self.T:
                    num = max_num
                    self.finished = True
                return (arm, num)
            else:
                self.round += 1
                self.round_done_flag = True
                return None

    def compute_qomax_list(self, l, q):
        return np.quantile(l, q)

    def compute_qomax_dic(self, mx, q):
        """
        Computation of the QoMax using the storage trick
        """
        return np.quantile([list(mx[i].values())[0] for i in range(mx.__len__())], q)

    def qomax_duel(self, l, k, chosen_arms_prev, fe, q=None):
        if k == l:
            return k
        if self.n[k] <= fe(self.round):
            return k
        # Compute leader's QoMax (on Last Block subsample)
        l_max = np.zeros(self.nb_batch[k])
        for i in range(self.nb_batch[k]):
            last_idx = self.n[l] - self.n[k]
            idx_dic = [*self.maxima[l][i]]
            l_max[i] = self.maxima[l][i][idx_dic[bisect(idx_dic, last_idx)]]
        sub_qomax = self.compute_qomax_list(l_max, q)
        # Comparison with challenger's QoMax
        if k in chosen_arms_prev or self.qomax[k] == np.inf:
            self.qomax[k] = self.compute_qomax_dic(self.maxima[k], q)  # Update if QoMax changes
        if sub_qomax <= self.qomax[k]:
            return k

    def choice(self):
        if not self.round_done_flag:
            result = self.get_pulls()
            if not result: ## 拉完了
                pass
            else:
                return result
        self.round_done_flag = False
        if self.chosen_arms == [self.l_prev]:
            self.l = self.l_prev
        else:
            self.l = self.get_leader_qomax(self.n, self.qomax)
            self.l_prev = self.l
        self.chosen_arms_prev = [x for x in self.chosen_arms]
        self.chosen_arms = []
        for k in range(self.nbArms):
            if self.qomax_duel(self.l, k, self.chosen_arms_prev, self.fe, self.q) == k and k != self.l:
                self.chosen_arms.append(k)
        if self.n[self.l] <= self.fe(self.round):
            self.chosen_arms.append(self.l)
        if len(self.chosen_arms) == 0:
            self.chosen_arms = [self.l]
        self.current_running_arms = self.chosen_arms
        self.num_round_arms = len(self.current_running_arms)
        return self.get_pulls()
        
        
        