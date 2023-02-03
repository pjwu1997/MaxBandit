# -*- coding: utf-8 -*-
""" Optimization based arm. (CMAES)

Example of creating an arm:

>>> import random; import numpy as np
>>> random.seed(0); np.random.seed(0)
>>> B03 = Bernoulli(0.3)
>>> B03
B(0.3)
>>> B03.mean
0.3

Examples of sampling from an arm:

>>> B03.draw()
0
>>> B03.draw_nparray(20)
array([1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
       1., 1., 1.])
"""
from __future__ import division, print_function  # Python 2 compatibility

import numpy as np
from numpy.random import binomial

# Local imports
try:
    from .Arm import Arm
    from .kullback import klBern
except ImportError:
    from Arm import Arm
    from kullback import klBern

import pickle
import os

# %%
from .Generate_cec2005 import create_problem
import matplotlib.pyplot as plt

def generate_Op_cmaes_problem(name, dimension, n_arms, budget, load=False, save=False):
    result = []
    for ind in range(n_arms):
        result.append(Op_cmaes(name, dimension, ind, budget, load, save))
    ## Normalization
    maximum = -np.inf
    minimum = np.inf
    for ind in range(n_arms):
        cur_max = max(result[ind].best)
        cur_min = min(result[ind].best)
        if cur_max > maximum:
            maximum = cur_max
        if cur_min < minimum:
            minimum = cur_min
    for ind in range(n_arms):
        result[ind].best = (result[ind].best - minimum) / (maximum - minimum)
    return result

def generate_arm_pic(name, dimension, n_arms):
    fig = plt.figure(figsize=(50,30))
    for arm_id in range(n_arms):
        arm = pickle.load(open(f'./saved_problems/{name}/{dimension}/pickle/{arm_id}.pickle', 'rb'))
        plt.plot(arm, label=arm_id)
    plt.legend()
    plt.savefig(f'./saved_problems/{name}/{dimension}/{n_arms}/img.pdf')
    plt.close()


class Op_cmaes(Arm):
    """ Arm optimized by CMA-ES """
    def __init__(self, name, dimension, arm_id, budget, load=False, save=False):
        """Create a new arm, optimized by cma-es on given CEC2005 problem.
           The optimiation results will be stored in self.problem.best, self.problem.median
           name: 'F1' to 'F25'
           dimension: problem dimension, any element in {2,10,30,50} for all problem, {1,2,3,...,100} for some problem.
           arm_name: the name of the arm, should be different for every arm.
           budget: budget of the number of iterations. 
        """
        self.name = name
        self.dimension = dimension
        self.arm_id = arm_id
        self.budget = budget
        self.num_draws = 0
        filename = f'./saved_problems/{name}/{dimension}/pickle/{arm_id}.pickle'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if load:
            problem = pickle.load(open(filename, 'rb'))
            self.best = np.array(problem)
            self.mean = np.array(problem)[-1]
        else:
            problem = create_problem(name, dimension, arm_id, maxiter=budget)  #: Problem for the arm
            if save:
                pickle.dump(problem.best, open(filename, 'wb'))
            self.best = np.array(problem.best)
            self.mean = np.array(problem.best)[-1]
        # self.median = np.array(problem.median)

    # --- Random samples
    def draw(self, t=1, size=1):
        """ Draw one random sample."""
        try:
            return_value = np.array(self.best[self.num_draws: self.num_draws + size])
            self.num_draws += size
            return return_value
        except:
            return np.ones(size) * self.best[-1]
        # return np.asarray(binomial(1, self.probability), dtype=float)

    def draw_nparray(self, size):
        """ Draw a numpy array of random samples, of a certain shape."""
        try:
            num_draws = size
            return_value = np.array(self.best[self.num_draws: self.num_draws + num_draws])
            self.num_draws += num_draws
            return return_value
        except:
            return np.ones(size) * self.best[-1]

    # def set_mean_param(self, probability):
    #     self.probability = self.mean = probability

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    #TODO
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return -1000000, 2000000

    def __str__(self):
        return "CMA-ES ARM"

    def __repr__(self):
        return f"{self.name}"

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        return klBern(x, y)

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Bernoulli arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBern(mu, mumax)


# Only export and expose the class defined here
__all__ = ["cma-es"]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
