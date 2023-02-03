# -*- coding: utf-8 -*-
""" Optimization based arm.

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


class Op(Arm):
    """ Bernoulli distributed arm."""

    def __init__(self, problem, sampler, budget, batch_size=10):
        """New arm."""
        self.problem = problem  #: Problem for the arm
        ## self.mean = probability  #: Mean for this Bernoulli arm
        self.sampler = sampler
        self.budget = budget
        self.batch_size = batch_size
        self.current_center = self.starting_point = np.random.random_sample(self.problem.dimension)
        ## first, just set the mean as the value at starting point
        self.record = [] ## store the optimization record
        self.history = []
        self.pre_run()
        self.mean = self.record[-1]['max_value']
        self.max = self.record[-1]['max_value']
        print(self.max)
        self.num_draws = 0

    # --- Random samples
    def pre_run(self):
        current_max = None
        current_max_point = None
        current_mean = None
        num_batch = self.budget // self.batch_size
        for i in range(num_batch):
            evaluate_points = self.sampler.get_multiple_sample_points(self.batch_size) ## get sampled points
            evaluate_values = self.problem.sample(evaluate_points)
            self.history += evaluate_values
            if current_max is None:
                current_max = max(self.history)
                current_max_point = evaluate_points[evaluate_values.index(current_max)]
            else:
                if max(evaluate_values) > current_max:
                    current_max = max(evaluate_values)
                    current_max_point = evaluate_points[evaluate_values.index(current_max)]
            self.sampler.update_center(current_max_point)
            self.record.append({"sampler_center": self.sampler.center,
                                "max_value": current_max,
                                "max_point": current_max_point,
                                "mean": sum(self.history) / len(self.history)
                                })

    def draw(self, t=1):
        """ Draw one random sample."""
        self.num_draws += 1
        print(self.num_draws)
        return np.array(self.history[self.num_draws - 1])
        # return np.asarray(binomial(1, self.probability), dtype=float)

    def draw_nparray(self, num_draws):
        """ Draw a numpy array of random samples, of a certain shape."""
        return_value = np.array(self.history[self.num_draws: self.num_draws + num_draws])
        self.num_draws += num_draws
        return return_value

    # def set_mean_param(self, probability):
    #     self.probability = self.mean = probability

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return self.problem.min_value, self.problem.max_value - self.problem.min_value

    def __str__(self):
        return "Op"

    def __repr__(self):
        return f"{self.problem.name}"

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
__all__ = ["Op"]


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)
