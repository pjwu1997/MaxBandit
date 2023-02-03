## Defining a nD problem surface.

import numpy as np
# multimodal test function
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import sin
from numpy import e
from numpy import pi
from numpy import absolute

def simple_1D_problem(x):
    """ 1 / (1 + x^2)."""
    return 1 / (1 + x ** 2)

def multimodal_2D_problem(x,y):
    if abs(x) > 10 or abs(y) > 10:
        return 0
    return -(-absolute(sin(x) * cos(y) * exp(absolute(1 - (sqrt(x**2 + y**2)/pi))))) / 20


class Problem(object):
    def __init__(self, name):
        if name == "simple":
            ## return a simple 1D problem surface
            self.dimension = 1  
            self.function = simple_1D_problem
            self.min = 0
            self.max = 1
            self.op_point = 0
            self.name = 'simple1D'
        elif name == 'multi2D':
            self.dimension = 1  
            self.function = multimodal_2D_problem
            self.min = 0
            self.max = 1
            self.op_point = 0
            self.name = 'multi2D'
        else:
            pass ## Need to add cec2005 functions
    def sample(self, points):
        values = []
        for point in points:
            print(point)
            values.append(self.function(*point))
        return values
    



