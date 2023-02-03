
# %%
from optproblems import cec2005
from optproblems import *
import cma
import numpy as np
import sys
import matplotlib.pyplot as plt

def create_problem(name, dimension, arm_name, low=-4.5, high=4.5, sigma=None, popsize=None, maxiter=1000):
    instance = getattr(sys.modules['optproblems.cec2005'], name)
    if name == 'F6':
        low = -100
        high = 100
    elif name == 'F8':
        low = -32
        high = 32
    elif name == 'F11':
        low = -0.5
        high = 0.5
    elif name == 'F12':
        low = -np.pi
        high = np.pi
    elif name == 'F13':
        low = -3
        high = 1
    elif name == 'F14':
        low = -100
        high = 100
    else:
        low = -5
        high = 5
    return Problem(instance, dimension, f'arms/{name}/{dimension}/{arm_name}', low, high, sigma, popsize, maxiter)


class Problem():
    def __init__(self, instance, dimension, arm_name, low=-4.5, high=4.5, sigma=None, popsize=None, maxiter=1000):
        self.dimension = dimension
        self.low = low
        self.high = high
        self.popsize = popsize
        self.maxiter = maxiter
        self.arm_name = arm_name
        if not sigma:
            self.sigma = 0.25 * (self.high - self.low) ## recommended sigma value
        else:
            self.sigma = sigma
        self.instance = instance(dimension)
        self.generate_record(self.arm_name, self.maxiter, self.popsize, self.dimension, self.low, self.high)
    def __call__(self, phenome):
        try:
            return self.instance(phenome)
        except BoundConstraintError:
            return 10000000
    def generate_record(self, file_name, maxiter, popsize, dimension, low, high):
        sample_point = np.random.uniform(low=low, high=high, size=dimension) ## create random starting point
        options = {
            'verb_filenameprefix': f'{file_name}/',
            'verbose':0,
            'tolfun': 0,
            'maxiter': maxiter,
            'tolflatfitness':10000,
            'tolfunhist':0,
            'tolx':0,
            'tolstagnation':1000,
            'bounds': [[low],[high]],
            'verbose': -1
        }
        res = cma.fmin(self, sample_point, (options["bounds"][1][0] - options["bounds"][0][0]) / 4.0 , options)
        self.best = []
        self.median = []
        with open(f'./{file_name}/fit.dat', 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                best_value = -float(line.split()[5])
                median_value = -float(line.split()[6])
                if median_value == -10000000:
                    if len(self.best) == 0:
                        pass
                    else:
                        self.best.append(self.best[-1])
                        self.median.append(self.median[-1])
                else:
                    self.best.append(best_value)
                    self.median.append(median_value)
        num_left = maxiter - len(self.best)
        self.best += [self.best[-1]] * num_left
        self.median += [self.median[-1]] * num_left

# %%
## This generates plot for a problem
def generate_plot(problem_name, dimension):
    problem = create_problem(problem_name, dimension)
    sample_point = np.random.uniform(low=-1, high=1, size=dimension)
    print(sample_point)
    res = cma.fmin(problem, sample_point, 1)
    with open('./outcmaes/fit.dat', 'r') as f:
        lines = f.readlines()[1:]
        best = []
        for line in lines:
            best.append(-float(line.split()[5]))
    plt.plot(best)

# %%
## This generates plot for a problem
def generate_Mplots(problem_name, dimension, num_points, maxiter, popsize):
    fig = plt.figure()
    problem = create_problem(problem_name, dimension)
    for point_id in range(num_points):
        sample_point = np.random.uniform(low=-0.3, high=0.3, size=dimension)
        res = cma.fmin(problem, sample_point, 0.1, {'verb_filenameprefix': f'{point_id}/','verbose':0, 'tolfun': 0, 'maxiter':maxiter+1, 'tolflatfitness':maxiter+1, 'tolfunhist':0, 'tolx':0, 'tolstagnation':maxiter+1, 'popsize':popsize})
        with open(f'./{point_id}/fit.dat', 'r') as f:
            lines = f.readlines()[1:]
            best = []
            for line in lines:
                value = -float(line.split()[5])
                if value == -10000000:
                    if len(best) == 0:
                        print(point_id)
                        pass
                    else:
                        best.append(best[-1])
                else:
                    best.append(value)
            print(len(best))
        plt.plot(best[:], label=point_id)
    plt.legend()
    plt.show()
    
# %%
if __name__ == '__main__':
    getattr(sys.modules['optproblems.cec2005'], 'F11')
    f1 = create_problem('F1', 5)
    f1([101,101])

    res = cma.fmin(f1, np.array([1,2,3,4,5]), 1)
    # %%
    f1([-39.311,58.899]) # it works
    generate_Mplots('F11', 10, 5)