import sys
import os, os.path
sys.path.insert(0, '../..')

from SMPyBandits.Arms.cma import Op_cmaes, generate_arm_pic, generate_Op_cmaes_problem

problems = [f'F{i}' for i in range(7,26)]
total_n_arms = [100, 80, 50, 30, 20, 10, 5]
total_trials_per_arm = [50, 40, 30, 20]
# HORIZON = n_arms * 30
# n_horizon = HORIZON
dimension = 10
REPETITIONS = 100
for problem in problems:
    for n_arms in total_n_arms:
        generate_arm_pic(name, dimension, n_arms)