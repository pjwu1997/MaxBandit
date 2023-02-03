# %%
import sys
import os, os.path
import json
import csv
sys.path.insert(0, '..')
import SMPyBandits
# except ImportError:
#    !pip3 install SMPyBandits
import numpy as np
FIGSIZE = (19.80, 10.80)
DPI = 160
# Large figures for pretty notebooks
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = FIGSIZE
mpl.rcParams['figure.dpi'] = DPI
# Local imports
from SMPyBandits.Environment import Evaluator, tqdm
# Import arms
from SMPyBandits.Arms import Op
from SMPyBandits.Arms.cma import Op_cmaes, generate_arm_pic, generate_Op_cmaes_problem
# Import algorithms
from SMPyBandits.Policies import *

from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
N_JOBS = CPU_COUNT if CPU_COUNT <= 4 else CPU_COUNT - 4

# problems = [f'F{i}' for i in range(7,26)]
problems = ['F7', 'F8', 'F9', 'F10']
# total_n_arms = [200, 100, 80, 50, 30, 20, 10]
total_n_arms = [30, 50, 80, 100, 200]
problem = 'F14'
dimension = 10
# n_arms = 50
total_trials_per_arm = [30, 40, 50]

title_row = ['n_arms','ucb-h', 'ucb', 'swucb', 'greedy', 'MaxMedian', 'Qomax_SDA', 'MaxB', 'Uniform']
keywords = ['uch-h', 'ucb(', 'sw-ucb', 'greedy', 'median', 'sda', 'bandit', 'uniform']

for problem in problems:
    print(problem)
    path = f'./saved_problems/{problem}/{dimension}/'
    for n_arms in total_n_arms:
        ## 同樣arms數，不同平均抽取數
        file_name_1 = f'./saved_problems/{problem}/{dimension}/csv/fixed_arms/{n_arms}_mean.csv'
        file_name_2 = f'./saved_problems/{problem}/{dimension}/csv/fixed_arms/{n_arms}_std.csv'
        os.makedirs(os.path.dirname(file_name_1), exist_ok=True)
        with open(file_name_1, 'w') as csv_mean, open(file_name_2, 'w') as csv_std:
            writer_mean = csv.writer(csv_mean, delimiter=' ')
            writer_std = csv.writer(csv_std, delimiter=' ')
            writer_mean.writerow(title_row)
            writer_std.writerow(title_row)
            for trials_per_arm in total_trials_per_arm:
                json_path = f'{path}{n_arms}/{trials_per_arm}/result.json'
                f = open(json_path)
                data = json.load(f)
                mean_row = [n_arms]
                std_row = [n_arms]
                for keyword in keywords:
                    for key, value in data.items():
                        if keyword in key.lower():
                            # max_rewards = value['maxrewards']
                            total_max_rewards = np.array(value['allmaxrewards'])[-1,:]
                            mean = np.mean(total_max_rewards)
                            std = np.std(total_max_rewards)
                            mean_row.append(mean)
                            std_row.append(std)
                f.close()
                writer_mean.writerow(mean_row)
                writer_std.writerow(std_row)
                            
                            
                            
                            
                    
                    
                
            
# %%
