# %%
import sys
import os, os.path
import json
import csv
sys.path.insert(0, '..')
# except ImportError:
#    !pip3 install SMPyBandits
import numpy as np
FIGSIZE = (19.80, 10.80)
DPI = 160
# Large figures for pretty notebooks
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = FIGSIZE
mpl.rcParams['figure.dpi'] = DPI
# Local imports

from multiprocessing import cpu_count
import pandas as pd

CPU_COUNT = cpu_count()
N_JOBS = CPU_COUNT if CPU_COUNT <= 4 else CPU_COUNT - 4

# problems = [f'F{i}' for i in range(7,26)]
               
# total_n_arms = [10, 20, 30, 50, 80, 100]
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
        file_name = f'./saved_problems/{problem}/{dimension}/csv/fixed_arms/{n_arms}_mean.csv'
        img_name = f'./saved_problems/{problem}/{dimension}/img/fixed_arms/{n_arms}_img.pdf'
        os.makedirs(os.path.dirname(img_name), exist_ok=True)
        dataframe = pd.read_csv(file_name, sep=' ')
        # print(dataframe)
        x_data = dataframe['n_arms']
        sw_ucb = dataframe['swucb']
        ucb = dataframe['ucb']
        ucb_h = dataframe['ucb-h']
        MaxB = dataframe['MaxB']
        MaxMedian = dataframe['MaxMedian']
        Qomax_SDA = dataframe['Qomax_SDA']
        greedy = dataframe['greedy']
        Uniform = dataframe['Uniform']
        plt.plot(x_data, sw_ucb, label='SW-UCB')
        plt.plot(x_data, ucb, label='UCB')
        plt.plot(x_data, MaxB, label='MaxBandit')
        plt.plot(x_data, MaxMedian, label='MaxMedian')
        plt.plot(x_data, greedy, label='epsilon greedy')
        plt.plot(x_data, Qomax_SDA, label='Qomax_SDA')
        plt.plot(x_data, ucb-h, label='UCB-H')
        plt.plot(x_data, Uniform, label='Uniform')
        plt.legend()
        plt.title(f'Problem {problem}, with {n_arms} arms.')
        plt.xlabel('Average pulls per arm')
        plt.ylabel('Performace')
        plt.savefig(img_name)
        plt.close()
        
                            
                            
                            
                            
                    
                    
                
            
# %%
