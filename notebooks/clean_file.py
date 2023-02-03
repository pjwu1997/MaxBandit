# %%
import os
import shutil

path ='./'

for i in range(200):
    shutil.rmtree(f'{path}{i}')
# %%
problems = [f'F{i}' for i in range(7,26)]
total_n_arms = [100, 80, 50, 30, 20, 10]
problem = 'F14'
dimension = 10
# n_arms = 50
total_trials_per_arm = [50, 40, 30, 20]

for problem in problems:
    for n_arms in total_n_arms:
        for trials_per_arm in total_trials_per_arm:
            for k in range(100):
                path = f'./saved_problems/{problem}/{dimension}/{k}.pickle'
                try:
                    os.remove(path)
                except:
                    pass
# %%
