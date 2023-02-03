# %%
import sys
import os, os.path
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

def plotAll(evaluation, envId, mainfig=None):
    savefig = mainfig
    
    if savefig is not None: savefig = "{}__MeanReward__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean rewards ...")
    fig = evaluation.plotRegrets(envId, meanReward=True, savefig=savefig)
    fig.savefig(f'{path}mean.pdf')

    if savefig is not None: savefig = "{}__MaxReward__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the max rewards ...")
    fig = evaluation.plotRegrets(envId, maxReward=True, savefig=savefig)
    fig.savefig(f'{path}max.pdf')

def printAll(evaluation, envId):
    print("\nGiving the vector of final regrets ...")
    evaluation.printLastRegrets(envId)
    print("\nGiving the final ranks ...")
    evaluation.printFinalRanking(envId)
    print("\nGiving the mean and std running times ...")
    evaluation.printRunningTimes(envId)
    print("\nGiving the mean and std memory consumption ...")
    evaluation.printMemoryConsumption(envId)

print("Using {} jobs in parallel...".format(N_JOBS))
problems = [f'F{i}' for i in range(17, 6, -1)]
# problems = ['F9', 'F11', 'F16', 'F14', 'F10']
total_n_arms = [200, 100, 80, 70, 60, 50, 40, 30, 20, 10]
problem = 'F7'
dimension = 10
# n_arms = 50
total_trials_per_arm = [50, 40, 30, 20, 10]
# HORIZON = n_arms * 30
# n_horizon = HORIZON
REPETITIONS = 50
for problem in problems:
    for n_arms in total_n_arms:
        for trials_per_arm in total_trials_per_arm:
            HORIZON = n_arms * trials_per_arm
            n_horizon = HORIZON
            print("Using T = {}, and N = {} repetitions".format(HORIZON, REPETITIONS))
            from SMPyBandits.Arms.Sampler import uniform_sampler
            from SMPyBandits.Arms.Problem import Problem
            ENVIRONMENTS = []
            if n_arms == total_n_arms[0] and trials_per_arm == total_trials_per_arm[0]:
                arms = generate_Op_cmaes_problem(problem, dimension, n_arms, n_horizon, load=True, save=True)
            else:
                arms = generate_Op_cmaes_problem(problem, dimension, n_arms, n_horizon, load=True, save=False)
            path = f'./saved_problems/{problem}/{dimension}/{n_arms}/{trials_per_arm}/'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            generate_arm_pic(problem, dimension, n_arms)
            ENVIRONMENT_0 = {   # A simple piece-wise stationary problem
                *arms
            }
            ENVIRONMENTS = [
                ENVIRONMENT_0,
            ]

            NB_ARMS = n_arms
            open('record.txt', 'w').close()
            POLICIES = [  # XXX Regular stochastic bandits algorithms!
                    { "archtype": UCBH, "params": { "horizon": HORIZON, "alpha":1} },
                ] + [
                    { "archtype": UCBalpha, "params": {"alpha":1} }
                ] + [ # --- # XXX experimental other version of the sliding window algorithm, knowing the horizon
                    { "archtype": SWUCBPlus, "params": {
                        "horizon": HORIZON, "alpha": alpha,
                    } }
                    for alpha in [1.0]
                ] + [
                    { "archtype": EpsilonGreedy, "params": {} }
                ] + [
                    { "archtype": MaxMedian, "params": {"budget": HORIZON} }
                ] + [
                    { "archtype": Qomax_SDA, "params": {"budget": HORIZON, "q": 0.5} }
                ] + [
                    { "archtype": MaximumBandit, "params": {"budget": HORIZON,} }
                ] + [
                    { "archtype": Uniform, "params": {} }
                ] 
            # POLICIES = [
            #         { "archtype": MaximumBandit, "params": {"budget": HORIZON,} }
            #     ]
            # POLICIES = [
            #         { "archtype": Qomax_SDA, "params": {"budget": HORIZON, "q": 0.5} }
            #     ]

            configuration = {
                # --- Duration of the experiment
                "horizon": HORIZON,
                # --- Number of repetition of the experiment (to have an average)
                "repetitions": REPETITIONS,
                # --- Parameters for the use of joblib.Parallel
                "n_jobs": N_JOBS,    # = nb of CPU cores
                "verbosity": 0,      # Max joblib verbosity
                # --- Arms
                "environment": ENVIRONMENTS,
                # --- Algorithms
                "policies": POLICIES,
                # --- Random events
                "nb_break_points": 0,
                # --- Should we plot the lower-bounds or not?
                "plot_lowerbound": False,  # XXX Default
                "path": path
            }
            # (almost) unique hash from the configuration
            hashvalue = abs(hash((tuple(configuration.keys()), tuple([(len(k) if isinstance(k, (dict, tuple, list)) else k) for k in configuration.values()]))))
            print("This configuration has a hash value = {}".format(hashvalue))

            subfolder = "SP__K{}_T{}_N{}__{}_algos".format(NB_ARMS, HORIZON, REPETITIONS, len(POLICIES))
            PLOT_DIR = "plots"
            plot_dir = os.path.join(PLOT_DIR, subfolder)

            # Create the sub folder
            if os.path.isdir(plot_dir):
                print("{} is already a directory here...".format(plot_dir))
            elif os.path.isfile(plot_dir):
                raise ValueError("[ERROR] {} is a file, cannot use it as a directory !".format(plot_dir))
            else:
                os.makedirs(plot_dir)

            print("Using sub folder = '{}' and plotting in '{}'...".format(subfolder, plot_dir))
            mainfig = os.path.join(plot_dir, "main")
            print("Using main figure name as '{}_{}'...".format(mainfig, hashvalue))

            evaluation = Evaluator(configuration)

            envId = 0
            env = evaluation.envs[envId]

            # Evaluate just that env
            evaluation.startOneEnv(envId, env)


            envId = 0
            _ = plotAll(evaluation, envId, mainfig=mainfig)
# %%
