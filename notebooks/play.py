# %%
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib as mpl
try:
    import SMPyBandits
except ImportError:
    !pip3 install SMPyBandits

# Local imports
from SMPyBandits.Environment import Evaluator, tqdm

# Import arms
from SMPyBandits.Arms import Op
from SMPyBandits.Arms.cma import Op_cmaes

# Import algorithms
from SMPyBandits.Policies import *

from multiprocessing import cpu_count

from SMPyBandits.Arms.Sampler import uniform_sampler
from SMPyBandits.Arms.Problem import Problem

import os, os.path

import dill
# %%
FIGSIZE = (19.80, 10.80)
DPI = 160
mpl.rcParams['figure.figsize'] = FIGSIZE
mpl.rcParams['figure.dpi'] = DPI

CPU_COUNT = cpu_count()
N_JOBS = CPU_COUNT if CPU_COUNT <= 4 else CPU_COUNT - 4

print("Using {} jobs in parallel...".format(N_JOBS))

HORIZON = 3000
REPETITIONS = 3

print("Using T = {}, and N = {} repetitions".format(HORIZON, REPETITIONS))

ENVIRONMENTS = []

ENVIRONMENT_0 = {   # A simple piece-wise stationary problem
    Op_cmaes('F1', 10, 'Arm1', 4000),
    Op_cmaes('F1', 10, 'Arm2', 4000),
    Op_cmaes('F1', 10, 'Arm3', 4000),
    Op_cmaes('F1', 10, 'Arm4', 4000),
    Op_cmaes('F1', 10, 'Arm5', 4000)
}

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
}

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

def printAll(evaluation, envId):
    print("\nGiving the vector of final regrets ...")
    evaluation.printLastRegrets(envId)
    print("\nGiving the final ranks ...")
    evaluation.printFinalRanking(envId)
    print("\nGiving the mean and std running times ...")
    evaluation.printRunningTimes(envId)
    print("\nGiving the mean and std memory consumption ...")
    evaluation.printMemoryConsumption(envId)

envId = 0
env = evaluation.envs[envId]
# Show the problem
# evaluation.plotHistoryOfMeans(envId)
dir(env)
# import pickle
# with open('a.pickle', 'wb') as f:
#     pickle.dump('abc', f)
# type(env.arms[0])
dill.pickles(env)

# Evaluate just that env
evaluation.startOneEnv(envId, env)

def plotAll(evaluation, envId, mainfig=None):
    savefig = mainfig
    if savefig is not None: savefig = "{}__LastRegrets__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting a boxplot of the final regrets ...")
    evaluation.plotLastRegrets(envId, boxplot=True, savefig=savefig)

    if savefig is not None: savefig = "{}__RunningTimes__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean and std running times ...")
    evaluation.plotRunningTimes(envId, savefig=savefig)

    if savefig is not None: savefig = "{}__MemoryConsumption__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean and std memory consumption ...")
    evaluation.plotMemoryConsumption(envId, savefig=savefig)

    if savefig is not None: savefig = "{}__Regrets__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean regrets ...")
    evaluation.plotRegrets(envId, savefig=savefig)

    if savefig is not None: savefig = "{}__MeanReward__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting the mean rewards ...")
    evaluation.plotRegrets(envId, meanReward=True, savefig=savefig)

    if savefig is not None: savefig = "{}__LastRegrets__env{}-{}".format(mainfig, envId+1, len(evaluation.envs))
    print("\nPlotting an histogram of the final regrets ...")
    evaluation.plotLastRegrets(envId, subplots=True, sharex=True, sharey=False, savefig=savefig)

envId = 0
# _ = evaluation.plotHistoryOfMeans(envId, savefig="{}__HistoryOfMeans__env{}-{}".format(mainfig, envId+1, len(evaluation.envs)))
_ = plotAll(evaluation, envId, mainfig=mainfig)