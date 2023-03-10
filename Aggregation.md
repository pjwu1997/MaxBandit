# **Policy aggregation algorithms**

- Remark: I wrote a small research article on that topic, it will be a better introduction as a small self-contained document to explain this idea and the algorithms. Reference: [[Aggregation of Multi-Armed Bandits Learning Algorithms for Opportunistic Spectrum Access, Lilian Besson and Emilie Kaufmann and Christophe Moy, 2017]](https://hal.inria.fr/hal-01705292), presented at the [IEEE WCNC 2018](http://wcnc2018.ieee-wcnc.org/) conference.

> PDF : [BKM_IEEEWCNC_2018.pdf](https://hal.inria.fr/hal-01705292/document) | HAL notice : [BKM_IEEEWCNC_2018](https://hal.inria.fr/hal-01705292/) | BibTeX : [BKM_IEEEWCNC_2018.bib](https://hal.inria.fr/hal-01705292/bibtex) | [Source code and documentation](Aggregation.html)
> [![Published](https://img.shields.io/badge/Published%3F-accepted-green.svg)](https://hal.inria.fr/hal-01705292)  [![Maintenance](https://img.shields.io/badge/Maintained%3F-finished-green.svg)](https://bitbucket.org/lbesson/aggregation-of-multi-armed-bandits-learning-algorithms-for/commits/)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

## Idea

The basic idea of a policy aggregation algorithm is to run in parallel some online learning algorithms, denoted `$A_1,\ldots,A_N$` (`$A_i$`), and make them all vote at each step, and use some probabilistic scheme to select a decision from their votes.

Hopefully, if all the algorithms `$A_i$` are not too bad and at least one of them is efficient for the problem at hand, the aggregation algorithm will learn to mainly trust the efficient one(s) and discard the votes from the others.
An efficient aggregation algorithm should have performances similar to the best child algorithm `$A_i$`, in any problem.

The [Exp4 algorithm](http://sbubeck.com/SurveyBCB12.pdf) by [Auer et al, 2002] is the first aggregation algorithm for online bandit algorithms, and recently other algorithms include [`LearnExp`](https://smpybandits.github.io/docs/Policies.LearnExp.html) ([[Singla et al, 2017](https://arxiv.org/abs/1702.04825)]) and [`CORRAL`](https://smpybandits.github.io/docs/Policies.CORRAL.html) ([[Agarwal et al, 2017](https://arxiv.org/abs/1612.06246v2)]).

---

### Mathematical explanations
Initially, every child algorithms `$A_i$` has the same "trust" probability `$p_i$`, and at every step, the aggregated bandit first listen to the decision from all its children `$A_i$` (`$a_{i,t}$` in `$\{1,\ldots,K\}$`), and then decide which arm to select by a probabilistic vote: the probability of selecting arm `$k$` is the sum of the trust probability of the children who voted for arm `$k$`.
It could also be done the other way: the aggregated bandit could first decide which children to listen to, then trust him.

But we want to update the trust probability of all the children algorithms, not only one, when it was wised to trust them.
Mathematically, when the aggregated arm choose to pull the arm `$k$` at step `$t$`, if it yielded a positive reward `$r_{k,t}$`, then the probability of all children algorithms `$A_i$` who decided (independently) to chose `$k$` (i.e., `$a_{i,t} = k$`) are increased multiplicatively: `$p_i \leftarrow p_i * \exp(+ \beta * r_{k,t})$` where `$\beta$` is a positive *learning rate*, e.g., `$\beta = 0.1$`.

It is also possible to decrease multiplicatively the trust of all the children algorithms who did not decided to chose the arm `$k$` at every step `$t$`: if `$a_{i,t} \neq k$` then `$p_i \leftarrow p_i * \exp(- \beta * r_{k,t})$`. I did not observe any difference of behavior between these two options (implemented with the Boolean parameter `updateAllChildren`).

### Ensemble voting for MAB algorithms
This algorithm can be seen as the Multi-Armed Bandits (i.e., sequential reinforcement learning) counterpart of an *ensemble voting* technique, as used for classifiers or regression algorithm in usual supervised machine learning (see, e.g., [`sklearn.ensemble.VotingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier) in [scikit-learn](http://scikit-learn.org/)).

Another approach could be to do some sort of [grid search](http://scikit-learn.org/stable/modules/grid_search.html).

### My algorithm: [Aggregator](https://smpybandits.github.io/docs/Policies.Aggregator.html)

It is based on a modification of Exp4, and the details are given in its documentation, see [`Aggregator`](https://smpybandits.github.io/docs/Policies.Aggregator.html).

All the mathematical details can be found in my paper, [[Aggregation of Multi-Armed Bandits Learning Algorithms for Opportunistic Spectrum Access, Lilian Besson and Emilie Kaufmann and Christophe Moy, 2017]](https://hal.inria.fr/hal-01705292), presented at the [IEEE WCNC 2018](http://wcnc2018.ieee-wcnc.org/) conference.

----

## Configuration:
A simple python file, [`configuration_comparing_aggregation_algorithms.py`](https://smpybandits.github.io/docs/configuration_comparing_aggregation_algorithms.html), is used to import the [arm classes](Arms/), the [policy classes](Policies/) and define the problems and the experiments.

For example, this will compare the classical MAB algorithms [`UCB`](https://smpybandits.github.io/docs/Policies.UCB.html), [`Thompson`](https://smpybandits.github.io/docs/Policies.Thompson.html), [`BayesUCB`](https://smpybandits.github.io/docs/Policies.BayesUCB.html), [`klUCB`](https://smpybandits.github.io/docs/Policies.klUCB.html) algorithms.

```python
configuration = {
    "horizon": 10000,    # Finite horizon of the simulation
    "repetitions": 100,  # number of repetitions
    "n_jobs": -1,        # Maximum number of cores for parallelization: use ALL your CPU
    "verbosity": 5,      # Verbosity for the joblib calls
    # Environment configuration, you can set up more than one.
    "environment": [
        {
            "arm_type": Bernoulli,  # Only Bernoulli is available as far as now
            "params": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.1]
        }
    ],
    # Policies that should be simulated, and their parameters.
    "policies": [
        {"archtype": UCB, "params": {} },
        {"archtype": Thompson, "params": {} },
        {"archtype": klUCB, "params": {} },
        {"archtype": BayesUCB, "params": {} },
    ]
}
```

To add an aggregated bandit algorithm ([`Aggregator` class](https://smpybandits.github.io/docs/Policies.Aggregator.html)), you can use this piece of code, to aggregate all the algorithms defined before and dynamically add it to `configuration`:
```python
current_policies = configuration["policies"]
configuration["policies"] = current_policies +
    [{  # Add one Aggregator policy, from all the policies defined above
        "archtype": Aggregator,
        "params": {
            "learningRate": 0.05,  # Tweak this if needed
            "updateAllChildren": True,
            "children": current_policies,
        },
    }]
```

The learning rate can be tuned automatically, by using the heuristic proposed by [[Bubeck and Cesa-Bianchi](http://sbubeck.com/SurveyBCB12.pdf), Theorem 4.2], without knowledge of the horizon, a decreasing learning rate `$\eta_t = \sqrt(\frac{\log(N)}{t * K})$`.

----

## [How to run the experiments ?](How_to_run_the_code.md)

You should use the provided [`Makefile`](Makefile) file to do this simply:
```bash
# if not already installed, otherwise update with 'git pull'
git clone https://github.com/SMPyBandits/SMPyBandits/
cd SMPyBandits
make install  # install the requirements ONLY ONCE
make comparing_aggregation_algorithms   # run and log the main.py script
```

----

## Some illustrations
Here are some plots illustrating the performances of the different [policies](https://smpybandits.github.io/docs/Policies/) implemented in this project, against various problems (with [`Bernoulli`](https://smpybandits.github.io/docs/Arms.Bernoulli.html) arms only):

### On a "simple" Bernoulli problem (semi-log-y scale)
![On a "simple" Bernoulli problem (semi-log-y scale).](plots/main_semilogy____env1-4_932221613383548446.png)

Aggregator is the most efficient, and very similar to Exp4 here.

### On a "harder" Bernoulli problem
![On a "harder" Bernoulli problem, they all have similar performances, except LearnExp.](plots/main____env2-4_932221613383548446.png)

They all have similar performances, except LearnExp, which performs badly.
We can check that the problem is indeed harder as the lower-bound (in black) is much larger.

### On an "easy" Gaussian problem
![On an "easy" Gaussian problem, only Aggregator shows reasonable performances, thanks to BayesUCB and Thompson sampling.](plots/main____env3-4_932221613383548446.png)

Only Aggregator shows reasonable performances, thanks to BayesUCB and Thompson sampling.
CORRAL and LearnExp clearly appears sub-efficient.

### On a harder problem, mixing Bernoulli, Gaussian, Exponential arms
![On a harder problem, mixing Bernoulli, Gaussian, Exponential arms, with 3 arms of each types with the *same mean*.](plots/main_semilogy____env4-4_932221613383548446.png)

This problem is much harder as it has 3 arms of each types with the *same mean*.

![The semi-log-x scale clearly shows the logarithmic growth of the regret for the best algorithms and our proposal Aggregator, even in a hard "mixed" problem.](plots/main_semilogx____env4-4_932221613383548446.png)

The semi-log-x scale clearly shows the logarithmic growth of the regret for the best algorithms and our proposal Aggregator, even in a hard "mixed" problem.

> These illustrations come from my article, [[Aggregation of Multi-Armed Bandits Learning Algorithms for Opportunistic Spectrum Access, Lilian Besson and Emilie Kaufmann and Christophe Moy, 2017]](https://hal.inria.fr/hal-01705292), presented at the [IEEE WCNC 2018](http://wcnc2018.ieee-wcnc.org/) conference.

----

### :scroll: License ? [![GitHub license](https://img.shields.io/github/license/SMPyBandits/SMPyBandits.svg)](https://github.com/SMPyBandits/SMPyBandits/blob/master/LICENSE)
[MIT Licensed](https://lbesson.mit-license.org/) (file [LICENSE](LICENSE)).

?? 2016-2018 [Lilian Besson](https://GitHub.com/Naereen).

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/SMPyBandits/SMPyBandits/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/SMPyBandits/SMPyBandits/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![Analytics](https://ga-beacon.appspot.com/UA-38514290-17/github.com/SMPyBandits/SMPyBandits/README.md?pixel)](https://GitHub.com/SMPyBandits/SMPyBandits/)
![![PyPI version](https://img.shields.io/pypi/v/smpybandits.svg)](https://pypi.org/project/SMPyBandits)
![![PyPI implementation](https://img.shields.io/pypi/implementation/smpybandits.svg)](https://pypi.org/project/SMPyBandits)
[![![PyPI pyversions](https://img.shields.io/pypi/pyversions/smpybandits.svg?logo=python)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI download](https://img.shields.io/pypi/dm/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![![PyPI status](https://img.shields.io/pypi/status/smpybandits.svg)](https://pypi.org/project/SMPyBandits)](https://pypi.org/project/SMPyBandits)
[![Documentation Status](https://readthedocs.org/projects/smpybandits/badge/?version=latest)](https://SMPyBandits.ReadTheDocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/SMPyBandits/SMPyBandits.svg?branch=master)](https://travis-ci.org/SMPyBandits/SMPyBandits)
[![Stars of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/stars/SMPyBandits/SMPyBandits)](https://GitHub.com/SMPyBandits/SMPyBandits/stargazers)
[![Releases of https://github.com/SMPyBandits/SMPyBandits/](https://badgen.net/github/release/SMPyBandits/SMPyBandits)](https://github.com/SMPyBandits/SMPyBandits/releases)