# **Multi-players simulation environment**

> **For more details**, refer to [this article](https://hal.inria.fr/hal-01629733).
>  Reference: [[Multi-Player Bandits Revisited, Lilian Besson and Emilie Kaufmann, 2017]](https://hal.inria.fr/hal-01629733), presented at the [Internation Conference on Algorithmic Learning Theorey 2018](http://www.cs.cornell.edu/conferences/alt2018/index.html#accepted).

> PDF : [BK__ALT_2018.pdf](https://hal.inria.fr/hal-01629733/document) | HAL notice : [BK__ALT_2018](https://hal.inria.fr/hal-01629733/) | BibTeX : [BK__ALT_2018.bib](https://hal.inria.fr/hal-01629733/bibtex) | [Source code and documentation](MultiPlayers.html)
> [![Published](https://img.shields.io/badge/Published%3F-accepted-green.svg)](http://www.cs.cornell.edu/conferences/alt2018/index.html#accepted)  [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://bitbucket.org/lbesson/multi-player-bandits-revisited/commits/)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://bitbucket.org/lbesson/ama)

There is another point of view: instead of comparing different single-player policies on the same problem, we can make them play *against each other*, in a multi-player setting.

The basic difference is about **collisions** : at each time `$t$`, if two or more user chose to sense the same channel, there is a *collision*. Collisions can be handled in different way from the base station point of view, and from each player point of view.

### Collision models
For example, I implemented these different collision models, in [`CollisionModels.py`](https://smpybandits.github.io/docs/Environment.CollisionModels.html):

- `noCollision` is a limited model *where* all players can sample an arm with collision. It corresponds to the single-player simulation: each player is a policy, compared without collision. This is for testing only, not so interesting.
- `onlyUniqUserGetsReward` is a simple collision model where only the players alone on one arm sample it and receive the reward. This is the default collision model in the literature, for instance cf. [[Shamir et al., 2015]](https://arxiv.org/abs/0910.2065v3) collision model 1 or cf [[Liu & Zhao, 2009]](https://arxiv.org/abs/0910.2065v3). [Our article](https://hal.inria.fr/hal-01629733) also focusses on this model.
- `rewardIsSharedUniformly` is similar: the players alone on one arm sample it and receive the reward, and in case of more than one player on one arm, only one player (uniform choice, chosen by the base station) can sample it and receive the reward.
- `closerUserGetsReward` is similar but uses another approach to chose who can emit. Instead of randomly choosing the lucky player, it uses a given (or random) vector indicating the *distance* of each player to the base station (it can also indicate the quality of the communication), and when two (or more) players are colliding, only the one who is closer to the base station can transmit. It is the more physically plausible.

----

### More details on the code
Have a look to:
- [`main_multiplayers.py`](https://smpybandits.github.io/docs/main_multiplayers.html) and [`configuration_multiplayers.py`](https://smpybandits.github.io/docs/configuration_multiplayers.html) to run and configure the simulation,
- the [`EvaluatorMultiPlayers`](https://smpybandits.github.io/docs/Environment.EvaluatorMultiPlayers.html) class that performs the simulation,
- the [`ResultMultiPlayers`](https://smpybandits.github.io/docs/Environment.ResultMultiPlayers.html) class to store the results,
- and some naive policies are implemented in the [`PoliciesMultiPlayers/`](https://smpybandits.github.io/docs/PoliciesMultiPlayers/) folder. As far as now, there is the [`Selfish`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.Selfish.html), [`CentralizedFixed`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.CentralizedFixed.html), [`CentralizedCycling`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.CentralizedCycling.html), [`OracleNotFair`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.OracleNotFair.html), [`OracleFair`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.OracleFair.html) multi-players policy.

### Policies designed to be used in the multi-players setting
- The first one I implemented is the ["Musical Chair"](https://arxiv.org/abs/1512.02866) policy, from [[Shamir et al., 2015]](https://arxiv.org/abs/0910.2065v3), in [`MusicalChair`](https://smpybandits.github.io/docs/Policies.MusicalChair.html).
- Then I implemented the ["MEGA"](https://arxiv.org/abs/1404.5421) policy from [[Avner & Mannor, 2014]](https://arxiv.org/abs/1404.5421), in [`MEGA`](https://smpybandits.github.io/docs/Policies.MEGA.html). But it has too much parameter, the question is how to chose them.
- The [`rhoRand`](https://smpybandits.github.io/docs/PoliciesMultiplayers.rhoRand.html) and variants are from [[Distributed Algorithms for Learning..., Anandkumar et al., 2010](http://ieeexplore.ieee.org/document/5462144/).
- Our algorithms introduced in [[Multi-Player Bandits Revisited, Lilian Besson and Emilie Kaufmann, 2017]](https://hal.inria.fr/hal-01629733) are in [`RandTopM`](https://smpybandits.github.io/docs/PoliciesMultiplayers.RandTopM.html): `RandTopM` and `MCTopM`.
- We also studied deeply the [`Selfish`](https://smpybandits.github.io/docs/PoliciesMultiplayers.Selfish.html) policy, without being able to prove that it is as efficient as `rhoRand`, `RandTopM` and `MCTopM`.

----

### Configuration:
A simple python file, [`configuration_multiplayers.py`](https://smpybandits.github.io/docs/configuration_multiplayers.html), is used to import the [arm classes](Arms/), the [policy classes](Policies/) and define the problems and the experiments.
See the explanations given for [the simple-player case](Aggregation.md).

```python
configuration["successive_players"] = [
    CentralizedMultiplePlay(NB_PLAYERS, klUCB, nbArms).children,
    RandTopM(NB_PLAYERS, klUCB, nbArms).children,
    MCTopM(NB_PLAYERS, klUCB, nbArms).children,
    Selfish(NB_PLAYERS, klUCB, nbArms).children,
    rhoRand(NB_PLAYERS, klUCB, nbArms).children,
]
```

- The multi-players policies are added by giving a list of their children (eg `Selfish(*args).children`), who are instances of the proxy class [`ChildPointer`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.ChildPointer.html). Each child methods is just passed back to the mother class (the multi-players policy, e.g., `Selfish`), who can then handle the calls as it wants (can be centralized or not).

----

## [How to run the experiments ?](How_to_run_the_code.md)

You should use the provided [`Makefile`](Makefile) file to do this simply:
```bash
# if not already installed, otherwise update with 'git pull'
git clone https://github.com/SMPyBandits/SMPyBandits/
cd SMPyBandits
make install            # install the requirements ONLY ONCE
make multiplayers       # run and log the main_multiplayers.py script
make moremultiplayers   # run and log the main_more_multiplayers.py script
```

----

### Some illustrations of multi-players simulations

![plots/MP__K9_M6_T5000_N500__4_algos__all_RegretCentralized____env1-1_8318947830261751207.png](plots/MP__K9_M6_T5000_N500__4_algos__all_RegretCentralized____env1-1_8318947830261751207.png)

> Figure 1 : Regret, `$M=6$` players, `$K=9$` arms, horizon `$T=5000$`, against `$500$` problems `$\mu$` uniformly sampled in `$[0,1]^K$`. rhoRand (top blue curve) is outperformed by the other algorithms (and the gain increases with `$M$`). MCTopM (bottom yellow) outperforms all the other algorithms is most cases.

![plots/MP__K9_M6_T10000_N1000__4_algos__all_RegretCentralized_loglog____env1-1_8200873569864822246.png](plots/MP__K9_M6_T10000_N1000__4_algos__all_RegretCentralized_loglog____env1-1_8200873569864822246.png)
![plots/MP__K9_M6_T10000_N1000__4_algos__all_HistogramsRegret____env1-1_8200873569864822246.png](plots/MP__K9_M6_T10000_N1000__4_algos__all_HistogramsRegret____env1-1_8200873569864822246.png)


> Figure 2 : Regret (in loglog scale), for `$M=6$` players for `$K=9$` arms, horizon `$T=5000$`, for `$1000$` repetitions on problem `$\mu=[0.1,\ldots,0.9]$`. RandTopM (yellow curve) outperforms Selfish (green), both clearly outperform rhoRand. The regret of MCTopM is logarithmic, empirically with the same slope as the lower bound. The `$x$` axis on the regret histograms have different scale for each algorithm.


[plots/MP__K9_M3_T123456_N100__8_algos__all_RegretCentralized_semilogy____env1-1_7803645526012310577.png](plots/MP__K9_M3_T123456_N100__8_algos__all_RegretCentralized_semilogy____env1-1_7803645526012310577.png)

> Figure 3 : Regret (in logy scale) for `$M=3$` players for `$K=9$` arms, horizon `$T=123456$`, for `$100$` repetitions on problem `$\mu=[0.1,\ldots,0.9]$`. With the parameters from their respective article, MEGA and MusicalChair fail completely, even with knowing the horizon for MusicalChair.

> These illustrations come from my article, [[Multi-Player Bandits Revisited, Lilian Besson and Emilie Kaufmann, 2017]](https://hal.inria.fr/hal-01629733), presented at the [Internation Conference on Algorithmic Learning Theorey 2018](http://www.cs.cornell.edu/conferences/alt2018/index.html#accepted).

----

### Fairness vs. unfairness
For a multi-player policy, being fair means that on *every* simulation with `$M$` players, each player access any of the `$M$` best arms (about) the same amount of time.
It is important to highlight that it has to be verified on each run of the MP policy, having this property in average is NOT enough.

- For instance, the oracle policy [`OracleNotFair`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.OracleNotFair.html) affects each of the `$M$` players to one of the `$M$` best arms, orthogonally, but once they are affected they always pull this arm. It's unfair because one player will be lucky and affected to the best arm, the others are unlucky. The centralized regret is optimal (null, in average), but it is not fair.
- And the other oracle policy [`OracleFair`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.OracleFair.html) affects an offset to each of the `$M$` players corresponding to one of the `$M$` best arms, orthogonally, and once they are affected they will cycle among the best `$M$` arms. It's fair because every player will pull the `$M$` best arms an equal number of time. And the centralized regret is also optimal (null, in average).

- Usually, the [`Selfish`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.Selfish.html) policy is *not* fair: as each player is selfish and tries to maximize her personal regret, there is no reason for them to share the time on the `$M$` best arms.

- Conversely, the [`MusicalChair`](https://smpybandits.github.io/docs/Policies.MusicalChair.html) policy is *not* fair either, and cannot be: when each player has attained the last step, ie. they are all choosing the same arm, orthogonally, and they are not sharing the `$M$` best arms.

- The [`MEGA`](https://smpybandits.github.io/docs/Policies.MEGA.html) policy is designed to be fair: when players collide, they all have the same chance of leaving or staying on the arm, and they all sample from the `$M$` best arms equally.

- The [`rhoRand`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.rhoRand.html) policy is not designed to be fair for every run, but it is fair in average.
- Similarly for our algorithms `RandTopM` and `MCTopM`, defined in [`RandTopM`](https://smpybandits.github.io/docs/PoliciesMultiPlayers.RandTopM.html).


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
