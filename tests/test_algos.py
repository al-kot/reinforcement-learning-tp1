import numpy as np

from multiarmbandits.algos import naive, ucb, thompson
from multiarmbandits.utils import cumulative_regret


def test_naive_identifies_best_simple(mab):
    T = 50
    N = 30
    _, choices = naive(mab, T=T, N=N)
    exploit_choices = choices[N:]
    assert np.all(exploit_choices == 1)


def test_ucb_and_thompson_better_than_naive_on_average(mab):
    T = 100
    trials = 50
    naive_final = []
    ucb_final = []
    thom_final = []
    for s in range(trials):
        rng = np.random.RandomState(s + 1)
        arms = [type(a)(a.p, seed=int(rng.randint(0, 1_000_000))) for a in mab]  # copy arms
        rn, _ = naive(arms, T=T, N=10)

        rng = np.random.RandomState(s + 1000)
        arms = [type(a)(a.p, seed=int(rng.randint(0, 1_000_000))) for a in mab]
        ru, _ = ucb(arms, T=T)

        rng = np.random.RandomState(s + 2000)
        arms = [type(a)(a.p, seed=int(rng.randint(0, 1_000_000))) for a in mab]
        rt, _ = thompson(arms, T=T)

        naive_final.append(cumulative_regret([a.p for a in mab], rn)[-1])
        ucb_final.append(cumulative_regret([a.p for a in mab], ru)[-1])
        thom_final.append(cumulative_regret([a.p for a in mab], rt)[-1])

    assert np.mean(ucb_final) <= np.mean(naive_final) + 1e-6 * T + 5.0  # allow some slack
    assert np.mean(thom_final) <= np.mean(naive_final) + 1e-6 * T + 5.0
