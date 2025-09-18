from __future__ import annotations

import typer
import numpy as np
import matplotlib.pyplot as plt

from .arms import ArmBernoulli
from .algos import naive, ucb, thompson
from .utils import cumulative_regret

app = typer.Typer(help="Simple CLI for multi-armed bandit experiments")


@app.command()
def simulate(
    algorithm: str = typer.Option("ucb", help="one of: naive, ucb, thompson"),
    t: int = 100,
    seed: int = 1,
    visualize: bool = True,
) -> None:
    np.random.seed(seed)
    K = 5
    means = np.random.RandomState(seed).random(K)
    mab = [ArmBernoulli(p, seed=np.random.randint(1_000_000)) for p in means]

    if algorithm == "naive":
        rewards, _ = naive(mab, T=t, N=20)
    elif algorithm == "ucb":
        rewards, _ = ucb(mab, T=t)
    elif algorithm == "thompson":
        rewards, _ = thompson(mab, T=t)
    else:
        raise typer.BadParameter("algorithm must be one of naive, ucb, thompson")

    reg = cumulative_regret(means, rewards)
    typer.echo(f"True means: {means}")
    typer.echo(f"Final cumulative regret: {reg[-1]:.4f}")

    if visualize:
        plt.plot(reg, label=f"{algorithm} regret")
        plt.xlabel("t")
        plt.ylabel("Cumulative regret")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    app()
