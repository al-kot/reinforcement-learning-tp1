# TP1 Reinforcement Learning : le problème du bandit-manchot multi-bras

## Installation

```bash
poetry install
poetry env activate
```

## Tests
```
poetry run pytest -q
```

## Simulation
```
poetry run multiarmbandits --algorithm [ucb, naive, thompson (choose one)] --t 100 --seed 42
```
