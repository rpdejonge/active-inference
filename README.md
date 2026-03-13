## Active Inference Lab

playing around with active inference (AIF) agents (Parr, Pezzulo & Friston 2022).

** note: work in progress - not yet operational **

## Taxi

The first implemented domain is the **Taxi environment** from Gymnasium.

Goals of the project:

- clean POMDP implementations
- explicit generative models
- interpretable inference
- reproducible experiments
- comparisons against classical RL baselines

---

## Installation

```bash
git clone <repo>

python -m venv .venv
source .venv/bin/activate

pip install -e .
