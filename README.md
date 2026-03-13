# active-inference
playing around with active inference agents (Parr, Pezzulo & Friston 2022).

## Active Inference Lab

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
cd active-inference-lab

python -m venv .venv
source .venv/bin/activate

pip install -e .