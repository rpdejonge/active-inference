"""
Rewards
"""

from dataclasses import dataclass

from aiflab.domains.taxi.transitions import (
    ILLEGAL_PICKUP_DROPOFF_REWARD,
    STEP_REWARD,
    SUCCESSFUL_DROPOFF_REWARD,
)


@dataclass(frozen=True, slots=True)
class TaxiRewardSpec:
    step_reward: float = STEP_REWARD
    illegal_pickup_or_dropoff_reward: float = ILLEGAL_PICKUP_DROPOFF_REWARD
    successful_dropoff_reward: float = SUCCESSFUL_DROPOFF_REWARD


DEFAULT_REWARD_SPEC = TaxiRewardSpec()
