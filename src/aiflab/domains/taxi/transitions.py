"""
Transitions
"""


from aiflab.domains.taxi.specs import (
    ACTION_INDEX_TO_NAME,
    ACTION_PICKUP,
    ACTION_DROPOFF,
    TaxiStepSemantics,
)

STEP_REWARD: float = -1.0
ILLEGAL_PICKUP_DROPOFF_REWARD: float = -10.0
SUCCESSFUL_DROPOFF_REWARD: float = 20.0


def action_name(action: int) -> str:
    try:
        return ACTION_INDEX_TO_NAME[action]
    except KeyError as exc:
        raise ValueError(f"unknown action index: {action}") from exc


def classify_step(
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> TaxiStepSemantics:
    return TaxiStepSemantics(
        action=action,
        action_name=action_name(action),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )


def is_illegal_special_action(action: int, reward: float) -> bool:
    """
    In Taxi-v3, illegal pickup/dropoff yields reward -10.
    """
    return (
        action in (ACTION_PICKUP, ACTION_DROPOFF)
        and reward == ILLEGAL_PICKUP_DROPOFF_REWARD
    )


def is_successful_dropoff(terminated: bool, reward: float) -> bool:
    """
    In Taxi-v3, successful completion terminates the episode with reward 20.
    """
    return terminated and reward == SUCCESSFUL_DROPOFF_REWARD
