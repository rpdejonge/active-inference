"""
Observations
"""

from aiflab.core.dataclasses import Observation
from aiflab.domains.taxi.factorization import decode_state
from aiflab.domains.taxi.specs import ACTION_NAMES


def normalize_action_mask(
    action_mask: list[int] | tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if action_mask is None:
        return None
    return tuple(int(x) for x in action_mask)


def observation_from_gym(
    obs_index: int,
    info: dict | None = None,
) -> Observation:
    """
    Build a structured Observation from Gymnasium Taxi output.
    """
    info = info or {}
    components = decode_state(obs_index)
    action_mask = normalize_action_mask(info.get("action_mask"))

    return Observation(
        index=obs_index,
        values=(
            components.row,
            components.col,
            components.passenger_location,
            components.destination,
        ),
        action_mask=action_mask,
        metadata={
            "row": components.row,
            "col": components.col,
            "passenger_location": components.passenger_location,
            "destination": components.destination,
            "prob": info.get("prob"),
            "action_names": ACTION_NAMES,
        },
    )
