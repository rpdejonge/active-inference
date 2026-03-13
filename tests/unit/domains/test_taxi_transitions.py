import pytest

from aiflab.domains.taxi.specs import ACTION_PICKUP, ACTION_DROPOFF
from aiflab.domains.taxi.transitions import (
    ILLEGAL_PICKUP_DROPOFF_REWARD,
    SUCCESSFUL_DROPOFF_REWARD,
    action_name,
    classify_step,
    is_illegal_special_action,
    is_successful_dropoff,
)


class TestTaxiTransitions:
    @pytest.mark.parametrize(
        ("action", "expected_name"),
        [
            (0, "south"),
            (1, "north"),
            (2, "east"),
            (3, "west"),
            (4, "pickup"),
            (5, "dropoff"),
        ],
    )
    def test_action_name(self, action: int, expected_name: str) -> None:
        assert action_name(action) == expected_name

    def test_action_name_rejects_unknown_action(self) -> None:
        with pytest.raises(ValueError, match="unknown action index"):
            action_name(999)

    def test_classify_step_returns_semantics(self) -> None:
        semantics = classify_step(
            action=2,
            reward=-1.0,
            terminated=False,
            truncated=False,
        )
        assert semantics.action == 2
        assert semantics.action_name == "east"
        assert semantics.reward == -1.0
        assert semantics.terminated is False
        assert semantics.truncated is False

    def test_detects_illegal_pickup(self) -> None:
        assert is_illegal_special_action(
            action=ACTION_PICKUP,
            reward=ILLEGAL_PICKUP_DROPOFF_REWARD,
        )

    def test_detects_illegal_dropoff(self) -> None:
        assert is_illegal_special_action(
            action=ACTION_DROPOFF,
            reward=ILLEGAL_PICKUP_DROPOFF_REWARD,
        )

    def test_non_special_actions_are_not_illegal_special_actions(self) -> None:
        assert not is_illegal_special_action(action=0, reward=ILLEGAL_PICKUP_DROPOFF_REWARD)

    def test_detects_successful_dropoff(self) -> None:
        assert is_successful_dropoff(
            terminated=True,
            reward=SUCCESSFUL_DROPOFF_REWARD,
        )

    def test_nonterminal_step_is_not_successful_dropoff(self) -> None:
        assert not is_successful_dropoff(
            terminated=False,
            reward=SUCCESSFUL_DROPOFF_REWARD,
        )