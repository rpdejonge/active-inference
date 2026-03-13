from __future__ import annotations

from aiflab.domains.taxi.observations import normalize_action_mask, observation_from_gym


class TestTaxiObservations:
    def test_normalize_action_mask_returns_tuple(self) -> None:
        mask = normalize_action_mask([1, 0, 1, 1, 0, 0])
        assert mask == (1, 0, 1, 1, 0, 0)

    def test_normalize_action_mask_accepts_none(self) -> None:
        assert normalize_action_mask(None) is None

    def test_observation_from_gym_exposes_flat_and_factorized_structure(self) -> None:
        info = {"prob": 1.0, "action_mask": [1, 1, 0, 1, 0, 0]}
        obs = observation_from_gym(386, info)

        assert obs.index == 386
        assert obs.values is not None
        assert len(obs.values) == 4
        assert obs.action_mask == (1, 1, 0, 1, 0, 0)
        assert obs.metadata["prob"] == 1.0
        assert "row" in obs.metadata
        assert "col" in obs.metadata
        assert "passenger_location" in obs.metadata
        assert "destination" in obs.metadata
        assert "action_names" in obs.metadata