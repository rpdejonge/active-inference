from __future__ import annotations

import pytest

from aiflab.domains.taxi.specs import (
    ACTION_NAMES,
    GRID_COLS,
    GRID_ROWS,
    LANDMARK_COORDS,
    LOCATION_NAMES,
    NUM_ACTIONS,
    NUM_STATES,
    TaxiStateComponents,
)


class TestTaxiSpecs:
    def test_global_cardinalities(self) -> None:
        assert GRID_ROWS == 5
        assert GRID_COLS == 5
        assert NUM_ACTIONS == 6
        assert NUM_STATES == 500

    def test_location_names_are_canonical(self) -> None:
        assert LOCATION_NAMES == ("R", "G", "Y", "B")
        assert set(LANDMARK_COORDS.keys()) == {"R", "G", "Y", "B"}

    def test_action_names_are_canonical(self) -> None:
        assert ACTION_NAMES == (
            "south",
            "north",
            "east",
            "west",
            "pickup",
            "dropoff",
        )

    def test_taxi_state_components_accept_valid_values(self) -> None:
        state = TaxiStateComponents(row=0, col=4, passenger_location=2, destination=1)
        assert state.row == 0
        assert state.col == 4
        assert state.passenger_location == 2
        assert state.destination == 1

    @pytest.mark.parametrize(
        ("kwargs",),
        [
            ({"row": -1, "col": 0, "passenger_location": 0, "destination": 0},),
            ({"row": 0, "col": 5, "passenger_location": 0, "destination": 0},),
            ({"row": 0, "col": 0, "passenger_location": 5, "destination": 0},),
            ({"row": 0, "col": 0, "passenger_location": 0, "destination": 4},),
        ],
    )
    def test_taxi_state_components_reject_invalid_values(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            TaxiStateComponents(**kwargs)