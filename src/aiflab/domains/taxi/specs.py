"""
SPECS
"""

from dataclasses import dataclass

# Grid dimensions for Gymnasium Taxi-v3.
GRID_ROWS: int = 5
GRID_COLS: int = 5

# Canonical named locations in Taxi.
# Gymnasium Taxi uses four landmarks: R, G, Y, B
LOCATION_NAMES: tuple[str, ...] = ("R", "G", "Y", "B")
LOCATION_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(LOCATION_NAMES)
}
INDEX_TO_LOCATION: dict[int, str] = {
    idx: name for idx, name in enumerate(LOCATION_NAMES)
}

# Passenger locations:
# 0..3 -> at landmark
# 4    -> in taxi
PASSENGER_IN_TAXI: int = 4
PASSENGER_LOCATION_NAMES: tuple[str, ...] = (*LOCATION_NAMES, "IN_TAXI")

# Actions in Taxi-v3.
ACTION_SOUTH: int = 0
ACTION_NORTH: int = 1
ACTION_EAST: int = 2
ACTION_WEST: int = 3
ACTION_PICKUP: int = 4
ACTION_DROPOFF: int = 5

ACTION_NAMES: tuple[str, ...] = (
    "south",
    "north",
    "east",
    "west",
    "pickup",
    "dropoff",
)

ACTION_NAME_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(ACTION_NAMES)
}
ACTION_INDEX_TO_NAME: dict[int, str] = {
    idx: name for idx, name in enumerate(ACTION_NAMES)
}

# Taxi-v3 cardinalities.
NUM_ROWS: int = 5
NUM_COLS: int = 5
NUM_PASSENGER_LOCATIONS: int = 5
NUM_DESTINATIONS: int = 4

NUM_ACTIONS: int = 6
NUM_STATES: int = 500


# Landmark coordinates in Taxi-v3.
# These are the standard Taxi coordinates.
LANDMARK_COORDS: dict[str, tuple[int, int]] = {
    "R": (0, 0),
    "G": (0, 4),
    "Y": (4, 0),
    "B": (4, 3),
}


@dataclass(frozen=True, slots=True)
class TaxiStateComponents:
    row: int
    col: int
    passenger_location: int
    destination: int

    def __post_init__(self) -> None:
        if not (0 <= self.row < NUM_ROWS):
            raise ValueError(f"row out of range: {self.row}")
        if not (0 <= self.col < NUM_COLS):
            raise ValueError(f"col out of range: {self.col}")
        if not (0 <= self.passenger_location < NUM_PASSENGER_LOCATIONS):
            raise ValueError(
                f"passenger_location out of range: {self.passenger_location}"
            )
        if not (0 <= self.destination < NUM_DESTINATIONS):
            raise ValueError(f"destination out of range: {self.destination}")


@dataclass(frozen=True, slots=True)
class TaxiStepSemantics:
    action: int
    action_name: str
    reward: float
    terminated: bool
    truncated: bool
