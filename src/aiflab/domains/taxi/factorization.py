""" Factorization """


from aiflab.core.dataclasses import LatentState
from aiflab.domains.taxi.specs import (
    NUM_COLS,
    NUM_DESTINATIONS,
    NUM_PASSENGER_LOCATIONS,
    NUM_ROWS,
    NUM_STATES,
    TaxiStateComponents,
)


def encode_state(components: TaxiStateComponents) -> int:
    """
    Encode factorized Taxi state into Gymnasium's flat discrete state index.
    """
    row = components.row
    col = components.col
    passenger_location = components.passenger_location
    destination = components.destination

    state = row
    state = state * NUM_COLS + col
    state = state * NUM_PASSENGER_LOCATIONS + passenger_location
    state = state * NUM_DESTINATIONS + destination
    return state


def decode_state(state_index: int) -> TaxiStateComponents:
    """
    Decode Gymnasium's flat discrete Taxi state index into structured factors.
    """
    if not (0 <= state_index < NUM_STATES):
        raise ValueError(f"state_index out of range: {state_index}")

    destination = state_index % NUM_DESTINATIONS
    state_index //= NUM_DESTINATIONS

    passenger_location = state_index % NUM_PASSENGER_LOCATIONS
    state_index //= NUM_PASSENGER_LOCATIONS

    col = state_index % NUM_COLS
    state_index //= NUM_COLS

    row = state_index

    return TaxiStateComponents(
        row=row,
        col=col,
        passenger_location=passenger_location,
        destination=destination,
    )


def latent_state_from_index(state_index: int) -> LatentState:
    components = decode_state(state_index)
    return LatentState(
        index=state_index,
        factors=(
            components.row,
            components.col,
            components.passenger_location,
            components.destination,
        ),
        metadata={
            "row": components.row,
            "col": components.col,
            "passenger_location": components.passenger_location,
            "destination": components.destination,
        },
    )