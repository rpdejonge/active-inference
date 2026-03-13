import pytest

from aiflab.domains.taxi.factorization import decode_state, encode_state, latent_state_from_index
from aiflab.domains.taxi.specs import NUM_STATES, TaxiStateComponents


class TestTaxiFactorization:
    @pytest.mark.parametrize("state_index", [0, 1, 37, 123, 386, 499])
    def test_decode_then_encode_roundtrip(self, state_index: int) -> None:
        components = decode_state(state_index)
        rebuilt = encode_state(components)
        assert rebuilt == state_index

    def test_encode_then_decode_roundtrip(self) -> None:
        components = TaxiStateComponents(
            row=3,
            col=4,
            passenger_location=2,
            destination=1,
        )
        state_index = encode_state(components)
        recovered = decode_state(state_index)
        assert recovered == components

    @pytest.mark.parametrize("state_index", [-1, NUM_STATES])
    def test_decode_rejects_out_of_range(self, state_index: int) -> None:
        with pytest.raises(ValueError, match="state_index out of range"):
            decode_state(state_index)

    def test_latent_state_from_index_exposes_expected_structure(self) -> None:
        latent = latent_state_from_index(386)

        assert latent.index == 386
        assert latent.factors is not None
        assert len(latent.factors) == 4
        assert latent.metadata["row"] == latent.factors[0]
        assert latent.metadata["col"] == latent.factors[1]
        assert latent.metadata["passenger_location"] == latent.factors[2]
        assert latent.metadata["destination"] == latent.factors[3]