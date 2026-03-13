"""
Dataclasses
"""


from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Observation:
    """
    Structured observation emitted by a Domain.

    Attributes
    ----------
    index:
        Optional flat/discrete observation index.
    values:
        Optional structured observation components.
    action_mask:
        Optional valid-action mask. Convention: 1 = valid, 0 = invalid.
    metadata:
        Arbitrary domain-specific metadata.
    """

    index: int | None = None
    values: tuple[int, ...] | None = None
    action_mask: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index is None and self.values is None:
            raise ValueError(
                "Observation requires at least one of 'index' or 'values'."
            )

        if self.action_mask is not None and any(
            v not in (0, 1) for v in self.action_mask
        ):
            raise ValueError("Observation.action_mask must contain only 0/1 values.")


@dataclass(frozen=True, slots=True)
class LatentState:
    """
    Structured latent/hidden state.

    Attributes
    ----------
    index:
        Optional flat state index.
    factors:
        Optional factorized latent state components.
    metadata:
        Arbitrary domain/model-specific metadata.
    """

    index: int | None = None
    factors: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index is None and self.factors is None:
            raise ValueError(
                "LatentState requires at least one of 'index' or 'factors'."
            )


@dataclass(frozen=True, slots=True)
class BeliefState:
    """
    Posterior/prior belief over latent states.

    Attributes
    ----------
    probabilities:
        Flat probability vector over latent states.
    factor_probabilities:
        Optional factorized beliefs for multi-factor models.
    metadata:
        Arbitrary inference-related metadata.
    """

    probabilities: tuple[float, ...]
    factor_probabilities: tuple[tuple[float, ...], ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.probabilities) == 0:
            raise ValueError("BeliefState.probabilities cannot be empty.")

        total = sum(self.probabilities)
        if total <= 0:
            raise ValueError("BeliefState probabilities must sum to a positive value.")

        if any(p < 0 for p in self.probabilities):
            raise ValueError("BeliefState probabilities must be non-negative.")


@dataclass(frozen=True, slots=True)
class Policy:
    """
    A finite-horizon policy represented as an action sequence.
    """

    actions: tuple[int, ...]
    horizon: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.actions) == 0:
            raise ValueError("Policy.actions cannot be empty.")

        if self.horizon is not None and self.horizon != len(self.actions):
            raise ValueError("Policy.horizon must equal len(actions) when provided.")


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """
    Evaluation of a candidate policy.

    Attributes
    ----------
    policy:
        The evaluated policy.
    score:
        General-purpose scalar score. Convention left open for now.
    expected_free_energy:
        Optional EFE scalar.
    epistemic_value:
        Optional epistemic component.
    pragmatic_value:
        Optional pragmatic component.
    posterior_probability:
        Optional posterior probability q(pi).
    metadata:
        Arbitrary diagnostics.
    """

    policy: Policy
    score: float
    expected_free_energy: float | None = None
    epistemic_value: float | None = None
    pragmatic_value: float | None = None
    posterior_probability: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimestepRecord:
    """
    Single environment/agent interaction record.
    """

    episode_id: int
    timestep: int
    observation: Observation
    action: int | None
    reward: float
    terminated: bool
    truncated: bool
    latent_state: LatentState | None = None
    belief_state: BeliefState | None = None
    policy_evaluations: tuple[PolicyEvaluation, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """
    Sequence-level record for a single episode.
    """

    episode_id: int
    timesteps: tuple[TimestepRecord, ...]
    total_reward: float
    success: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.timesteps) == 0:
            raise ValueError("EpisodeRecord.timesteps cannot be empty.")


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """
    Top-level experiment result bundle.
    """

    experiment_name: str
    episodes: tuple[EpisodeRecord, ...]
    metrics: Mapping[str, float]
    config: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("ExperimentResult.experiment_name cannot be empty.")
