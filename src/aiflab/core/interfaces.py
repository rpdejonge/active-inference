from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, runtime_checkable


from aiflab.core.dataclasses import (
    BeliefState,
    EpisodeRecord,
    ExperimentResult,
    Observation,
    Policy,
    PolicyEvaluation,
    TimestepRecord,
)


@runtime_checkable
class SupportsConfig(Protocol):
    """
    Protocol for objects that can expose a serializable config.
    """

    def to_config(self) -> Mapping[str, Any]: ...


class Domain(ABC):
    """
    Abstract environment/domain boundary.

    A Domain owns interaction with the underlying environment and is responsible
    for returning structured observations and timestep records.
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> TimestepRecord:
        raise NotImplementedError

    @abstractmethod
    def action_space_n(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def observation_space_n(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class GenerativeModel(ABC):
    """
    Abstract generative model interface.

    Concrete implementations will eventually own A, B, C, D, policy spaces,
    and all relevant dimensional metadata.
    """

    @abstractmethod
    def num_actions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def num_observations(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def num_states(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def to_config(self) -> Mapping[str, Any]:
        raise NotImplementedError


class StateInferenceEngine(ABC):
    """
    Interface for hidden-state inference.
    """

    @abstractmethod
    def infer(
        self,
        prior_belief: BeliefState,
        observation: Observation,
        model: GenerativeModel,
    ) -> BeliefState:
        raise NotImplementedError


class PolicyInferenceEngine(ABC):
    """
    Interface for policy scoring / posterior computation.
    """

    @abstractmethod
    def evaluate(
        self,
        belief_state: BeliefState,
        candidate_policies: list[Policy],
        model: GenerativeModel,
    ) -> list[PolicyEvaluation]:
        raise NotImplementedError


class Agent(ABC):
    """
    Abstract decision-making agent.
    """

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe(self, observation: Observation) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_action(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        observation: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        raise NotImplementedError


class ExperimentRunner(ABC):
    """
    Interface for experiment orchestration.
    """

    @abstractmethod
    def run(self) -> ExperimentResult:
        raise NotImplementedError


class ArtifactStore(ABC):
    """
    Interface for saving/loading experiment artifacts.
    """

    @abstractmethod
    def save_episode(self, episode: EpisodeRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_result(self, result: ExperimentResult) -> None:
        raise NotImplementedError
