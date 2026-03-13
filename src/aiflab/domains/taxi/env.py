"""
Env
"""

from typing import Any

import gymnasium as gym

from aiflab.core.dataclasses import TimestepRecord
from aiflab.core.interfaces import Domain
from aiflab.domains.taxi.factorization import latent_state_from_index
from aiflab.domains.taxi.observations import observation_from_gym
from aiflab.domains.taxi.transitions import classify_step


class TaxiDomain(Domain):
    """
    Gymnasium Taxi-v3 wrapped behind the AIFLab Domain interface.
    """

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        episode_id: int = 0,
    ) -> None:
        self._env = gym.make("Taxi-v3", render_mode=render_mode)
        self._episode_id = episode_id
        self._timestep = 0
        self._last_observation = None

    def reset(self, seed: int | None = None):
        obs_index, info = self._env.reset(seed=seed)
        self._timestep = 0
        observation = observation_from_gym(obs_index, info)
        self._last_observation = observation
        return observation

    def step(self, action: int) -> TimestepRecord:
        obs_index, reward, terminated, truncated, info = self._env.step(action)

        observation = observation_from_gym(obs_index, info)
        latent_state = latent_state_from_index(obs_index)
        semantics = classify_step(
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

        record = TimestepRecord(
            episode_id=self._episode_id,
            timestep=self._timestep,
            observation=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            latent_state=latent_state,
            belief_state=None,
            policy_evaluations=(),
            info={
                "prob": info.get("prob"),
                "action_mask": (
                    tuple(info["action_mask"]) if "action_mask" in info else None
                ),
                "action_name": semantics.action_name,
            },
        )

        self._last_observation = observation
        self._timestep += 1
        return record

    def action_space_n(self) -> int:
        return int(self._env.action_space.n)

    def observation_space_n(self) -> int | None:
        return int(self._env.observation_space.n)

    def render(self) -> Any:
        return self._env.render()

    def decode(self, state_index: int):
        """
        Convenience passthrough to Gymnasium Taxi's decode semantics, but routed
        through our canonical factorization.
        """
        return latent_state_from_index(state_index)

    def close(self) -> None:
        self._env.close()
