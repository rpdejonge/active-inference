from aiflab.core.dataclasses import (
    BeliefState,
    EpisodeRecord,
    ExperimentResult,
    LatentState,
    Observation,
    Policy,
    PolicyEvaluation,
    TimestepRecord,
)


def test_observation_accepts_index_only() -> None:
    obs = Observation(index=3)
    assert obs.index == 3
    assert obs.values is None


def test_observation_accepts_values_only() -> None:
    obs = Observation(values=(1, 2, 3))
    assert obs.values == (1, 2, 3)
    assert obs.index is None


def test_observation_rejects_missing_payload() -> None:
    try:
        Observation()
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "at least one" in str(exc)


def test_observation_rejects_invalid_action_mask() -> None:
    try:
        Observation(index=1, action_mask=(1, 2, 0))
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "0/1" in str(exc)


def test_latent_state_accepts_factors() -> None:
    state = LatentState(factors=(4, 2, 1))
    assert state.factors == (4, 2, 1)


def test_belief_state_requires_nonempty_positive_probabilities() -> None:
    belief = BeliefState(probabilities=(0.2, 0.8))
    assert abs(sum(belief.probabilities) - 1.0) < 1e-9


def test_belief_state_rejects_negative_probabilities() -> None:
    try:
        BeliefState(probabilities=(0.5, -0.5, 1.0))
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "non-negative" in str(exc)


def test_policy_horizon_matches_actions() -> None:
    policy = Policy(actions=(0, 1, 2), horizon=3)
    assert policy.horizon == 3


def test_policy_rejects_mismatched_horizon() -> None:
    try:
        Policy(actions=(0, 1), horizon=3)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "horizon" in str(exc)


def test_policy_evaluation_holds_policy_and_score() -> None:
    policy = Policy(actions=(1, 2))
    pe = PolicyEvaluation(policy=policy, score=-1.25, posterior_probability=0.7)
    assert pe.policy == policy
    assert pe.score == -1.25
    assert pe.posterior_probability == 0.7


def test_timestep_record_done_property() -> None:
    obs = Observation(index=42)
    ts = TimestepRecord(
        episode_id=1,
        timestep=0,
        observation=obs,
        action=2,
        reward=-1.0,
        terminated=False,
        truncated=True,
    )
    assert ts.done is True


def test_episode_record_requires_nonempty_timesteps() -> None:
    obs = Observation(index=0)
    ts = TimestepRecord(
        episode_id=1,
        timestep=0,
        observation=obs,
        action=0,
        reward=0.0,
        terminated=True,
        truncated=False,
    )
    ep = EpisodeRecord(
        episode_id=1,
        timesteps=(ts,),
        total_reward=0.0,
    )
    assert len(ep.timesteps) == 1


def test_experiment_result_requires_name() -> None:
    obs = Observation(index=0)
    ts = TimestepRecord(
        episode_id=1,
        timestep=0,
        observation=obs,
        action=0,
        reward=1.0,
        terminated=True,
        truncated=False,
    )
    ep = EpisodeRecord(
        episode_id=1,
        timesteps=(ts,),
        total_reward=1.0,
    )
    result = ExperimentResult(
        experiment_name="smoke",
        episodes=(ep,),
        metrics={"reward_mean": 1.0},
        config={"seed": 0},
    )
    assert result.experiment_name == "smoke"
    assert result.metrics["reward_mean"] == 1.0