import gymnasium as gym


def main() -> None:
    env = gym.make("Taxi-v3")
    obs, info = env.reset(seed=42)

    print("initial obs:", obs)
    print("action space:", env.action_space)
    print("observation space:", env.observation_space)
    print("initial info keys:", list(info.keys()))

    terminated = False
    truncated = False
    step_idx = 0

    while not (terminated or truncated) and step_idx < 10:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={step_idx} action={action} "
            f"obs={next_obs} reward={reward} "
            f"terminated={terminated} truncated={truncated}"
        )

        obs = next_obs
        step_idx += 1

    env.close()

if __name__ == "__main__":
    main()