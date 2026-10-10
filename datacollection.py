import argparse
import numpy as np
import gymnasium as gym


def collect(n_episodes: int, seed: int = 0, env_id: str = "Pendulum-v1"):
    env = gym.make(env_id)
    env.action_space.seed(seed)
    obs_all, act_all = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_ep, act_ep = [obs], []
        while True:
            a = env.action_space.sample()  # uniform random torque
            obs, _, terminated, truncated, _ = env.step(a)
            act_ep.append(a)
            obs_ep.append(obs)
            if terminated or truncated:
                break
        obs_all.append(np.asarray(obs_ep, dtype=np.float32))
        act_all.append(np.asarray(act_ep, dtype=np.float32))
    env.close()
    return np.stack(obs_all), np.stack(act_all)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100, help="200 env steps each")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/random_data.npz")
    args = p.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    obs, act = collect(args.episodes, args.seed)
    np.savez(args.out, obs=obs, act=act)
    print(f"Saved {obs.shape[0]} episodes ({act.shape[0] * act.shape[1]} transitions) to {args.out}")
