
import argparse
import json
import os
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from planner import Planner, load_world_model
from train import get_device


def run(policy, n_episodes, seed=1000, reset=None):
    env = gym.make("Pendulum-v1")
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)  # same start states across methods
        if reset:
            reset()
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ret += r
            done = term or trunc
        returns.append(ret)
    env.close()
    return np.array(returns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/worldmodel.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--candidates", type=int, default=1000)
    p.add_argument("--sac-steps", type=int, default=0, help="0 = skip SAC baseline")
    p.add_argument("--sac-eval-every", type=int, default=5000)
    p.add_argument("--outdir", default="plots")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = get_device()
    model = load_world_model(args.model, device)
    metrics_path = os.path.splitext(args.model)[0] + "_metrics.json"
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else None
    n_data_steps = metrics["n_episodes"] * 200 if metrics else None

    results = {}
    rng = np.random.default_rng(0)
    results["Random policy"] = run(lambda o: rng.uniform(-2, 2, size=(1,)).astype(np.float32), args.episodes)
    for name, method in [("MPC: random shooting", "random"), ("MPC: CEM", "cem")]:
        pl = Planner(model, args.horizon, args.candidates, method, device=device)
        results[name] = run(lambda o, pl=pl: pl.act(o), args.episodes, reset=pl.reset)
    for k, v in results.items():
        print(f"{k:24s} mean {v.mean():8.1f}  std {v.std():6.1f}")

    # --- model-free SAC baseline (learning curve vs env steps) ---
    sac_curve = None
    if args.sac_steps > 0:
        from stable_baselines3 import SAC
        env = gym.make("Pendulum-v1")
        sac = SAC("MlpPolicy", env, seed=0, verbose=0)
        steps, means = [], []
        done_steps = 0
        while done_steps < args.sac_steps:
            sac.learn(args.sac_eval_every, reset_num_timesteps=False)
            done_steps += args.sac_eval_every
            r = run(lambda o: sac.predict(o, deterministic=True)[0], 10)
            steps.append(done_steps); means.append(r.mean())
            print(f"SAC @ {done_steps:6d} steps: {r.mean():8.1f}")
        sac_curve = (steps, means)

    # --- plot 1: final returns ---
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(results)
    ax.bar(names, [results[n].mean() for n in names], yerr=[results[n].std() for n in names], capsize=4)
    ax.set_ylabel(f"Episode return (mean ± std, {args.episodes} eps)")
    ax.set_title("Pendulum-v1: random vs. planning with learned world model")
    plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(f"{args.outdir}/returns.png", dpi=150); plt.close()

    # --- plot 2: sample efficiency vs SAC ---
    if sac_curve:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(*sac_curve, marker="o", label="SAC (model-free)")
        if n_data_steps:
            ax.scatter([n_data_steps], [results["MPC: CEM"].mean()], color="C3", marker="*", s=150,
                       label="CEM-MPC (world model, random data only)", zorder=3)
        ax.axhline(results["Random policy"].mean(), color="gray", ls="--", label="Random policy")
        ax.set_xlabel("Environment steps used for learning"); ax.set_ylabel("Mean return")
        ax.legend(); plt.tight_layout()
        plt.savefig(f"{args.outdir}/sample_efficiency.png", dpi=150); plt.close()

    # --- plot 3: world model quality ---
    if metrics:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(metrics["history"]["train"], label="train")
        axs[0].plot(metrics["history"]["val"], label="val")
        axs[0].set_yscale("log"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("normalized MSE")
        axs[0].set_title("World model training"); axs[0].legend()
        ms = {int(k): v for k, v in metrics["multistep_mse"].items()}
        axs[1].plot(list(ms), list(ms.values()), marker="o")
        axs[1].set_yscale("log"); axs[1].set_xlabel("rollout horizon (steps)")
        axs[1].set_ylabel("held-out MSE"); axs[1].set_title("Open-loop prediction error")
        plt.tight_layout(); plt.savefig(f"{args.outdir}/world_model.png", dpi=150); plt.close()

    np.savez(f"{args.outdir}/results.npz", **{k.replace(" ", "_").replace(":", ""): v for k, v in results.items()})


if __name__ == "__main__":
    main()
