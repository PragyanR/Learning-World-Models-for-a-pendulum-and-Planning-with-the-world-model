import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from worldmodel import WorldModel


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def flatten(obs, act):
    """(E,T+1,3),(E,T,1) -> per-transition tensors."""
    o = obs[:, :-1].reshape(-1, obs.shape[-1])
    n = obs[:, 1:].reshape(-1, obs.shape[-1])
    a = act.reshape(-1, act.shape[-1])
    return [torch.as_tensor(x) for x in (o, a, n)]


@torch.no_grad()
def multistep_error(model, obs, act, horizons=(1, 5, 10, 20, 50), device="cpu"):
    """Open-loop rollout error using the true action sequence, on held-out episodes.
    Returns mean squared error of the (cos, sin, thdot) state at each horizon."""
    model.eval()
    obs = torch.as_tensor(obs).to(device)
    act = torch.as_tensor(act).to(device)
    s = obs[:, 0]
    errs = {}
    for t in range(max(horizons)):
        s = model(s, act[:, t])
        if (t + 1) in horizons:
            errs[t + 1] = F.mse_loss(s, obs[:, t + 1]).item()
    return errs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/random_data.npz")
    p.add_argument("--out", default="checkpoints/worldmodel.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print("device:", device)

    d = np.load(args.data)
    obs, act = d["obs"], d["act"]
    E = obs.shape[0]
    perm = np.random.permutation(E)
    n_train = int(0.9 * E)
    tr, va = perm[:n_train], perm[n_train:]

    o_tr, a_tr, n_tr = [x.to(device) for x in flatten(obs[tr], act[tr])]
    o_va, a_va, n_va = [x.to(device) for x in flatten(obs[va], act[va])]
    print(f"train transitions: {len(o_tr)}, val transitions: {len(o_va)}")

    model = WorldModel(obs.shape[-1], act.shape[-1], args.hidden).to(device)
    model.set_normalization(o_tr, a_tr, n_tr)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def loss_fn(o, a, n):
        target = (n - o - model.delta_mean) / model.delta_std
        return F.mse_loss(model.predict_delta_norm(o, a), target)

    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(len(o_tr), device=device)
        total = 0.0
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            loss = loss_fn(o_tr[b], a_tr[b], n_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = loss_fn(o_va, a_va, n_va).item()
        history["train"].append(total / len(o_tr))
        history["val"].append(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  train {history['train'][-1]:.5f}  val {val:.5f}")

    model.load_state_dict(best_state)
    errs = multistep_error(model, obs[va], act[va], device=device)
    print("held-out multi-step MSE:", {k: round(v, 5) for k, v in errs.items()})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden,
                "obs_dim": obs.shape[-1], "act_dim": act.shape[-1]}, args.out)
    with open(os.path.splitext(args.out)[0] + "_metrics.json", "w") as f:
        json.dump({"history": history, "multistep_mse": errs, "n_episodes": int(E)}, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
