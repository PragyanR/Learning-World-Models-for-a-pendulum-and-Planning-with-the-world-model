
import numpy as np
import torch



from worldmodel import WorldModel

MAX_TORQUE = 2.0


def pendulum_cost(obs, act):
    
    theta = torch.atan2(obs[..., 1], obs[..., 0])
    return theta ** 2 + 0.1 * obs[..., 2] ** 2 + 0.001 * act.squeeze(-1) ** 2


def load_world_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = WorldModel(ck["obs_dim"], ck["act_dim"], ck["hidden"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


class Planner:
    def __init__(self, model, horizon=20, n_candidates=1000, method="cem",
                 n_iters=3, elite_frac=0.1, device="cpu"):
        assert method in ("random", "cem")
        self.model, self.device = model, device
        self.H, self.N, self.method = horizon, n_candidates, method
        self.n_iters = n_iters if method == "cem" else 1
        self.n_elite = max(2, int(elite_frac * n_candidates))
        self.prev_mean = None

    def reset(self):
        self.prev_mean = None

    @torch.no_grad()
    def _cost(self, obs, acts):
        """obs: (3,), acts: (N,H,1) -> total predicted cost per candidate (N,)."""
        s = obs.unsqueeze(0).expand(acts.shape[0], -1)
        total = torch.zeros(acts.shape[0], device=self.device)
        for t in range(self.H):
            total += pendulum_cost(s, acts[:, t])
            s = self.model(s, acts[:, t])
        return total + pendulum_cost(s, torch.zeros_like(acts[:, 0]))  # terminal state cost

    @torch.no_grad()
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        H, N = self.H, self.N

        if self.method == "random":
            acts = (torch.rand(N, H, 1, device=self.device) * 2 - 1) * MAX_TORQUE
            best = acts[self._cost(obs, acts).argmin()]
            return best[0].cpu().numpy()

        # CEM, warm-started from the previous plan shifted by one step
        mean = torch.zeros(H, 1, device=self.device)
        if self.prev_mean is not None:
            mean[:-1] = self.prev_mean[1:]
        std = torch.full((H, 1), MAX_TORQUE / 2, device=self.device)
        for _ in range(self.n_iters):
            acts = (mean + std * torch.randn(N, H, 1, device=self.device)).clamp(-MAX_TORQUE, MAX_TORQUE)
            elite = acts[self._cost(obs, acts).topk(self.n_elite, largest=False).indices]
            mean, std = elite.mean(0), elite.std(0) + 1e-3
        self.prev_mean = mean
        return mean[0].cpu().numpy()
