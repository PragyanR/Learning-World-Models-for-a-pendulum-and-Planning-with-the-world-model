
import torch
import torch.nn as nn

MAX_SPEED = 8.0  

class WorldModel(nn.Module):
    def __init__(self, obs_dim: int = 3, act_dim: int = 1, hidden: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )
        
        for name, dim in [("in_mean", obs_dim + act_dim), ("in_std", obs_dim + act_dim),
                          ("delta_mean", obs_dim), ("delta_std", obs_dim)]:
            self.register_buffer(name, torch.zeros(dim) if "mean" in name else torch.ones(dim))

    @torch.no_grad()
    def set_normalization(self, obs, act, next_obs):
        x = torch.cat([obs, act], dim=-1)
        delta = next_obs - obs
        self.in_mean.copy_(x.mean(0))
        self.in_std.copy_(x.std(0) + 1e-6)
        self.delta_mean.copy_(delta.mean(0))
        self.delta_std.copy_(delta.std(0) + 1e-6)

    def predict_delta_norm(self, obs, act):
        """Normalized predicted delta; this is what the training loss is on."""
        x = (torch.cat([obs, act], dim=-1) - self.in_mean) / self.in_std
        return self.net(x)

    def forward(self, obs, act):
        """Predict next obs (un-normalized)."""
        delta = self.predict_delta_norm(obs, act) * self.delta_std + self.delta_mean
        nxt = obs + delta
        # keep (cos, sin) on the unit circle and theta_dot in the valid range so
        # errors don't compound into impossible states during long rollouts
        cs = nxt[..., :2]
        cs = cs / cs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        thdot = nxt[..., 2:].clamp(-MAX_SPEED, MAX_SPEED)
        return torch.cat([cs, thdot], dim=-1)
