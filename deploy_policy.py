"""Standalone policy inference for sim2real deployment.

Pure PyTorch — no mjlab / rsl_rl dependency.
Loads a trained PPO actor checkpoint and computes actions from observations.
"""

import argparse
import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Running-mean / running-std observation normalizer (matches rsl_rl)."""

    def __init__(self, shape: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class PolicyNetwork(nn.Module):
    """Standalone actor MLP that mirrors the rsl_rl MLPModel layout.

    Architecture (from checkpoint):
        obs_normalizer -> Linear(40,512)->ELU -> Linear(512,256)->ELU
                       -> Linear(256,128)->ELU -> Linear(128,6)
    """

    def __init__(self, obs_dim: int = 40, action_dim: int = 6,
                 hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(obs_dim)

        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic forward: normalise obs, run MLP, return action means."""
        x = self.obs_normalizer(obs)
        return self.mlp(x)

    @staticmethod
    def from_checkpoint(path: str, device: str = "cpu") -> "PolicyNetwork":
        """Load a trained rsl_rl PPO checkpoint into this standalone network."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        actor_sd = ckpt["actor_state_dict"]

        # Infer dimensions from weight shapes
        obs_dim = actor_sd["mlp.0.weight"].shape[1]
        action_dim = actor_sd["mlp.6.weight"].shape[0]
        hidden_dims = (
            actor_sd["mlp.0.weight"].shape[0],
            actor_sd["mlp.2.weight"].shape[0],
            actor_sd["mlp.4.weight"].shape[0],
        )

        policy = PolicyNetwork(obs_dim, action_dim, hidden_dims)
        # Load only the keys we care about (skip 'std' — that's exploration noise)
        filtered = {k: v for k, v in actor_sd.items() if k != "std"}
        policy.load_state_dict(filtered, strict=True)
        policy.to(device).eval()
        return policy


def main():
    parser = argparse.ArgumentParser(description="Test standalone policy inference")
    parser.add_argument("model", type=str, help="Path to model_XXX.pt checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of inference steps to run with random obs")
    args = parser.parse_args()

    policy = PolicyNetwork.from_checkpoint(args.model, args.device)
    print(f"Loaded policy: obs_dim=40  action_dim=6  device={args.device}")
    print(policy)

    # Run inference with random observations
    print(f"\n--- Running {args.num_steps} steps with random observations ---")
    for i in range(args.num_steps):
        obs = torch.randn(1, 40, device=args.device)
        with torch.no_grad():
            action = policy(obs)
        print(f"step {i:3d} | action: {action.squeeze().tolist()}")


if __name__ == "__main__":
    main()
