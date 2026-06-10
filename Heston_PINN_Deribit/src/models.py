import torch
import torch.nn as nn
import numpy as np


class HestonPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the Heston PDE.

    Input:  (s, v, tau, k) — normalised spot, variance, time-to-maturity,
                              normalised strike  k = K / S_ref
    Output: V(s, v, tau, k) — normalised option price  V / V_ref

    Adding strike K as a 4th input allows the network to generalise
    across the full surface of contracts with different strikes.
    The Heston PDE is independent of K; K only appears in the terminal
    payoff condition  V(s, v, 0, k) = max(s - k, 0).

    Architecture: deep MLP with Tanh activations and Xavier initialisation.
    """

    def __init__(self, hidden_layers: int = 6, hidden_units: int = 128):
        super().__init__()

        layers = []
        in_features = 4  # (s, v, tau, k)

        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.Tanh())
            in_features = hidden_units

        layers.append(nn.Linear(in_features, 1))   # scalar output V_norm
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        s:   torch.Tensor,
        v:   torch.Tensor,
        tau: torch.Tensor,
        k:   torch.Tensor,
    ) -> torch.Tensor:
        """
        s, v, tau, k: tensors of shape (N, 1).
        s and tau must have requires_grad=True for the PDE autograd pass.
        Returns V of shape (N, 1).
        """
        x = torch.cat([s, v, tau, k], dim=1)
        return self.net(x)


def pde_residual(
    model:  "HestonPINN",
    s:      torch.Tensor,   # requires_grad=True
    v:      torch.Tensor,   # requires_grad=True
    tau:    torch.Tensor,   # requires_grad=True
    k:      torch.Tensor,   # strike — no grad needed for PDE
    r:      float,
    kappa:  float,
    theta:  float,
    xi:     float,
    rho:    float,
) -> torch.Tensor:
    """
    Heston PDE residual at interior collocation points (forward-time form):

        dV/dtau - 0.5*s^2*v*V_ss - rho*xi*s*v*V_sv
                - 0.5*xi^2*v*V_vv - r*s*V_s
                - kappa*(theta - v)*V_v + r*V = 0

    k is passed through to the network but K does not appear in the PDE;
    it only enters the terminal boundary condition.

    Returns tensor of shape (N, 1).
    """
    V = model(s, v, tau, k)

    ones = torch.ones_like(V)

    grad_V = torch.autograd.grad(
        V, [s, v, tau],
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )
    V_s, V_v, V_tau = grad_V

    V_ss = torch.autograd.grad(
        V_s, s,
        grad_outputs=torch.ones_like(V_s),
        create_graph=True,
        retain_graph=True,
    )[0]

    V_vv = torch.autograd.grad(
        V_v, v,
        grad_outputs=torch.ones_like(V_v),
        create_graph=True,
        retain_graph=True,
    )[0]

    V_sv = torch.autograd.grad(
        V_s, v,
        grad_outputs=torch.ones_like(V_s),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual = (
        V_tau
        - 0.5 * s**2 * v * V_ss
        - rho * xi * s * v * V_sv
        - 0.5 * xi**2 * v * V_vv
        - r * s * V_s
        - kappa * (theta - v) * V_v
        + r * V
    )
    return residual


def boundary_loss(
    model:  "HestonPINN",
    r:      float,
    S_max:  float = 5.0,
    K_max:  float = 4.5,
    N_bc:   int = 1024,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Boundary / terminal condition losses:

    1. Terminal payoff  (tau = 0):
       V(s, v, 0, k) = max(s - k, 0)   [call payoff]
       s is drawn uniformly over [0, S_max], k is drawn over [0, K_max].

    2. Left asset boundary  (s = 0):
       V(0, v, tau, k) = 0

    3. Right asset boundary  (s = S_max):
       V(S_max, v, tau, k) = S_max - k * exp(-r * tau)   [intrinsic value at large s]

    Returns scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=device)

    # ── 1. Terminal payoff ──────────────────────────────────────────────────
    s_t   = torch.rand(N_bc, 1, device=device) * S_max
    v_t   = torch.rand(N_bc, 1, device=device) * 0.5 + 1e-4
    tau_0 = torch.zeros(N_bc, 1, device=device)
    k_t   = torch.rand(N_bc, 1, device=device) * K_max   # random normalised strikes
    V_pred = model(s_t, v_t, tau_0, k_t)
    V_true = torch.clamp(s_t - k_t, min=0.0)
    loss = loss + torch.mean((V_pred - V_true) ** 2)

    # ── 2. Left boundary: s = 0  →  V = 0 ─────────────────────────────────
    s_0   = torch.zeros(N_bc, 1, device=device)
    v_0   = torch.rand(N_bc, 1, device=device) * 0.5 + 1e-4
    tau_0b = torch.rand(N_bc, 1, device=device)
    k_0   = torch.rand(N_bc, 1, device=device) * K_max
    V_left = model(s_0, v_0, tau_0b, k_0)
    loss = loss + torch.mean(V_left ** 2)

    # ── 3. Right boundary: s = S_max  →  V ≈ S_max - k * exp(-r*tau) ─────────────
    s_r   = torch.full((N_bc, 1), S_max, device=device)
    v_r   = torch.rand(N_bc, 1, device=device) * 0.5 + 1e-4
    tau_r = torch.rand(N_bc, 1, device=device)
    k_r   = torch.rand(N_bc, 1, device=device) * K_max
    V_right_pred = model(s_r, v_r, tau_r, k_r)
    V_right_true = S_max - k_r * torch.exp(-r * tau_r)
    loss = loss + torch.mean((V_right_pred - V_right_true) ** 2)

    return loss

