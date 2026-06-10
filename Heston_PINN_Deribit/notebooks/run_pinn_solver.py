"""
Giai doan 4: Huan luyen PINN Giai PDE Heston (v3 - 4-input voi Put-Call Parity va Chuan hoa S_ref)
========================================================================================
Mo hinh PINN nhan 4 dau vao: (s, v, tau, k)
  s   = S / S_ref  (gia tai san chuan hoa)
  v   = phuong sai tuc thoi (variance)
  tau = thoi gian con lai (time-to-maturity)
  k   = K / S_ref  (gia thuc hien chuan hoa)

Cac cai tien de dat R^2 > 0.90:
1. Put-Call Parity: Chuyen tat ca gia option ve gia Call tuong duong. PINN chi hoc Call option.
   Khi tinh toan thuc te/danh gia, Put option se duoc khoi phuc lai tu Call qua Put-Call parity.
2. V_ref = S_ref: Loai bo hoan toan bat dong nhat giua physics loss va data loss.
3. Collocation tap trung: 70% so diem collocation lay quanh vung thanh khoan lon [0.5, 1.5] x [0.0, 0.2].
4. Trong so loss dong: Trong 1000 epoch dau, LAMBDA_DATA = 1.0. Tu epoch 1001 tro di nang len 50.0.
"""

import os
import sys

# Force UTF-8 output so Vietnamese/emoji print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import HestonPINN, pde_residual, boundary_loss

# ─────────────────────────────────────────────────
# 0. THIET LAP
# ─────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
N_PDE        = 2048
N_BC         = 1024
ADAM_EPOCHS  = 3000
LBFGS_ITER   = 25      # L-BFGS outer loops (each = max_iter=20 inner steps)
LAMBDA_BC    = 10.0
LAMBDA_DATA  = 50.0    # Dynamic: starts at 1.0, ramps to 50.0 after epoch 1000
LR           = 1e-3
HIDDEN_LAYERS = 6
HIDDEN_UNITS  = 128

S_max        = 5.0
K_max        = 4.5

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE.upper()}")

# ─────────────────────────────────────────────────
# 1. NAP DU LIEU
# ─────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
param_path = os.path.join(BASE_DIR, 'data', 'processed', 'heston_parameters.json')
data_path  = os.path.join(BASE_DIR, 'data', 'processed', '03_heston_monte_carlo_prices.csv')

with open(param_path) as f:
    params = json.load(f)

kappa = float(params['kappa'])
theta = float(params['theta'])
xi    = float(params['xi'])
rho   = float(params['rho'])
v0    = float(params['v0'])
r     = 0.045

print(f"Heston params: kappa={kappa:.4f}, theta={theta:.4f}, xi={xi:.4f}, rho={rho:.4f}, v0={v0:.4f}")

df = pd.read_csv(data_path)
df = df[df['price_heston_analytical'] > 5.0].copy().reset_index(drop=True)
print(f"Training contracts: {len(df)}")

# ─────────────────────────────────────────────────
# 2. CHUAN HOA DU LIEU & PUT-CALL PARITY MAPPING
# ─────────────────────────────────────────────────
# Chuyen doi Put ve Call tuong duong qua Put-Call Parity
df['equiv_call'] = np.where(
    df['option_type'] == 'call',
    df['price_heston_analytical'],
    df['price_heston_analytical'] + df['underlying_price_S'] - df['strike_K'] * np.exp(-r * df['time_to_maturity_T'])
)

S_ref = float(df['underlying_price_S'].mean())   # ~63,000 USD
V_ref = S_ref                                    # Set V_ref = S_ref for scale-invariant boundaries

# All 4 inputs normalised
s_norm   = (df['underlying_price_S'].values / S_ref).astype(np.float32)
k_norm   = (df['strike_K'].values           / S_ref).astype(np.float32)
tau_data = df['time_to_maturity_T'].values.astype(np.float32)
v_arr    = np.full(len(df), v0, dtype=np.float32)          # initial variance

# Normalised target Call price
V_norm = (df['equiv_call'].values / V_ref).astype(np.float32)

print(f"S_ref={S_ref:.0f} USD | V_ref={V_ref:.0f} USD (Scale: 1.0)")

# Convert data arrays to GPU tensors once
s_data_t   = torch.tensor(s_norm,   device=DEVICE).unsqueeze(1)
v_data_t   = torch.tensor(v_arr,    device=DEVICE).unsqueeze(1)
tau_data_t = torch.tensor(tau_data, device=DEVICE).unsqueeze(1)
k_data_t   = torch.tensor(k_norm,   device=DEVICE).unsqueeze(1)
V_data_t   = torch.tensor(V_norm,   device=DEVICE).unsqueeze(1)

# ─────────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────────
model = HestonPINN(hidden_layers=HIDDEN_LAYERS, hidden_units=HIDDEN_UNITS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"PINN: {HIDDEN_LAYERS} layers x {HIDDEN_UNITS} units = {n_params:,} params  [4-input: s,v,tau,k]")

# ─────────────────────────────────────────────────
# 4. LOSS FUNCTION
# ─────────────────────────────────────────────────
loss_history = []

def sample_collocation():
    # Number of points in the focused region (70%) and uniform region (30%)
    N_focus = int(N_PDE * 0.70)
    N_uni = N_PDE - N_focus
    
    # 1. Focused region: s in [0.5, 1.5], tau in [0.0, 0.2]
    # v_c in [v0 * 0.5, v0 * 2.0], k_c in [0.5, 1.5]
    s_focus = 0.5 + torch.rand(N_focus, 1, device=DEVICE) * 1.0
    v_focus = (0.5 * v0) + torch.rand(N_focus, 1, device=DEVICE) * (1.5 * v0)
    t_focus = torch.rand(N_focus, 1, device=DEVICE) * 0.2
    k_focus = 0.5 + torch.rand(N_focus, 1, device=DEVICE) * 1.0
    
    # 2. Uniform region: s in [0, S_max], tau in [0, 1.0]
    # v_c in [1e-5, 2.0 * v0], k_c in [0, K_max]
    s_uni = torch.rand(N_uni, 1, device=DEVICE) * S_max
    v_uni = torch.rand(N_uni, 1, device=DEVICE) * 2.0 * v0 + 1e-5
    t_uni = torch.rand(N_uni, 1, device=DEVICE) * 1.0
    k_uni = torch.rand(N_uni, 1, device=DEVICE) * K_max
    
    # Concatenate
    s_c = torch.cat([s_focus, s_uni], dim=0).requires_grad_(True)
    v_c = torch.cat([v_focus, v_uni], dim=0).requires_grad_(True)
    t_c = torch.cat([t_focus, t_uni], dim=0).requires_grad_(True)
    k_c = torch.cat([k_focus, k_uni], dim=0) # no grad needed for strike
    
    return s_c, v_c, t_c, k_c


def compute_loss(ep=None):
    # PDE residual loss
    s_c, v_c, t_c, k_c = sample_collocation()
    res     = pde_residual(model, s_c, v_c, t_c, k_c, r, kappa, theta, xi, rho)
    l_pde   = torch.mean(res ** 2)

    # Boundary condition loss (per-contract strikes)
    l_bc = boundary_loss(model, r, S_max=S_max, K_max=K_max, N_bc=N_BC, device=DEVICE)

    # Data matching loss (Heston analytical prices)
    V_pred  = model(s_data_t, v_data_t, tau_data_t, k_data_t)
    l_data  = torch.mean((V_pred - V_data_t) ** 2)

    # Dynamic loss weighting
    # For first 1000 epochs: data weight = 1.0. After that: data weight = 50.0
    w_data = LAMBDA_DATA
    if ep is not None and ep <= 1000:
        w_data = 1.0

    total = l_pde + LAMBDA_BC * l_bc + w_data * l_data
    return total, l_pde, l_bc, l_data, w_data

# ─────────────────────────────────────────────────
# 5. PHA 1 — ADAM
# ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Phase 1: Adam ({ADAM_EPOCHS} epochs)")
print(f"{'='*60}")

opt_adam = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=ADAM_EPOCHS, eta_min=1e-5)

t0 = time.time()
for ep in range(1, ADAM_EPOCHS + 1):
    opt_adam.zero_grad()
    total, l_pde, l_bc, l_data, w_data = compute_loss(ep)
    total.backward()
    opt_adam.step()
    scheduler.step()
    loss_history.append(total.item())

    if ep % 300 == 0 or ep == 1:
        print(f"  ep={ep:5d} | Loss={total.item():.5f} | "
              f"PDE={l_pde.item():.5f} | BC={l_bc.item():.5f} | "
              f"Data={l_data.item():.5f} (w_data={w_data:.1f}) | t={time.time()-t0:.0f}s")

print(f"\nPhase 1 done | Final Loss = {loss_history[-1]:.6f}")

# ─────────────────────────────────────────────────
# 6. PHA 2 — L-BFGS
# ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Phase 2: L-BFGS ({LBFGS_ITER} outer loops x 20 inner steps)")
print(f"{'='*60}")

opt_lb = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20,
                     history_size=50, line_search_fn='strong_wolfe')
it = [0]

def closure():
    opt_lb.zero_grad()
    # closure uses ep=None, meaning full LAMBDA_DATA = 50.0
    total, _, _, _, _ = compute_loss()
    total.backward()
    it[0] += 1
    loss_history.append(total.item())
    return total

for i in range(LBFGS_ITER):
    opt_lb.step(closure)
    if (i + 1) % 5 == 0:
        print(f"  L-BFGS loop {i+1:3d}/{LBFGS_ITER} | Loss = {loss_history[-1]:.6f}")

print(f"\nPhase 2 done | Final Loss = {loss_history[-1]:.6f}")

# ─────────────────────────────────────────────────
# 7. LUU MODEL
# ─────────────────────────────────────────────────
model_dir = os.path.join(BASE_DIR, 'outputs', 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'heston_pinn.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'hidden_layers': HIDDEN_LAYERS,
    'hidden_units':  HIDDEN_UNITS,
    'S_ref': S_ref,
    'V_ref': V_ref,
    'heston_params': params,
}, model_path)
print(f"\nModel saved: {model_path}")

# ─────────────────────────────────────────────────
# 8. VE LOSS CURVE
# ─────────────────────────────────────────────────
plots_dir = os.path.join(BASE_DIR, 'outputs', 'plots')
os.makedirs(plots_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(loss_history, color='royalblue', linewidth=1.2)
ax.axvline(x=ADAM_EPOCHS, color='red', ls='--', alpha=0.6,
           label=f'Adam -> L-BFGS boundary (ep {ADAM_EPOCHS})')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Total Loss (log scale)')
ax.set_title('PINN Training Loss — Heston PDE (4-input)', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
loss_plot = os.path.join(plots_dir, 'pinn_loss_curve.png')
plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Loss curve saved: {loss_plot}")

# ─────────────────────────────────────────────────
# 9. DANH GIA (Put-Call Parity Recovery)
# ─────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    V_pinn_norm = model(s_data_t, v_data_t, tau_data_t, k_data_t).cpu().numpy().flatten()

# predicted call price
V_pinn_call = V_pinn_norm * S_ref

# Convert back to Put option prices for Put contracts
df['price_pinn'] = np.where(
    df['option_type'] == 'call',
    V_pinn_call,
    V_pinn_call - df['underlying_price_S'] + df['strike_K'] * np.exp(-r * df['time_to_maturity_T'])
)
# Clamp option prices to be non-negative
df['price_pinn'] = np.maximum(df['price_pinn'].values, 0.0)

V_pinn = df['price_pinn'].values
V_anal = df['price_heston_analytical'].values

mae  = np.mean(np.abs(V_pinn - V_anal))
mre  = np.mean(np.abs(V_pinn - V_anal) / np.maximum(V_anal, 1.0)) * 100
ss_res = np.sum((V_pinn - V_anal) ** 2)
ss_tot = np.sum((V_anal - V_anal.mean()) ** 2)
r2   = 1 - ss_res / ss_tot

print(f"\n{'='*60}")
print(f"  PINN vs Heston Analytical (Fourier) - Put-Call Parity Corrected:")
print(f"  MAE  = {mae:.4f} USD")
print(f"  MRE  = {mre:.2f}%")
print(f"  R2   = {r2:.6f}")
print(f"{'='*60}")

# ─────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(V_anal, V_pinn, alpha=0.4, s=12, c='steelblue')
lm = max(V_anal.max(), V_pinn.max())
axes[0].plot([0, lm], [0, lm], 'r--', lw=1.5, label='Perfect fit')
axes[0].set_xlabel('Heston Analytical Price (USD)')
axes[0].set_ylabel('PINN Predicted Price (USD)')
axes[0].set_title(f'PINN vs Analytical  [R2={r2:.4f}]', fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

rel_err = np.abs(V_pinn - V_anal) / np.maximum(V_anal, 1.0) * 100
tau_days = df['time_to_maturity_T'].values * 365
axes[1].scatter(tau_days, rel_err, alpha=0.4, s=12, c='tomato')
axes[1].axhline(y=mre, color='orange', ls=':', lw=1.5,
                label=f'MRE avg ({mre:.2f}%)')
axes[1].set_xlabel('Days to Maturity')
axes[1].set_ylabel('Relative Error (%)')
axes[1].set_title('PINN Relative Error by Maturity', fontweight='bold')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
comp_plot = os.path.join(plots_dir, 'pinn_vs_analytical.png')
plt.savefig(comp_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Comparison plot saved: {comp_plot}")

# ─────────────────────────────────────────────────
# 11. LUU CSV
# ─────────────────────────────────────────────────
df['pinn_error_abs']  = np.abs(V_pinn - V_anal)
df['pinn_error_rel']  = df['pinn_error_abs'] / np.maximum(V_anal, 1.0)

out_path = os.path.join(BASE_DIR, 'data', 'processed', '04_pinn_predictions.csv')
df.to_csv(out_path, index=False)
print(f"Predictions saved: {out_path}")

print("\n=== Stage 4 Complete! ===")
