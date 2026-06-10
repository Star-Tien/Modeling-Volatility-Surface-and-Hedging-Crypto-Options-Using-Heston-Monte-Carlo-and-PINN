import os
import sys
import json
import numpy as np
import pandas as pd
import time

# Reconfigure stdout/stderr to utf-8 for Windows consoles
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add parent directory to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
from src.math_utils import heston_analytical_price, heston_monte_carlo_price

def main():
    # 1. Load Heston parameters
    # Cố định hạt giống ngẫu nhiên để đảm bảo tính tái lập (Reproducibility)
    np.random.seed(42)

    param_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/heston_parameters.json'))
    if not os.path.exists(param_path):
        print(f"❌ Không tìm thấy file {param_path}. Vui lòng chạy Giai đoạn 2 trước.")
        return

    with open(param_path, "r") as f:
        heston_params = json.load(f)

    v0 = heston_params["v0"]
    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    xi = heston_params["xi"]
    rho = heston_params["rho"]

    print("🎉 Nạp thành công bộ tham số Heston:")
    print(f"   v0: {v0:.5f}, kappa: {kappa:.5f}, theta: {theta:.5f}, xi: {xi:.5f}, rho: {rho:.5f}")

    # 2. Load clean options data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/02_btc_options_with_iv.csv'))
    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file {data_path}. Vui lòng chạy Giai đoạn 2 trước.")
        return

    df_clean = pd.read_csv(data_path)
    print(f"📊 Đã nạp {len(df_clean)} hợp đồng để chạy kiểm thử Monte Carlo.")

    # 3. Pricing using Analytical and Monte Carlo
    print("\n🔄 Đang chạy mô phỏng Monte Carlo ngẫu nhiên (20,000 paths, 100 steps)...")
    start_time = time.time()
    
    # Pre-allocate arrays to speed up operations
    S_arr = df_clean['underlying_price_S'].to_numpy()
    K_arr = df_clean['strike_K'].to_numpy()
    T_arr = df_clean['time_to_maturity_T'].to_numpy()
    r_arr = df_clean['risk_free_rate_r'].to_numpy()
    types = df_clean['option_type'].to_numpy()
    
    n_contracts = len(df_clean)
    prices_analytical = np.zeros(n_contracts)
    prices_mc = np.zeros(n_contracts)
    
    for i in range(n_contracts):
        # Analytical Price (Fourier)
        prices_analytical[i] = heston_analytical_price(
            S_arr[i], K_arr[i], T_arr[i], r_arr[i],
            v0, kappa, theta, xi, rho, types[i]
        )
        
        # Monte Carlo Price (Euler-Maruyama)
        prices_mc[i] = heston_monte_carlo_price(
            S_arr[i], K_arr[i], T_arr[i], r_arr[i],
            v0, kappa, theta, xi, rho, types[i],
            N_steps=100, N_paths=20000
        )
        
        if (i + 1) % 100 == 0:
            print(f"   ✅ Đã định giá xong {i + 1}/{n_contracts} hợp đồng...")
            
    df_clean['price_heston_analytical'] = prices_analytical
    df_clean['price_heston_monte_carlo'] = prices_mc
    
    # 4. Compute error metrics
    df_clean['mc_error_abs'] = np.abs(df_clean['price_heston_analytical'] - df_clean['price_heston_monte_carlo'])
    df_clean['mc_error_rel'] = df_clean['mc_error_abs'] / np.maximum(df_clean['price_heston_analytical'], 1.0)
    
    elapsed = time.time() - start_time
    mae = df_clean['mc_error_abs'].mean()
    mre = df_clean['mc_error_rel'].mean() * 100
    
    print(f"\n✅ Mô phỏng hoàn tất trong {elapsed:.2f} giây!")
    print(f" 🔹 Sai số tuyệt đối trung bình (MAE): {mae:.4f} USD")
    print(f" 🔹 Sai số tương đối trung bình (MRE): {mre:.2f}%")

    # 5. Save results
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/03_heston_monte_carlo_prices.csv'))
    df_clean.to_csv(output_path, index=False)
    print(f"💾 Kết quả đã được lưu tại: {output_path}")

    # 6. Vẽ đồ thị phân bổ sai số theo Moneyness (S/K)
    print("\n🔄 Đang vẽ đồ thị sai số theo Moneyness (S/K)...")
    df_clean['moneyness'] = df_clean['underlying_price_S'] / df_clean['strike_K']

    fig, ax = plt.subplots(figsize=(11, 6))
    call_mask = df_clean['option_type'] == 'call'
    put_mask  = df_clean['option_type'] == 'put'

    ax.scatter(df_clean.loc[call_mask, 'moneyness'],
               df_clean.loc[call_mask, 'mc_error_rel'] * 100,
               c='steelblue', alpha=0.6, s=20, label='Call')
    ax.scatter(df_clean.loc[put_mask,  'moneyness'],
               df_clean.loc[put_mask,  'mc_error_rel'] * 100,
               c='tomato',    alpha=0.6, s=20, label='Put')

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.2, label='ATM (S/K = 1)')
    ax.axhline(y=df_clean['mc_error_rel'].mean() * 100, color='orange',
               linestyle=':', linewidth=1.5, label=f'MRE trung bình ({mre:.2f}%)')

    ax.set_title('Monte Carlo Relative Error vs. Moneyness (S/K)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Moneyness  S / K', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/plots'))
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'mc_error_by_moneyness.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Đồ thị phân bổ sai số đã được lưu tại: {plot_path}")

if __name__ == '__main__':
    main()
