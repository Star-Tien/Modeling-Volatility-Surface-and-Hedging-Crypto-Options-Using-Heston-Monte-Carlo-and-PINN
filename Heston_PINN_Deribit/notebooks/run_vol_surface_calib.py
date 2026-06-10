import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

# Reconfigure stdout/stderr to utf-8 for Windows consoles
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add parent directory to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.math_utils import (
    calculate_implied_volatility,
    heston_analytical_price
)

def heston_calibration_loss(params, data_df):
    v0, kappa, theta, xi, rho = params
    
    # Mathematical constraints (Feller condition and boundary limits)
    if xi <= 0 or kappa <= 0 or theta <= 0 or v0 <= 0 or abs(rho) >= 1.0:
        return 1e10
    if 2 * kappa * theta <= xi**2:  # Soft Feller constraint penalty
        penalty = 1e4
    else:
        penalty = 0.0
        
    loss = 0.0
    # Chuyển thành mảng numpy trước vòng lặp để tăng tốc độ xử lý
    S_arr = data_df['underlying_price_S'].to_numpy()
    K_arr = data_df['strike_K'].to_numpy()
    T_arr = data_df['time_to_maturity_T'].to_numpy()
    r_arr = data_df['risk_free_rate_r'].to_numpy()
    V_market = data_df['option_price_V'].to_numpy()
    types = data_df['option_type'].to_numpy()

    for i in range(len(data_df)):
        p_model = heston_analytical_price(
            S_arr[i], K_arr[i], T_arr[i], r_arr[i], 
            v0, kappa, theta, xi, rho, types[i]
        )
        loss += (p_model - V_market[i]) ** 2
        
    return (loss / len(data_df)) + penalty

def main():
    raw_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/btc_options_raw.csv'))
    if not os.path.exists(raw_csv_path):
        print(f"❌ Không tìm thấy file {raw_csv_path}. Vui lòng chạy Giai đoạn 1 trước.")
        return

    df = pd.read_csv(raw_csv_path)
    print(f"📊 Đã nạp thành công {len(df)} hợp đồng quyền chọn.")

    # Quy đổi giá quyền chọn từ BTC sang USD để đồng nhất đơn vị với S và K
    df['option_price_V'] = df['option_price_V'] * df['underlying_price_S']

    # Calculate IV
    print("🔄 Đang tính toán Implied Volatility (IV)...")
    df['implied_vol'] = df.apply(
        lambda row: calculate_implied_volatility(
            row['option_price_V'], row['underlying_price_S'], row['strike_K'], 
            row['time_to_maturity_T'], row['risk_free_rate_r'], row['option_type']
        ), axis=1
    )

    # Clean data (drop options where IV solver didn't converge)
    df_clean = df.dropna(subset=['implied_vol']).reset_index(drop=True)
    print(f"✅ Đã làm sạch dữ liệu. Số lượng hợp đồng giữ lại: {len(df_clean)} / {len(df)}")

    if len(df_clean) < 15:
        print(f"❌ Lỗi: Số lượng hợp đồng hợp lệ quá ít ({len(df_clean)}). Không đủ dữ liệu để hiệu chuẩn mô hình Heston.")
        return

    print("🔄 Bước 4: Đang tiến hành Khớp tham số Heston bằng thuật toán Nelder-Mead...")
    # Initial guess: [v0, kappa, theta, xi, rho]
    initial_params = [0.05, 2.0, 0.05, 0.4, -0.6]

    # Run Nelder-Mead with maxiter=150 as recommended
    result = minimize(
        heston_calibration_loss, 
        initial_params, 
        args=(df_clean,), 
        method='Nelder-Mead', 
        options={'maxiter': 150, 'disp': True}
    )

    v0_opt, kappa_opt, theta_opt, xi_opt, rho_opt = result.x
    print("\n🎉 BỘ THAM SỐ LÝ TƯỞNG CỦA MÔ HÌNH HESTON (ĐÃ CALIBRATED):")
    print(f" 🔹 Phương sai tức thời (v0) : {v0_opt:.5f}")
    print(f" 🔹 Tốc độ hút trung bình (kappa)   : {kappa_opt:.5f}")
    print(f" 🔹 Biến động dài hạn (theta)    : {theta_opt:.5f}")
    print(f" 🔹 Độ biến động phương sai (xi)  : {xi_opt:.5f}")
    print(f" 🔹 Hệ số tương quan đòn bẩy (rho): {rho_opt:.5f}")

    # 1. Save Heston parameters as JSON
    param_output = {
        "v0": float(v0_opt), "kappa": float(kappa_opt), 
        "theta": float(theta_opt), "xi": float(xi_opt), "rho": float(rho_opt)
    }
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(processed_dir, exist_ok=True)
    param_path = os.path.join(processed_dir, 'heston_parameters.json')
    with open(param_path, "w") as f:
        json.dump(param_output, f, indent=4)
    print(f"✅ Đã lưu bộ tham số Heston tại: {param_path}")

    # 2. Save processed dataframe containing IV
    clean_csv_path = os.path.join(processed_dir, '02_btc_options_with_iv.csv')
    df_clean.to_csv(clean_csv_path, index=False)
    print(f"✅ Đã lưu dữ liệu chứa IV tại: {clean_csv_path}")

    # 3. Plot 3D Volatility Surface
    print("🔄 Đang vẽ đồ thị 3D Volatility Surface...")
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    X = df_clean['strike_K']
    Y = df_clean['time_to_maturity_T']
    Z = df_clean['implied_vol']

    mappable = ax.scatter(X, Y, Z, c=Z, cmap='coolwarm', s=35, edgecolors='k', alpha=0.8)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label='Implied Volatility (IV)')

    ax.set_title("BTC Options Implied Volatility Surface (Deribit)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Strike Price ($K$)", fontsize=11)
    ax.set_ylabel("Time to Maturity ($T$ - Years)", fontsize=11)
    ax.set_zlabel("Implied Volatility (IV)", fontsize=11)

    # Perspective view angle
    ax.view_init(elev=25, azim=-120)

    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/plots'))
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'volatility_surface.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 Đồ thị 3D Volatility Surface đã được kết xuất tại: {plot_path}")

if __name__ == '__main__':
    main()
