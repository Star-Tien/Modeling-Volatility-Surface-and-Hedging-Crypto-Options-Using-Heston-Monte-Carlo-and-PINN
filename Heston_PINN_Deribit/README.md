# Modeling Volatility Surface and Hedging Crypto Options Using Heston, Monte Carlo, and PINN

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%2F%20SciPy-orange)](https://pytorch.org/)
[![Data Source](https://img.shields.io/badge/data%20source-Deribit%20API-green)](https://www.deribit.com/)

Mã nguồn phục vụ Đề tài Nghiên cứu Khoa học / Khóa luận tốt nghiệp:
**"Mô hình hóa bề mặt biến động và hedging quyền chọn Crypto bằng Heston, Monte Carlo và PINN: Bằng chứng thực nghiệm từ dữ liệu Deribit."**

---

## 📌 Tổng quan dự án (Project Overview)

Dự án này triển khai một cấu trúc nghiên cứu định lượng 4 tầng nhằm định giá, mô hình hóa bề mặt biến động ngầm định (Implicit Volatility Surface) và định giá quyền chọn Bitcoin (BTC Options).

Trọng tâm của nghiên cứu là sự đối đầu hiệu năng giữa hai phương pháp giải phương trình vi phân đạo hàm riêng (PDE) Heston:
1. **Phương pháp cổ điển:** Mô phỏng số ngẫu nhiên Monte Carlo (Euler-Maruyama scheme).
2. **Phương pháp đổi mới sáng tạo:** Mạng nơ-ron nhúng cấu trúc vật lý/toán học **PINN (Physics-Informed Neural Network)** sử dụng cơ chế tự động đạo hàm (Autograd) để tối ưu hóa hàm toán tử Loss PDE.

---

## 📂 Cấu trúc thư mục dự án (Repository Structure)

```text
📁 Heston_PINN_Deribit/
│
├── 📁 data/                      # Lưu trữ cơ sở dữ liệu của đề tài
│   ├── 📁 raw/                   # Dữ liệu ma trận quyền chọn thô tải từ API Deribit
│   └── 📁 processed/             # Dữ liệu sạch sau khi giải ngược IV, calib và mô phỏng
│
├── 📁 notebooks/                 # Tiến trình thực nghiệm (Jupyter Notebooks & Scripts)
│   ├── 📄 01_data_collection.ipynb   # Jupyter: Kết nối API Deribit và cào dữ liệu
│   ├── 📄 run_data_collection.py     # Script: Runner cào dữ liệu tự động
│   ├── 📄 02_vol_surface_calib.ipynb # Jupyter: Vẽ Vol Smile và ước lượng tham số Heston
│   ├── 📄 run_vol_surface_calib.py   # Script: Runner tối ưu tham số Nelder-Mead
│   ├── 📄 03_monte_carlo_simulation.ipynb # Jupyter: Chạy mô phỏng Monte Carlo đối chứng
│   ├── 📄 run_monte_carlo.py         # Script: Runner mô phỏng Monte Carlo
│   ├── 📄 04_pinn_heston_solver.ipynb # Jupyter: Huấn luyện mạng PINN giải PDE Heston
│   └── 📄 run_pinn_solver.py         # Script: Runner huấn luyện mạng PINN tối ưu
│
├── 📁 src/                       # Mã nguồn cốt lõi tái sử dụng (Production Code)
│   ├── 📄 __init__.py
│   ├── 📄 api_client.py          # Kết nối API sàn Deribit
│   ├── 📄 math_utils.py          # Hàm IV, Black-Scholes, Fourier Heston, Monte Carlo
│   └── 📄 models.py              # Kiến trúc mạng PyTorch Neural Network cho PINN
│
├── 📁 outputs/                   # Kết quả đầu ra của nghiên cứu
│   ├── 📁 models/                # Checkpoint trọng số mạng PINN (*.pt)
│   │   └── 📁 saved_models/      # Checkpoint hoàn thiện cuối cùng
│   └── 📁 plots/                 # Biểu đồ Vol Surface, Loss Curve, và Đồ thị sai số
│
├── 📄 requirements.txt           # Danh sách thư viện Python bắt buộc
└── 📄 README.md                  # Hướng dẫn chi tiết dự án (File này)
```

---

## 🛠️ Hướng dẫn cài đặt & Thiết lập (Installation & Setup)

### 1. Yêu cầu môi trường
* Python 3.9 trở lên
* Trình quản lý gói `pip` hoặc `conda`

### 2. Cài đặt thư viện bắt buộc
Chạy lệnh sau để cài đặt toàn bộ các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quy trình chạy tái lập kết quả (Replication Guide)

Dự án được thực thi tuần tự qua 4 giai đoạn logic chặt chẽ:

### Giai đoạn 1: Thu thập Dữ liệu (Data Collection)
Kết nối trực tiếp tới API Deribit để tải dữ liệu ma trận quyền chọn BTC thực tế (gồm giá Bid, Ask, Underlying, Strike, Maturity):
```bash
python notebooks/run_data_collection.py
```
* **Kết quả đầu ra**: Tệp dữ liệu thô tại `data/raw/btc_options_raw.csv`.

### Giai đoạn 2: Tính Implied Volatility & Hiệu chuẩn Heston (Calibration)
Giải ngược IV bằng Newton-Raphson, vẽ bề mặt biến động 3D và dùng Nelder-Mead để tìm bộ tham số tối ưu $(\kappa, \theta, \xi, \rho, v_0)$ khớp với thị trường:
```bash
python notebooks/run_vol_surface_calib.py
```
* **Kết quả đầu ra**:
  - Bề mặt biến động: `outputs/plots/volatility_surface.png`
  - Tham số Heston: `data/processed/heston_parameters.json`
  - Dữ liệu sạch có IV: `data/processed/02_btc_options_with_iv.csv`

### Giai đoạn 3: Đối chứng Monte Carlo (Monte Carlo Benchmark)
Sử dụng sơ đồ Full Truncation Euler-Maruyama định giá quyền chọn Heston với 20,000 đường đi mô phỏng để làm mốc đối chứng:
```bash
python notebooks/run_monte_carlo.py
```
* **Kết quả đầu ra**:
  - Dữ liệu so sánh: `data/processed/03_heston_monte_carlo_prices.csv`
  - Đồ thị sai số: `outputs/plots/mc_error_by_moneyness.png`

### Giai đoạn 4: Giải Heston PDE bằng PINN (PINN Heston Solver)
Huấn luyện mạng Neural Network nhúng cấu trúc vật lý của phương trình đạo hàm riêng Heston PDE kết hợp với dữ liệu thực tế:
```bash
python notebooks/run_pinn_solver.py
```
* **Kết quả đầu ra**:
  - Checkpoint mô hình: `outputs/models/saved_models/heston_pinn.pt`
  - Đường cong Loss: `outputs/plots/pinn_loss_curve.png`
  - Đồ thị đối chứng PINN vs Analytical: `outputs/plots/pinn_vs_analytical.png`
  - Bảng dự đoán chi tiết: `data/processed/04_pinn_predictions.csv`

---

## 📈 Kết quả thực nghiệm chính (Key Results)

### 1. Hiệu chuẩn tham số Heston (Deribit BTC Options)
* Hệ số tương quan đòn bẩy $\rho = 0.7608$ mang **dấu dương**, phản ánh hiệu ứng FOMO đặc thù của thị trường Bitcoin (khi giá tăng kéo theo sự biến động tăng mạnh), hoàn toàn ngược lại so với thị trường chứng khoán truyền thống ($\rho < 0$).

### 2. So sánh hiệu năng định giá (PINN vs. Monte Carlo)
Mốc đánh giá so sánh giá trị dự đoán với giá trị giải tích chính xác Fourier Heston:

| Chỉ báo | Monte Carlo Benchmark (Stage 3) | PINN Heston Solver (Stage 4) |
| :--- | :---: | :---: |
| **Hệ số xác định ($R^2$)** | - | **0.993648 (99.36%)** |
| **Sai số tuyệt đối trung bình (MAE)** | 897.03 USD | **681.97 USD** |

* **Độ chính xác đột phá**: Chỉ số $R^2 = 99.36\%$ chứng minh mạng PINN đã học và mô hình hóa thành công cấu trúc toàn bộ bề mặt biến động giá tùy chọn BTC. 
* **Regularization hiệu quả**: Sai số MAE của PINN nhỏ hơn Monte Carlo chứng tỏ ràng buộc vật lý Heston PDE hoạt động như một bộ chỉnh chuẩn (Regularizer) tuyệt vời, tránh được các nhiễu số học.

---

## 💡 Các đột phá toán học giúp PINN đạt $R^2 > 0.99$

1. **Put-Call Parity Mapping**: Mạng chỉ học hàm Call option trơn tru, toàn bộ Put option được ánh xạ sang Call khi huấn luyện và khôi phục ngược lại khi đánh giá, loại bỏ triệt để xung đột ánh xạ.
2. **Scale Invariance ($V_{ref} = S_{ref}$)**: Đồng bộ hóa tỷ lệ giúp loại bỏ hệ số tỷ lệ sai lệch trong các điều kiện biên của PDE.
3. **Focused Collocation Sampling**: Lấy mẫu tập trung 70% điểm nội miền vào vùng hoạt động chính ($s \in [0.5, 1.5], \tau \in [0.0, 0.2]$) thay vì rải đều vô ích.
4. **Dynamic Loss Weighting**: Điều chỉnh trọng số loss dữ liệu tăng dần từ $1.0$ lên $50.0$ sau 1000 epoch để mạng học được khung PDE vật lý trước khi khớp khít vào các điểm dữ liệu.