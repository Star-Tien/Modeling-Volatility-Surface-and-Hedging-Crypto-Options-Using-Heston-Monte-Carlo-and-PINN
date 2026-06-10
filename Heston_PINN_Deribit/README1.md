# Modeling Volatility Surface and Hedging Crypto Options Using Heston, Monte Carlo, and PINN

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%2F%20SciPy-orange)](https://pytorch.org/)
[![Data Source](https://img.shields.io/badge/data%20source-Deribit%20API-green)](https://www.deribit.com/)

Mã nguồn phục vụ Đề tài Nghiên cứu Khoa học / Khóa luận tốt nghiệp: 
**"Mô hình hóa bề mặt biến động và hedging quyền chọn Crypto bằng Heston, Monte Carlo và PINN: Bằng chứng thực nghiệm từ dữ liệu Deribit."**

---

## 📌 Tổng quan dự án (Project Overview)

Dự án này triển khai một cấu trúc nghiên cứu định lượng 4 tầng nhằm định giá, mô hình hóa bề mặt biến động ngầm định (Implicit Volatility Surface) và mô phỏng phòng ngừa rủi ro động (Dynamic Hedging) cho quyền chọn Bitcoin (BTC Options). 

Trọng tâm của nghiên cứu là sự đối đầu hiệu năng giữa hai phương pháp giải phương trình vi phân đạo hàm riêng (PDE) Heston:
1. **Phương pháp cổ điển:** Mô phỏng số ngẫu nhiên Monte Carlo (Euler-Maruyama scheme).
2. **Phương pháp đổi mới sáng tạo:** Mạng nơ-ron nhúng cấu trúc toán học **PINN (Physics-Informed Neural Network)** sử dụng cơ chế tự động đạo hàm (Autograd) để tối ưu hóa hàm toán tử Loss PDE.

---

## 📂 Cấu trúc thư mục dự án (Repository Structure)

```text
📁 Heston_PINN_Deribit/
│
├── 📁 data/                      # Lưu trữ cơ sở dữ liệu của đề tài
│   ├── 📁 raw/                   # Dữ liệu ma trận quyền chọn thô tải từ API Deribit
│   └── 📁 processed/             # Dữ liệu sạch sau khi giải ngược IV và Calib tham số
│
├── 📁 notebooks/                 # Tiến trình thực nghiệm (Jupyter Notebooks)
│   ├── 📄 01_data_collection.ipynb   # Bước 1: Kết nối API Deribit và cào dữ liệu
│   ├── 📄 02_vol_surface_calib.ipynb # Bước 2: Vẽ Vol Smile và ước lượng tham số Heston
│   ├── 📄 03_monte_carlo_bench.ipynb # Bước 3: Chạy mô phỏng Monte Carlo đối chứng
│   └── 📄 04_pinn_heston_solver.ipynb # Bước 4: Huấn luyện mạng PINN giải PDE Heston
│
├── 📁 src/                       # Mã nguồn cốt lõi tái sử dụng (Production Code)
│   ├── 📄 __init__.py
│   ├── 📄 api_client.py          # Cấu hình kết nối API (Public/Private Rest API)
│   ├── 📄 math_utils.py          # Thuật toán tính IV, Delta, và sơ đồ rời rạc hóa số
│   └── 📄 models.py              # Kiến trúc mạng Neural Network (MLP) cho PINN
│
├── 📁 outputs/                   # Kết quả đầu ra của nghiên cứu
│   ├── 📁 models/                # Lưu checkpoint trọng số mạng PINN (*.pt, *.pth)
│   └── 📁 plots/                 # Biểu đồ Vol Surface, Loss Curve, và Đồ thị sai số
│
├── 📄 .env                       # Lưu biến môi trường (API Keys) - Bảo mật, không commit
├── 📄 requirements.txt           # Danh sách thư viện Python bắt buộc
└── 📄 README.md                  # Hướng dẫn chi tiết dự án (File này)
```
