import os
import sys
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv

# Reconfigure stdout/stderr to utf-8 for Windows consoles
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Python versions < 3.7 do not have reconfigure

# Add parent directory to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.api_client import get_all_btc_options, get_order_book

def main():
    # Load environment variables
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env'))
    load_dotenv(dotenv_path=env_path)
    
    print("🔄 Bước 1: Đang quét danh sách hợp đồng quyền chọn BTC từ Deribit...")
    raw_instruments = get_all_btc_options()
    print(f"📊 Tìm thấy {len(raw_instruments)} hợp đồng đang hoạt động.")

    if not raw_instruments:
        print("❌ Không lấy được danh sách hợp đồng. Vui lòng kiểm tra lại kết nối hoặc API.")
        return

    # Quét toàn bộ danh sách hợp đồng quyền chọn khi chạy thực tế (khoảng 300 - 1000+ hợp đồng)
    sampled_instruments = raw_instruments 

    all_data = []
    print("\n🔄 Bước 2: Đang quét sổ lệnh và bóc tách giá thị trường thời gian thực...")

    for idx, inst in enumerate(sampled_instruments):
        name = inst['instrument_name']
        strike = inst['strike']
        opt_type = inst['option_type']
        exp_timestamp = inst['expiration_timestamp']
        
        market_prices = get_order_book(name)
        
        if market_prices:
            # Chỉ lấy khi cả hai bên mua và bán đều có người đặt lệnh (Thanh khoản thực)
            if market_prices["best_bid_price"] > 0 and market_prices["best_ask_price"] > 0:
                # Tính giá trung bình thị trường (Mid Price)
                mid_price = (market_prices["best_bid_price"] + market_prices["best_ask_price"]) / 2
                
                row = {
                    "instrument_name": name,
                    "strike_K": strike,
                    "option_type": opt_type,
                    "expiration_timestamp": exp_timestamp,
                    "underlying_price_S": market_prices["underlying_price"],
                    "option_price_V": mid_price,
                    "best_bid": market_prices["best_bid_price"],
                    "best_ask": market_prices["best_ask_price"]
                }
                all_data.append(row)
                
        # Tránh rate limit (20 requests/sec)
        time.sleep(0.05)
        
        if (idx + 1) % 50 == 0:
            print(f"   ✅ Đã xử lý xong {idx + 1}/{len(sampled_instruments)} hợp đồng...")

    # ==========================================
    # BƯỚC 4: CHUYỂN THÀNH DATAFRAME VÀ XUẤT FILE
    # ==========================================
    if all_data:
        df_final = pd.DataFrame(all_data)
        
        # Đổi định dạng ngày đáo hạn sang YYYY-MM-DD
        df_final['expiration_date'] = df_final['expiration_timestamp'].apply(
            lambda x: datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d')
        )
        
        # Tính thời gian đáo hạn T (đơn vị: Năm) - Giả định lãi suất phi rủi ro r = 4.5%
        today = datetime.now()
        df_final['time_to_maturity_T'] = df_final['expiration_timestamp'].apply(
            lambda x: max((datetime.fromtimestamp(x / 1000) - today).total_seconds() / (365.0 * 86400.0), 0.001)
        )
        df_final['risk_free_rate_r'] = 0.045
        
        # Sắp xếp lại các cột
        final_columns = [
            'instrument_name', 'option_type', 'underlying_price_S', 
            'strike_K', 'time_to_maturity_T', 'risk_free_rate_r', 'option_price_V'
        ]
        df_final = df_final[final_columns]
        
        # Tạo thư mục và lưu file
        raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw'))
        os.makedirs(raw_dir, exist_ok=True)
        csv_path = os.path.join(raw_dir, 'btc_options_raw.csv')
        df_final.to_csv(csv_path, index=False)
        
        print("\n🎉 GIAI ĐOẠN 1 HOÀN THÀNH XUẤT SẮC!")
        print(f"File dữ liệu '{csv_path}' đã được lưu thành công. Tổng số dòng sạch: {len(df_final)}")
        print(df_final.head())
    else:
        print("\n❌ Không thu thập được dữ liệu thị trường.")

if __name__ == '__main__':
    main()
