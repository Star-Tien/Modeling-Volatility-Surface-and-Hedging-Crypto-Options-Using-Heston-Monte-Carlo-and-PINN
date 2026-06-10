import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# We can fall back to testnet if not specified in the environment
API_BASE_URL = os.getenv("DERIBIT_API_URL", "https://test.deribit.com").rstrip("/")
BASE_URL = f"{API_BASE_URL}/api/v2" if not API_BASE_URL.endswith("/api/v2") else API_BASE_URL

def get_all_btc_options(currency="BTC"):
    """
    Fetch all active options contracts for a given currency (e.g. BTC, ETH) from Deribit.
    """
    endpoint = f"{BASE_URL}/public/get_instruments"
    params = {"currency": currency, "kind": "option", "expired": "false"}
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("result", [])
    except Exception as e:
        print(f"Error fetching options: {e}")
    return []

def get_order_book(instrument_name):
    """
    Fetch the order book details (bid, ask, underlying price, mark price) for a specific contract.
    """
    endpoint = f"{BASE_URL}/public/get_order_book"
    params = {"instrument_name": instrument_name}
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        if response.status_code == 200:
            res = response.json().get("result", {})
            return {
                "best_bid_price": res.get("best_bid_price", 0),       # Best bid price
                "best_ask_price": res.get("best_ask_price", 0),       # Best ask price
                "underlying_price": res.get("underlying_price", 0),   # Current spot price of underlying asset
                "mark_price": res.get("mark_price", 0)                # Current mark price
            }
    except Exception as e:
        print(f"Error fetching order book for {instrument_name}: {e}")
    return None
