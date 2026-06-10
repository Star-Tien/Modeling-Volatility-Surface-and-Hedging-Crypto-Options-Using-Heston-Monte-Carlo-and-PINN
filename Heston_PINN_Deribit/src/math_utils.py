import numpy as np
import cmath
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Standard Black-Scholes pricing formula for Call/Put options.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_vega(S, K, T, r, sigma):
    """
    Vega (first derivative of option price with respect to volatility) for Newton-Raphson.
    """
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def calculate_implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Invert the Black-Scholes formula using the Newton-Raphson solver to compute Implied Volatility.
    """
    intrinsic_value = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if market_price <= intrinsic_value:
        return np.nan
        
    sigma = 0.5  # Initial guess (50% volatility)
    max_iter = 100
    tolerance = 1e-6
    
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = black_scholes_vega(S, K, T, r, sigma)
        
        if vega < 1e-4:  # Avoid division by small vega values
            break
            
        diff = price - market_price
        if abs(diff) < tolerance:
            return sigma
            
        sigma = sigma - diff / vega  # Newton step
        
        if sigma <= 0 or sigma > 3.0:  # Boundaries for crypto IV
            break
            
    return np.nan

def heston_char_func(u, S0, v0, kappa, theta, xi, rho, t, r):
    """
    Characteristic function of the Heston model, vectorized over numpy array u.
    """
    rsi = rho * xi * u * 1j
    d = np.sqrt((rsi - kappa)**2 + xi**2 * (u**2 + u * 1j))
    g = (kappa - rsi - d) / (kappa - rsi + d)
    
    C = (r * u * 1j * t + (kappa * theta / xi**2) * ((kappa - rsi - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))))
    D = ((kappa - rsi - d) / xi**2) * ((1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t)))
    
    return np.exp(C + D * v0 + u * 1j * np.log(S0))

def heston_analytical_price(S0, K, T, r, v0, kappa, theta, xi, rho, option_type='call'):
    """
    Analytical Heston option pricing using Fourier transform integration (Gauss-Legendre approximation).
    Vectorized using numpy arrays over the integration grid.
    """
    # Numerical integration approximation from 0 to 200 with 500 grid points
    u_space = np.linspace(0.0005, 200, 500)
    du = u_space[1] - u_space[0]
    
    # Calculate P_num = 1
    u_param1 = u_space - 1j
    phi1 = heston_char_func(u_param1, S0, v0, kappa, theta, xi, rho, T, r)
    # denominator heston_char_func(-1j, S0, ...) is a single complex number
    denom = heston_char_func(-1j, S0, v0, kappa, theta, xi, rho, T, r)
    phi1 = phi1 / denom
    integrand1 = (np.exp(-1j * u_space * np.log(K)) * phi1 / (1j * u_space)).real
    I1 = np.sum(integrand1) * du
    
    # Calculate P_num = 2
    phi2 = heston_char_func(u_space, S0, v0, kappa, theta, xi, rho, T, r)
    integrand2 = (np.exp(-1j * u_space * np.log(K)) * phi2 / (1j * u_space)).real
    I2 = np.sum(integrand2) * du
    
    P1 = 0.5 + I1 / np.pi
    P2 = 0.5 + I2 / np.pi
    
    call_price = S0 * P1 - K * np.exp(-r * T) * P2
    
    if option_type == 'call':
        return max(call_price, 0.0)
    else:  # Put-Call Parity
        return max(call_price + K * np.exp(-r * T) - S0, 0.0)

def heston_monte_carlo_price(S0, K, T, r, v0, kappa, theta, xi, rho, option_type='call', N_steps=100, N_paths=20000):
    """
    Định giá quyền chọn Heston bằng mô phỏng Monte Carlo (Sơ đồ Full Truncation Euler-Maruyama).
    """
    if T <= 0:
        return max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)
        
    dt = T / N_steps
    
    # Khởi tạo ma trận lưu trữ đường đi của tài sản (S) và phương sai (v)
    S = np.full((N_paths,), float(S0))
    v = np.full((N_paths,), float(v0))
    
    # Tạo các biến ngẫu nhiên chuẩn có tương quan tương hỗ \rho
    for t in range(N_steps):
        Z_S = np.random.normal(0.0, 1.0, N_paths)
        Z_v_raw = np.random.normal(0.0, 1.0, N_paths)
        
        # Áp đặt hệ số tương quan rho
        Z_v = rho * Z_S + np.sqrt(1.0 - rho**2) * Z_v_raw
        
        # Sơ đồ Full Truncation: Lấy max(v, 0) để tránh phương sai bị âm dưới căn thức
        v_plus = np.maximum(v, 0.0)
        
        # Cập nhật đường đi của tài sản S
        S = S * np.exp((r - 0.5 * v_plus) * dt + np.sqrt(v_plus * dt) * Z_S)
        
        # Cập nhật đường đi của phương sai v (Euler-Maruyama)
        v = v + kappa * (theta - v_plus) * dt + xi * np.sqrt(v_plus * dt) * Z_v
        
    # Tính giá trị kỳ vọng nhận được tại thời điểm đáo hạn T (Payoff)
    if option_type == 'call':
        payoffs = np.maximum(S - K, 0.0)
    else:
        payoffs = np.maximum(K - S, 0.0)
        
    # Chiết khấu dòng tiền về hiện tại bằng lãi suất phi rủi ro r
    price = np.exp(-r * T) * np.mean(payoffs)
    return float(price)
