import yfinance as yf
import numpy as np
from arch import arch_model

# --- Configuration v3.0 ---
RISK_FREE_RATE = 0.045  # 4.5%
TRADING_DAYS = 252
COMMISSION_PER_CONTRACT = 0.65 # $0.65 per contract
SLIPPAGE_PCT = 0.02 # 2% slippage


def binomial_tree_price(S, K, T, r, sigma, N=100, option_type='call'):
    """
    Calculate American option price using Binomial Tree (CRR model).
    Handles early exercise.
    """
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros(N + 1)
    for i in range(N + 1):
        asset_prices[i] = S * (u ** (N - i)) * (d ** i)
        
    # Initialize option values at maturity
    option_values = np.zeros(N + 1)
    for i in range(N + 1):
        if option_type == 'call':
            option_values[i] = max(0, asset_prices[i] - K)
        else:
            option_values[i] = max(0, K - asset_prices[i])
            
    # Step back through the tree
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            asset_prices[i] = S * (u ** (j - i)) * (d ** i)
            continuation_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            
            # Check for early exercise (American Option)
            if option_type == 'call':
                exercise_value = max(0, asset_prices[i] - K)
            else:
                exercise_value = max(0, K - asset_prices[i])
                
            option_values[i] = max(continuation_value, exercise_value)
            
    return option_values[0]

def get_vol_skew(atm_vol, strike, spot):
    """
    Simulate Volatility Skew (Smirk).
    """
    moneyness = strike / spot
    skew_factor = 1.0
    
    if moneyness < 1.0: # OTM Put / ITM Call
        skew_factor = 1.0 + (1.0 - moneyness) * 0.5 
    else: # OTM Call / ITM Put
        skew_factor = 1.0 + (moneyness - 1.0) * 0.2
        
    return atm_vol * skew_factor

def apply_transaction_costs(price, contracts=1):
    """
    Add slippage and commission.
    """
    buy_price = price * (1 + SLIPPAGE_PCT) + (COMMISSION_PER_CONTRACT / 100)
    sell_price = price * (1 - SLIPPAGE_PCT) - (COMMISSION_PER_CONTRACT / 100)
    return buy_price, sell_price

# --- Advanced Volatility Modeling (GARCH + VIX) ---

def get_garch_volatility(history):
    """
    Estimate Conditional Volatility using GARCH(1,1).
    Returns the annualized volatility for the last day.
    """
    returns = 100 * np.log(history['Close'] / history['Close'].shift(1)).dropna()
    
    if len(returns) < 30:
        # Fallback to simple std dev if not enough data
        return returns.std() / 100 * np.sqrt(TRADING_DAYS)
        
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        # Forecast next day volatility
        forecast = res.forecast(horizon=1)
        daily_vol = np.sqrt(forecast.variance.iloc[-1, 0])
        annualized_vol = (daily_vol / 100) * np.sqrt(TRADING_DAYS)
        return annualized_vol
    except:
        # Fallback
        return returns.std() / 100 * np.sqrt(TRADING_DAYS)

def fetch_vix_data(start_date, end_date):
    """Fetch VIX data for scaling."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(start=start_date, end=end_date)
        return hist['Close']
    except:
        return None

def fetch_data(ticker_symbol, start_date, end_date):
    """Fetch historical data from yfinance."""
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(start=start_date, end=end_date)
    return hist