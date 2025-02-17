import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm


def calibrate_gbm(df):
    """
    Calibrate a simple GBM from historical data.
    df must have columns: Open, High, Low, Close, with a DateTime index.
    
    Returns dictionary of { 'mu': ..., 'sigma': ... }, daily parameters.
    """
    close_prices = df['Close'].dropna()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    return {
        'mu': mu,
        'sigma': sigma
    }

def calibrate_jump_diffusion(df, jump_threshold=3.0):
    """
    Calibrate Jump-Diffusion (Merton) using a basic threshold method for jumps.
    Returns dict with daily parameters: mu, sigma, lambda, nu, delta.
    jump_threshold is the factor of std dev above which a return is considered a jump.
    """
    close_prices = df['Close'].dropna()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    mean_ret = log_returns.mean()
    std_ret = log_returns.std()
    
    threshold_value = mean_ret + jump_threshold * std_ret
    negative_threshold_value = mean_ret - jump_threshold * std_ret
    
    jump_indices = log_returns[(log_returns > threshold_value) | 
                               (log_returns < negative_threshold_value)].index
    jump_values = log_returns.loc[jump_indices]
    
    total_days = len(log_returns)
    num_jumps = len(jump_values)
    lam = num_jumps / total_days  # daily jump intensity
    

    if num_jumps > 0:
        nu = jump_values.mean()    # mean of log(1+J) ~ mean(r)
        delta = jump_values.std()  # std of log(1+J)  ~ std(r)
    else:
        nu = 0.0
        delta = 0.0
    
    # Continuous part (excluding jump days)
    non_jumps = log_returns.drop(jump_indices)
    mu_cont = non_jumps.mean()
    sigma_cont = non_jumps.std()
    
    return {
        'mu': mu_cont,
        'sigma': sigma_cont,
        'lambda': lam,
        'nu': nu,
        'delta': delta
    }



def calibrate_heston(df):
    """
    Calibrate Heston model from historical data.
    Ensures positive kappa and theta estimates.
    
    Returns dict with: {'mu', 'kappa', 'theta', 'sigma_v', 'rho', 'v0'}
    """
    close_prices = df['Close'].dropna()
    log_rets = np.log(close_prices / close_prices.shift(1)).dropna()
    
    mu_est = log_rets.mean()
    
    # Proxy for variance: v_t = (log_rets)^2
    v = log_rets**2
    v_next = v.shift(-1).dropna()
    v_curr = v.dropna()
    v_next = v_next.loc[v_curr.index.intersection(v_next.index)]
    
    # Kappa-theta estimation using OLS
    y = v_next - v_curr
    X = -v_curr  
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    alpha, beta = model.params[0], model.params[1]
    
    # Ensure positive values
    kappa_est = max(-beta, 1e-3)
    theta_est = max(alpha / kappa_est, 1e-8)  # Ensure non-negative theta

    # Estimate sigma_v from residual approach
    res = model.resid
    valid_mask = (v_curr > 0)
    numerator = (res[valid_mask]**2).sum()
    denominator = v_curr[valid_mask].sum()
    sigma_v_est = np.sqrt(numerator / denominator) if denominator > 0 else 0.01
    
    # Rough correlation estimate between dW1 and dW2
    dW1 = log_rets / np.sqrt(v)
    dW2 = res / (sigma_v_est * np.sqrt(v))
    common_index = dW1.dropna().index.intersection(dW2.dropna().index)
    rho_est = np.corrcoef(dW1.loc[common_index], dW2.loc[common_index])[0,1]
    
    v0_est = max(v.iloc[0], 1e-8)  # Ensure positive v0
    
    return {
        'mu': mu_est,
        'kappa': kappa_est,
        'theta': theta_est,
        'sigma_v': sigma_v_est,
        'rho': rho_est,
        'v0': v0_est
    }


def calibrate_bates(df):
    """
    Bates Model = Heston + Jump-Diffusion.
    Ensures kappa, theta > 0 and lambda detection threshold.
    
    Returns dict: {'mu', 'kappa', 'theta', 'sigma_v', 'rho', 'v0', 'lambda', 'nu', 'delta'}
    """
    heston_params = calibrate_heston(df)
    jd_params = calibrate_jump_diffusion(df, jump_threshold=3.0)
    
    # Ensure lambda is meaningful
    lam = jd_params['lambda']
    if lam < 1e-4:  # If lambda is too small, ignore jumps
        lam = 0.0
        nu = 0.0
        delta = 0.0
    else:
        nu = jd_params['nu']
        delta = jd_params['delta']
    
    bates_params = {
        'mu': heston_params['mu'],
        'kappa': heston_params['kappa'],
        'theta': heston_params['theta'],
        'sigma_v': heston_params['sigma_v'],
        'rho': heston_params['rho'],
        'v0': heston_params['v0'],
        'lambda': lam,
        'nu': nu,
        'delta': delta
    }
    
    return bates_params

###############################################################################


def simulate_gbm(params, S0, n_days, dt=1/252, seed=42):
    """
    Simulate daily closes from a calibrated GBM model.
    params: {'mu':..., 'sigma':...} in daily terms
    """
    np.random.seed(seed)
    mu, sigma = params['mu'], params['sigma']
    S = np.zeros(n_days+1)
    S[0] = S0
    
    for t in range(1, n_days+1):
        z = np.random.normal()
        S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return S

def simulate_gbm_paths(params, S0, n_days, n_sims = 1, dt=1/252, seed=42):
    """
    Simulate multiple paths from a Geometric Brownian Motion (GBM) model.
    params: {'mu':..., 'sigma':...} in daily terms
    S0: Initial stock price
    n_days: Number of days to simulate
    n_sims: Number of simulation paths
    """
    np.random.seed(seed)
    mu, sigma = params['mu'], params['sigma']
    
    S = np.zeros((n_days + 1, n_sims))
    S[0] = S0
    
    for t in range(1, n_days + 1):
        z = np.random.normal(size=n_sims)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return S

def simulate_jump_diffusion(params, S0, n_days, dt=1/252, seed=42):
    """
    Simulates the Merton Jump-Diffusion process using Poisson-distributed jumps.
    params: Dictionary {'mu','sigma','lambda','nu','delta'} (daily values)
    S0: Initial stock price
    n_days: Number of days to simulate
    dt: Time step (default assumes daily steps)
    """
    np.random.seed(seed)
    mu, sigma = params['mu'], params['sigma']
    lam, nu, delta = params['lambda'], params['nu'], params['delta']
    
    S = np.zeros(n_days + 1)
    S[0] = S0
    
    for t in range(1, n_days + 1):

        z = np.random.normal()
        cont_part = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        N = np.random.poisson(lam * dt)  
        
        if N > 0:
            J = np.prod(np.exp(nu + delta * np.random.normal(size=N)) - 1 + 1)  # Product of multiple jumps
        else:
            J = 1
        
        S[t] = S[t - 1] * cont_part * J
    
    return S

def simulate_jump_diffusion_paths(params, S0, n_days, n_sims = 1, dt=1/252, seed=42):
    """
    Simulates multiple paths of the Merton Jump-Diffusion model.
    params: {'mu','sigma','lambda','nu','delta'} (daily values)
    S0: Initial stock price
    n_days: Number of days to simulate
    n_sims: Number of simulation paths
    """
    np.random.seed(seed)
    mu, sigma = params['mu'], params['sigma']
    lam, nu, delta = params['lambda'], params['nu'], params['delta']
    
    S = np.zeros((n_days + 1, n_sims))
    S[0] = S0
    
    for t in range(1, n_days + 1):
        z = np.random.normal(size=n_sims)
        cont_part = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        N = np.random.poisson(lam * dt, size=n_sims)
        J = np.exp(nu + delta * np.random.normal(size=n_sims)) - 1
        jump_part = np.where(N > 0, (1 + J)**N, 1)
        
        S[t] = S[t-1] * cont_part * jump_part
    
    return S


def simulate_heston(params, S0, n_days, dt=1/252, seed=42):
    """
    Heston daily closes (Euler discretization).
    params: {'mu','kappa','theta','sigma_v','rho','v0'}
    S0: initial price
    """
    np.random.seed(seed)
    mu = params['mu']
    kappa, theta, sigma_v, rho, v0 = params['kappa'], params['theta'], params['sigma_v'], params['rho'], params['v0']
    
    S = np.zeros(n_days + 1)
    v = np.zeros(n_days + 1)
    S[0] = S0
    v[0] = v0
    
    for t in range(1, n_days + 1):
        z1 = np.random.normal()
        z2 = np.random.normal()
        
        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        v_prev = max(v[t-1], 1e-8)
        dv = kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * w2
        v_new = max(v_prev + dv, 1e-8)  # Keep variance positive
        
        S_prev = S[t-1]
        S_new = S_prev * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * np.sqrt(dt) * w1)
        
        # Store values
        S[t] = max(S_new, 1e-8)
        v[t] = v_new
    
    return S

def simulate_heston_paths(params, S0, n_days, n_sims = 1, dt=1/252, seed=42):
    """
    Simulates multiple paths using the Heston stochastic volatility model.
    params: {'mu','kappa','theta','sigma_v','rho','v0'}
    S0: Initial stock price
    n_days: Number of days to simulate
    n_sims: Number of simulation paths
    """
    np.random.seed(seed)
    mu, kappa, theta, sigma_v, rho, v0 = params['mu'], params['kappa'], params['theta'], params['sigma_v'], params['rho'], params['v0']
    
    S = np.zeros((n_days + 1, n_sims))
    v = np.zeros((n_days + 1, n_sims))
    S[0] = S0
    v[0] = v0
    
    for t in range(1, n_days + 1):
        z1, z2 = np.random.normal(size=n_sims), np.random.normal(size=n_sims)
        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        v_prev = np.maximum(v[t-1], 1e-8)
        dv = kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * w2
        v_new = np.maximum(v_prev + dv, 1e-8)
        
        S_prev = S[t-1]
        S_new = S_prev * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * np.sqrt(dt) * w1)
        
        S[t] = np.maximum(S_new, 1e-8)
        v[t] = v_new
    
    return S, v


def simulate_bates(params, S0, n_days, dt=1/252, seed=42):
    """
    Bates Model = Heston Stochastic Volatility + Poisson Jump Process.
    Uses Euler discretization for variance and price updates.
    
    params: {'mu', 'kappa', 'theta', 'sigma_v', 'rho', 'v0', 'lambda', 'nu', 'delta'}
    S0: Initial stock price
    n_days: Number of trading days to simulate
    dt: Time step size (default = daily steps)
    """
    np.random.seed(seed)
    
    mu, kappa, theta = params['mu'], params['kappa'], params['theta']
    sigma_v, rho, v0 = params['sigma_v'], params['rho'], params['v0']
    lam, nu, delta = params['lambda'], params['nu'], params['delta']
    
    S = np.zeros(n_days + 1)
    v = np.zeros(n_days + 1)
    S[0] = S0
    v[0] = v0
    
    for t in range(1, n_days + 1):
        z1, z2 = np.random.normal(), np.random.normal()
        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        v_prev = max(v[t-1], 1e-8)  # Ensure variance remains positive
        dv = kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * w2
        v_new = max(v_prev + dv, 1e-8)  # Prevent negative variance
        
        S_prev = S[t-1]
        S_cont = S_prev * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * np.sqrt(dt) * w1)
        
        N = np.random.poisson(lam * dt)  
        if N > 0:
            J = np.prod(np.exp(nu + delta * np.random.normal(size=N)) - 1 + 1)  
        else:
            J = 1 
        
        S[t] = max(S_cont * J, 1e-8)  # Prevent negative prices
        v[t] = v_new  
    
    return S

def simulate_bates_paths(params, S0, n_days, n_sims = 1, dt=1/252, seed=42):
    """
    Bates Model = Heston Stochastic Volatility + Poisson Jump Process.
    Uses Euler discretization for variance and price updates.
    
    params: {'mu', 'kappa', 'theta', 'sigma_v', 'rho', 'v0', 'lambda', 'nu', 'delta'}
    S0: Initial stock price
    n_days: Number of trading days to simulate
    n_sims: Number of paths to simulate
    dt: Time step size (default = daily steps)
    """
    np.random.seed(seed)
    
    mu, kappa, theta = params['mu'], params['kappa'], params['theta']
    sigma_v, rho, v0 = params['sigma_v'], params['rho'], params['v0']
    lam, nu, delta = params['lambda'], params['nu'], params['delta']
    
    S = np.zeros((n_days + 1, n_sims))
    v = np.zeros((n_days + 1, n_sims))
    S[0] = S0
    v[0] = v0
    
    for t in range(1, n_days + 1):
        z1, z2 = np.random.normal(size=n_sims), np.random.normal(size=n_sims)
        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        v_prev = np.maximum(v[t-1], 1e-8)
        dv = kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * w2
        v_new = np.maximum(v_prev + dv, 1e-8)
        
        S_prev = S[t-1]
        S_cont = S_prev * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * np.sqrt(dt) * w1)
        
        N = np.random.poisson(lam * dt, size=n_sims)
        J = np.exp(nu + delta * np.random.normal(size=n_sims)) - 1
        jump_part = np.where(N > 0, (1 + J)**N, 1)
        
        S[t] = np.maximum(S_cont * jump_part, 1e-8)
        v[t] = v_new
    
    return S, v



###############################################################################
#    PART 3: BACKTEST THE ACCURACY OF HISTORICAL FIT (SIMPLE K-S COMPARISON)
###############################################################################
def backtest_historical_fit(df, simulated_prices):
    """
    Compare the distribution of historical log returns vs. simulated log returns 
    using a K-S test.
    df: historical data (with 'Close')
    simulated_prices: array-like or series of simulated closes (same length or can be large).
    
    Returns (ks_stat, ks_pvalue).
    """
    hist_close = df['Close'].dropna()
    hist_logr = np.log(hist_close / hist_close.shift(1)).dropna()
    
    sim_close = pd.Series(simulated_prices)
    sim_logr = np.log(sim_close / sim_close.shift(1)).dropna()
    
    ks_stat, ks_pvalue = kstest(hist_logr, sim_logr)
    return ks_stat, ks_pvalue


def backtest_historical_fit_paths(df, simulated_prices_matrix):
    """
    Compare the distribution of historical log returns vs. simulated log returns 
    using a Kolmogorov-Smirnov (K-S) test.
    
    df: DataFrame with historical 'Close' prices.
    simulated_prices_matrix: 2D array (n_days+1, n_sims) of simulated prices.
    
    Returns:
    - KS Statistic
    - KS p-value
    """
    hist_close = df['Close'].dropna()
    hist_logr = np.log(hist_close / hist_close.shift(1)).dropna()
    
    sim_logr = np.log(simulated_prices_matrix[1:] / simulated_prices_matrix[:-1])
    sim_logr_flat = sim_logr.flatten()  

    ks_stat, ks_pvalue = kstest(hist_logr, sim_logr_flat)
    
    return ks_stat, ks_pvalue


###############################################################################
#     PART 4: BROWNIAN BRIDGE TO GET DAILY HIGH/LOW FROM SIMULATED CLOSES
###############################################################################
def estimate_intraday_sigma(params, model_type="GBM", steps_per_day=24):
    """
    Estimate intraday volatility sigma for the Brownian Bridge based on model parameters.
    
    params: Dictionary of model parameters.
    model_type: One of ["GBM", "Jump-Diffusion", "Heston", "Bates"]
    steps_per_day: Number of intraday time steps (default: 24 steps = 1 hour intervals)
    
    Returns: Estimated sigma_intraday
    """
    if model_type == "GBM":
        sigma_daily = params["sigma"]
    
    elif model_type == "Jump-Diffusion":
        sigma = params["sigma"]
        lam = params["lambda"]
        nu = params["nu"]
        delta = params["delta"]
        sigma_daily = np.sqrt(sigma**2 + lam * (nu**2 + delta**2))
    
    elif model_type == "Heston":
        theta = params["theta"]
        sigma_daily = np.sqrt(theta)
    
    elif model_type == "Bates":
        theta = params["theta"]
        lam = params["lambda"]
        nu = params["nu"]
        delta = params["delta"]
        sigma_daily = np.sqrt(theta + lam * (nu**2 + delta**2))
    
    else:
        raise ValueError("Invalid model_type. Choose from 'GBM', 'Jump-Diffusion', 'Heston', or 'Bates'.")

    # Compute intraday sigma
    sigma_intraday = sigma_daily / np.sqrt(steps_per_day)
    
    return sigma_intraday


def brownian_bridge_daily(sim_closes, steps_per_day=24, seed=42):
    """
    For each day (from sim_closes[t-1] to sim_closes[t]), 
    construct an intraday path (brownian-bridge style) to get the daily High/Low.
    
    Returns a DataFrame with columns: 'Open','High','Low','Close'
    and the same length as len(sim_closes)-1 for each "day".
    """
    np.random.seed(seed)
    n_days = len(sim_closes) - 1
    
    records = []
    
    for i in range(n_days):
        open_price = sim_closes[i]
        close_price = sim_closes[i+1]
        
        # We'll create steps_per_day - 1 intraday points bridging
        path = np.zeros(steps_per_day+1)
        path[0] = open_price
        path[steps_per_day] = close_price
        
        for step in range(1, steps_per_day):
            # naive approach
            # linear interpolation as the drift of the "bridge"
            fraction = step / steps_per_day
            drift = path[step-1] + (close_price - path[step-1]) / (steps_per_day - step + 1)
            
            # random shock scaled by some fraction
            scale = 0.01 * path[step-1]  # can be tuned or linked to volatility
            path[step] = drift + np.random.normal(0, scale)
        
        day_high = path.max()
        day_low = path.min()
        
        records.append({
            'Open': path[0],
            'High': day_high,
            'Low': day_low,
            'Close': path[-1]
        })
    
    df_bridge = pd.DataFrame(records)
    return df_bridge

import numpy as np
import pandas as pd

def brownian_bridge_paths(sim_closes_matrix, steps_per_day=24, sigma_intraday=0.02, seed=42):
    """
    Simulate independent intraday paths using a Brownian Bridge approach.
    
    sim_closes_matrix: (n_days+1, n_sims) matrix of simulated daily close prices.
    steps_per_day: Number of intraday steps for the Brownian bridge.
    sigma_intraday: Estimated intraday volatility.
    
    Returns:
    - A dictionary of DataFrames, one per simulation, with columns: 'Open', 'High', 'Low', 'Close'.
    """
    np.random.seed(seed)
    
    n_days, n_sims = sim_closes_matrix.shape[0] - 1, sim_closes_matrix.shape[1]
    
    # Store DataFrames for each simulation separately
    sim_dataframes = {}

    for sim in range(n_sims):
        open_prices = sim_closes_matrix[:-1, sim]
        close_prices = sim_closes_matrix[1:, sim]

        high_prices = np.zeros(n_days)
        low_prices = np.zeros(n_days)
        
        for i in range(n_days):
            open_day = open_prices[i]
            close_day = close_prices[i]
            
            # Generate time grid for intraday steps
            t = np.linspace(0, 1, steps_per_day+1)
            
            # Brownian bridge formula: drift + volatility term
            drift = (1 - t) * open_day + t * close_day
            volatility = sigma_intraday * np.sqrt(t * (1 - t)) * np.random.randn(steps_per_day+1)

            bridge_path = drift + volatility
            
            # Store intraday high/low for the day
            high_prices[i] = bridge_path.max()
            low_prices[i] = bridge_path.min()

        # Create a DataFrame for this simulation
        df_bridge = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices
        })
        
        sim_dataframes[sim] = df_bridge  # Store each simulation separately
    
    return sim_dataframes


###############################################################################
#  PART 5: BACKTEST TRADER USING TRENDER COLUMNS
###############################################################################

def ewm_std(series, span):
    ewm_mean   = series.ewm(span=span, adjust=False).mean()
    ewm_meansq = (series**2).ewm(span=span, adjust=False).mean()
    return np.sqrt(ewm_meansq - ewm_mean**2)

def trender_bloomberg_style(df, period=14, sensitivity=1, use_close=True):
    df = df.copy()
    df.sort_index(inplace=True)

    high = df['High']
    low  = df['Low']
    close = df['Close']

    prev_close = close.shift(1)
    tr = pd.DataFrame({
        'r1': high - low,
        'r2': (high - prev_close).abs(),
        'r3': (low  - prev_close).abs()
    }).max(axis=1)

    mp = 0.5 * (high + low)

    ema_mp = mp.ewm(span=period, adjust=False).mean()
    ema_tr = tr.ewm(span=period, adjust=False).mean()

    sd_ema_tr = ewm_std(ema_tr, period)

    up_base   = ema_mp - 0.5 * ema_tr - sensitivity * sd_ema_tr
    down_base = ema_mp + 0.5 * ema_tr + sensitivity * sd_ema_tr

    n = len(df)
    up_line   = np.full(n, np.nan)
    down_line = np.full(n, np.nan)
    trend     = np.full(n, 0, dtype=int)

    # Pick an initial day where stdev is not NaN:
    first_valid = sd_ema_tr.first_valid_index()
    if first_valid is None:
        return pd.DataFrame(
            {'TrenderUp': up_line, 'TrenderDown': down_line, 'Trend': trend},
            index=df.index
        )
    start_i = df.index.get_loc(first_valid)

    price_series = close if use_close else high
    # Set initial trend
    if price_series.iloc[start_i] > up_base.iloc[start_i]:
        trend[start_i] = +1
        up_line[start_i] = up_base.iloc[start_i]
    else:
        trend[start_i] = -1
        down_line[start_i] = down_base.iloc[start_i]

    for i in range(start_i, n-1):
        if trend[i] == +1:
            if i > start_i:
                up_line[i] = max(up_base.iloc[i], up_line[i-1])
            if price_series.iloc[i+1] < up_line[i]:
                trend[i+1] = -1
                down_line[i+1] = down_base.iloc[i+1]  # new downtrend stop
            else:
                trend[i+1] = +1
        else:
            if i > start_i:
                down_line[i] = min(down_base.iloc[i], down_line[i-1])
            if price_series.iloc[i+1] > down_line[i]:
                trend[i+1] = +1
                up_line[i+1] = up_base.iloc[i+1]  # new uptrend stop
            else:
                trend[i+1] = -1

    if trend[n-1] == +1:
        up_line[n-1] = max(up_base.iloc[n-1], up_line[n-2])
    else:
        down_line[n-1] = min(down_base.iloc[n-1], down_line[n-2])

    for i in range(n):
        if trend[i] == +1:
            down_line[i] = np.nan
        else:
            up_line[i] = np.nan

    return pd.DataFrame({
        'TrenderUp': up_line,
        'TrenderDown': down_line,
        'Trend': trend,
        'ATR': ema_tr
    }, index=df.index)



def backtest_trender_strategy(df, initial_capital=100000.0, mode='long_only', 
                              annual_factor=252):
    import numpy as np
    import pandas as pd

    closes = df['Close'].values
    trender_down = df['TrenderDown'].values
    trender_up = df['TrenderUp'].values
    n = len(df)

    signal_up = np.empty_like(trender_up)
    signal_down = np.empty_like(trender_down)
    signal_up[0] = np.nan
    signal_down[0] = np.nan
    signal_up[1:] = trender_up[:-1]
    signal_down[1:] = trender_down[:-1]

    position = 0
    positions = np.zeros(n, dtype=int)

    capital = initial_capital
    equity_curve = np.zeros(n)


    trades = []

    equity_curve[0] = capital
    positions[0] = position

    for i in range(1, n):
        price = closes[i]

        if mode == 'long_only':
            if position == 0 and price > signal_up[i]:
                position = 1
                trades.append({
                    'side': 'buy',
                    'price': price,
                    'index': df.index[i],
                    'entry_capital': capital
                })
            elif position == 1 and price < signal_down[i]:
                position = 0
                trades.append({
                    'side': 'sell',
                    'price': price,
                    'index': df.index[i],
                    'exit_capital': capital
                })

        elif mode == 'long_short':
            if price > signal_up[i]:
                if position != 1:
                    if position == -1:
                        trades.append({
                            'side': 'cover',
                            'price': price,
                            'index': df.index[i],
                            'exit_capital': capital
                        })
                    trades.append({
                        'side': 'buy',
                        'price': price,
                        'index': df.index[i],
                        'entry_capital': capital
                    })
                    position = 1
            elif price < signal_down[i]:
                if position != -1:
                    if position == 1:
                        trades.append({
                            'side': 'sell',
                            'price': price,
                            'index': df.index[i],
                            'exit_capital': capital
                        })
                    trades.append({
                        'side': 'short',
                        'price': price,
                        'index': df.index[i],
                        'entry_capital': capital
                    })
                    position = -1

        daily_ret = (price - closes[i - 1]) / closes[i - 1]
        if position == 1:
            capital *= (1 + daily_ret)
        elif position == -1:
            capital *= (1 - daily_ret)

        equity_curve[i] = capital
        positions[i] = position

    total_return = (equity_curve[-1] / equity_curve[0]) - 1.0
    eq_daily_ret = pd.Series(np.diff(equity_curve) / equity_curve[:-1], index=df.index[1:])
    mean_daily_ret = eq_daily_ret.mean()
    std_daily_ret = eq_daily_ret.std()

    annual_vol = std_daily_ret * np.sqrt(annual_factor)
    if not np.isnan(mean_daily_ret):
        annual_return = (1 + mean_daily_ret)**annual_factor - 1
    else:
        annual_return = np.nan

    sharpe_ratio = (annual_return / annual_vol) if annual_vol != 0 else np.nan

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = drawdowns.min()

    dd_durations = []
    peak_idx = 0
    for i in range(1, n):
        if equity_curve[i] >= running_max[i - 1]:
            peak_idx = i
        dd_durations.append(i - peak_idx)
    max_dd_duration = max(dd_durations) if dd_durations else 0

    realized_pnl = []
    open_trade = None

    for trd in trades:
        side = trd['side']


        if side in ['buy', 'short']:
            open_trade = trd

        elif side in ['sell', 'cover'] and open_trade is not None:

            if (open_trade['side'] == 'buy' and side == 'sell') or \
               (open_trade['side'] == 'short' and side == 'cover'):
                entry_cap = open_trade.get('entry_capital', np.nan)
                exit_cap = trd.get('exit_capital', np.nan)
                pnl = exit_cap - entry_cap
            else:
                pnl = 0.0

            realized_pnl.append(pnl)
            open_trade = None


    if realized_pnl:
        wins = sum(1 for pnl in realized_pnl if pnl > 0)
        losses = sum(1 for pnl in realized_pnl if pnl <= 0)
        win_ratio = wins / (wins + losses) if (wins + losses) else np.nan
    else:
        win_ratio = np.nan

    num_trades = len(realized_pnl)

    results = {
        'final_equity': equity_curve[-1],
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        'max_drawdown_duration': max_dd_duration,
        'num_trades': num_trades,
        'win_ratio': win_ratio
    }

    return (
        results,
        pd.Series(equity_curve, index=df.index, name='Equity'),
        pd.Series(positions, index=df.index, name='Position')
    )



def apply_trender_to_simulations(simulated_prices, sensitivity=1):
    """
    Apply the Trender indicator to multiple simulated price paths.
    
    simulated_prices: dict of DataFrames containing 'Open', 'High', 'Low', 'Close' for each simulation.
    
    Returns:
    - dict of DataFrames containing 'TrenderUp', 'TrenderDown', 'Trend' for each simulation.
    """
    trender_results = {}
    
    for sim_id, df in simulated_prices.items():
        trender_results[sim_id] = trender_bloomberg_style(df, sensitivity)
    
    return trender_results

def backtest_trender_multiple_paths(simulated_prices, trender_results, mode='long_only', initial_capital=100000.0):
    """
    Backtest the Trender strategy across multiple simulated price paths.
    
    Parameters:
    - simulated_prices: dict of simulated OHLC price DataFrames.
    - trender_results: dict of Trender indicators for each simulation.
    - mode: 'long_only' or 'long_short'.
    - initial_capital: Starting capital for backtest.
    
    Returns:
    - Dictionary of performance metrics aggregated across all paths.
    - Dictionary of equity curves for each path.
    """
    performance_metrics = []
    equity_curves = {}
    positions = {}

    for sim_id in simulated_prices.keys():
        df = simulated_prices[sim_id].copy()
        df[['TrenderUp', 'TrenderDown', 'Trend']] = trender_results[sim_id][['TrenderUp', 'TrenderDown', 'Trend']]

        # Backtest strategy
        results, equity_curve, position = backtest_trender_strategy(df, initial_capital, mode)
        
        performance_metrics.append(results)
        equity_curves[sim_id] = equity_curve
        positions[sim_id] = position

    # Convert results into a DataFrame
    df_results = pd.DataFrame(performance_metrics)

    # Aggregate results
    aggregated_results = {
        "mean": df_results.mean(),
        "std": df_results.std(),
        "min": df_results.min(),
        "max": df_results.max(),
    }

    return aggregated_results, df_results, equity_curves, positions

def select_best_model(backtest_results):
    """
    Selects the best model based on KS-statistic.
    If all p-values are 0, selects the model with the lowest KS-stat.
    Prefers more complex models (Bates > Heston > Jump-Diffusion > GBM).
    """
    # Sort models by KS statistic (lower is better)
    ranked_models = backtest_results.T.sort_values(by="KS_Stat")
    
    # Select the model with the lowest KS-Stat
    best_model = ranked_models.index[0]
    
    # Ensure we prioritize more advanced models if KS stats are close
    if best_model == "GBM" and ranked_models.iloc[1]["KS_Stat"] < ranked_models.iloc[0]["KS_Stat"] * 1.05:
        best_model = ranked_models.index[1]  # Prefer Jump-Diffusion
    if best_model in ["GBM", "Jump-Diffusion"] and ranked_models.iloc[2]["KS_Stat"] < ranked_models.iloc[0]["KS_Stat"] * 1.05:
        best_model = ranked_models.index[2]  # Prefer Heston
    if best_model in ["GBM", "Jump-Diffusion", "Heston"] and ranked_models.iloc[3]["KS_Stat"] < ranked_models.iloc[0]["KS_Stat"] * 1.05:
        best_model = ranked_models.index[3]  # Prefer Bates

    return best_model
def backtest_buy_and_hold_strategy(
    df, 
    initial_capital=100000.0, 
    annual_factor=252
):
    import numpy as np
    import pandas as pd

    closes = df['Close'].values
    n = len(df)

    capital = initial_capital
    equity_curve = np.zeros(n)
    equity_curve[0] = capital

    # Fully invested, so just compound each day
    for i in range(1, n):
        daily_ret = (closes[i] - closes[i - 1]) / closes[i - 1]
        capital *= (1 + daily_ret)
        equity_curve[i] = capital

    # -- Performance metrics (same logic) --
    total_return = (equity_curve[-1] / equity_curve[0]) - 1.0
    eq_daily_ret = pd.Series(np.diff(equity_curve) / equity_curve[:-1], index=df.index[1:])
    mean_daily_ret = eq_daily_ret.mean()
    std_daily_ret = eq_daily_ret.std()

    annual_vol = std_daily_ret * np.sqrt(annual_factor)
    if not np.isnan(mean_daily_ret):
        annual_return = (1 + mean_daily_ret)**annual_factor - 1
    else:
        annual_return = np.nan

    sharpe_ratio = (annual_return / annual_vol) if annual_vol != 0 else np.nan

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = drawdowns.min()

    # max drawdown duration
    dd_durations = []
    peak_idx = 0
    for i in range(1, n):
        if equity_curve[i] >= running_max[i - 1]:
            peak_idx = i
        dd_durations.append(i - peak_idx)
    max_dd_duration = max(dd_durations) if dd_durations else 0

    results = {
        'final_equity': equity_curve[-1],
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        'max_drawdown_duration': max_dd_duration,
        # For consistency with your existing columns:
        'num_trades': 1,
        'win_ratio': np.nan,  # not really applicable to buy-and-hold
    }

    return (
        results,
        pd.Series(equity_curve, index=df.index, name='Equity')
    )
def backtest_buy_and_hold_multiple_paths(
    simulated_prices,
    initial_capital=100000.0
):
    import pandas as pd

    performance_metrics = []
    equity_curves = {}

    for sim_id, df_sim in simulated_prices.items():
        # Single-path buy & hold backtest
        results, eq_curve = backtest_buy_and_hold_strategy(
            df_sim, 
            initial_capital=initial_capital
        )
        performance_metrics.append(results)
        equity_curves[sim_id] = eq_curve

    # Convert results into a DataFrame
    df_results = pd.DataFrame(performance_metrics)

    # Just like your "aggregated_results" approach
    aggregated_results = {
        "mean": df_results.mean(),
        "std":  df_results.std(),
        "min":  df_results.min(),
        "max":  df_results.max(),
    }

    return aggregated_results, df_results, equity_curves
