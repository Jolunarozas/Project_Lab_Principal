# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:44:13 2025

@author: wangj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

df = pd.read_excel('GOLDTRNDR.xlsx', sheet_name='Sheet1')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date', ascending=True).set_index('Date')

df_filtered = df.loc['2024-03-04':'2025-01-24']

closes = df_filtered['Close'].values
log_returns = np.diff(np.log(closes))
daily_var = np.var(log_returns)
annualized_var = daily_var * 252  

# Heston parameters
S0 = closes[0]        # Initial price (2024-03-04)
V0 = annualized_var   # Initial variance
mu = np.mean(log_returns) * 252  # Annualized drift
kappa = 2             # Mean reversion speed
theta = V0            # Long-term variance
xi = 0.2              # Volatility of volatility
rho = -0.7            # Correlation
n = len(df_filtered) - 1  # Number of steps
T = n / 252           # Time horizon (annualized)
dt = T / n

# Simulate correlated Brownian motions
np.random.seed(123)
Z1 = np.random.normal(0, 1, n)
Z2 = np.random.normal(0, 1, n)
Z2_corr = rho * Z1 + np.sqrt(1 - rho**2) * Z2

# Simulate Heston paths (ensure S has same length as DataFrame)
S = np.zeros(n + 1)  # n+1 elements to match DataFrame length to avoid an awkward error
V = np.zeros(n + 1)
S[0] = S0
V[0] = V0

for i in range(1, n + 1):
    V_prev = max(V[i-1], 0)
    dV = kappa * (theta - V_prev) * dt + xi * np.sqrt(V_prev) * np.sqrt(dt) * Z2_corr[i-1]
    V[i] = max(V_prev + dV, 0)
    dS = mu * S[i-1] * dt + np.sqrt(V_prev) * S[i-1] * np.sqrt(dt) * Z1[i-1]
    S[i] = S[i-1] + dS

# Add simulated prices to DataFrame (align lengths)
df_sim = df_filtered.copy()
df_sim['Simulated'] = S  # Use full S array (n+1 elements)

# Plot candlestick and simulated prices
apd = [mpf.make_addplot(df_sim['Simulated'], color='blue', panel=0)]

mpf.plot(
    df_sim,
    type='candle',
    addplot=apd,
    style='charles',
    title='Gold Price: Actual vs Heston Simulation (2024-03-04 to 2025-01-24)',
    ylabel='Price',
    figratio=(12, 6)
)