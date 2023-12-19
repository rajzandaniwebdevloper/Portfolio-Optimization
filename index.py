# @title Default title text
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import cvxpy as cp

# Function to fetch stock data
def get_stock_data(stock_symbols, start_date, end_date):
    data = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']
    return data

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(weights, returns, risk_free_rate):
    portfolio_returns = np.dot(returns, weights)
    average_daily_return = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    sharpe_ratio = (average_daily_return - risk_free_rate) / volatility
    return average_daily_return, volatility, sharpe_ratio

# Function for portfolio optimization
def optimize_portfolio(returns, risk_free_rate, target_return=None):
    num_assets = returns.shape[1]

    # Define optimization variables
    weights = cp.Variable(num_assets)

    # Objective function (maximize Sharpe ratio or target return)
    if target_return is None:
        objective = cp.Maximize((cp.sum(returns.values @ weights) - risk_free_rate) / cp.std(returns.values @ weights))
    else:
        objective = cp.Maximize(cp.sum(returns.values @ weights))

    # Constraints
    constraints = [cp.sum(weights) == 1, weights >= 0]

    # Create and solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract optimized weights
    optimized_weights = weights.value

    return optimized_weights.flatten()

    # Constraints
    constraints = [cp.sum(weights) == 1, weights >= 0]

    # Create and solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract optimized weights
    optimized_weights = weights.value

    return optimized_weights.flatten()
    
    # Constraints
    constraints = [cp.sum(weights) == 1, weights >= 0]

    # Create and solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract optimized weights
    optimized_weights = weights.value

    return optimized_weights

# Set a seed for reproducibility
np.random.seed(42)

# Step 1: Fetch actual historical stock prices of Indian stocks
stocks = ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO"]  # Example stocks (Reliance Industries, TCS, HDFC Bank)

end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

data = get_stock_data(stocks, start_date, end_date)

# Step 2: Data Preprocessing
returns = data.pct_change().dropna()

# Step 3: Portfolio Calculation
weights = np.array([0.4, 0.3, 0.3])  # Example weights for three stocks

# Step 4: Performance Metrics
risk_free_rate = 0.01
average_daily_return, volatility, sharpe_ratio = calculate_portfolio_metrics(weights, returns, risk_free_rate)

# Step 5: Risk Assessment
cov_matrix = np.cov(returns.values, rowvar=False)

# Step 6: Visualization
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(data.index[:-1], np.cumprod(1 + returns @ weights) - 1, label='Cumulative Returns')
plt.title('Portfolio Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()

plt.subplot(2, 1, 2)
plt.bar(['Average Daily Return', 'Volatility', 'Sharpe Ratio'], [average_daily_return, volatility, sharpe_ratio])
plt.title('Portfolio Metrics')
plt.ylabel('Metric Value')

plt.tight_layout()
plt.show()

# Step 7: Optimization with cvxpy
# Target a specific return
target_return = 0.002

# Run optimization
optimized_weights = optimize_portfolio(returns, risk_free_rate, target_return)

# Print the results
print("\nReal-world Scenario (Indian Stocks):")
print("Optimized Weights:", optimized_weights)
print("Optimized Portfolio Metrics:")
print("   - Expected Return:", np.sum(returns @ optimized_weights))
print("   - Volatility:", np.std(returns @ optimized_weights))
print("   - Sharpe Ratio:", (np.sum(returns @ optimized_weights) - risk_free_rate) / np.std(returns @ optimized_weights))
