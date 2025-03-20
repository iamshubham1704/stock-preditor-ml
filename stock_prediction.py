import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Step 1: User Input for Stock Symbol
ticker = input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT): ").strip().upper()

# Step 2: Fetch Stock Data
stock = yf.download(ticker, start="2010-01-01", end="2025-01-01")

# Step 3: Error Handling for Invalid Stocks
if stock.empty:
    print("Error: No stock data found. Check stock symbol or internet connection.")
    exit()

print(stock.head())  # Print first few rows

# Step 4: Plot Historical Stock Prices
plt.figure(figsize=(12, 6))
plt.plot(stock["Close"], label=f"{ticker} Closing Price", color='blue')
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.title(f"{ticker} Stock Price Over Time")
plt.legend()
plt.show()

# Step 5: Feature Engineering
stock['SMA_10'] = stock['Close'].rolling(window=10).mean()
stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
stock['Prediction'] = stock['Close'].shift(-30)  # Predict 30 days ahead

# Step 6: Prepare Data
data = stock[['Close', 'SMA_10', 'SMA_50', 'Prediction']].dropna()
X = data[['Close', 'SMA_10', 'SMA_50']]
y = data['Prediction']

# Step 7: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Data:", X_train.shape, "Testing Data:", X_test.shape)

# Step 8: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"{ticker} - Mean Absolute Error: {mae:.2f}")

# Step 10: Predict Future Prices
last_data = np.array(stock[['Close', 'SMA_10', 'SMA_50']].iloc[-1]).reshape(1, -1)
future_predictions = []

for _ in range(30):  # Predict for next 30 days
    future_price = model.predict(last_data)[0]
    future_predictions.append(future_price)
    last_data = np.roll(last_data, -1)  # Shift data
    last_data[0, -1] = future_price  # Update last value with prediction

# Step 11: Plot Future Predictions
plt.figure(figsize=(10, 5))
plt.plot(range(1, 31), future_predictions, marker='o', linestyle='-', color='red', label="Predicted Prices")
plt.xlabel("Days in Future")
plt.ylabel("Stock Price ($)")
plt.title(f"Predicted Stock Prices for {ticker} (Next 30 Days)")
plt.legend()
plt.show()

# Step 12: Print Predictions
print(f"\nPredicted Stock Prices for {ticker} for the next 30 days:")
for i, price in enumerate(future_predictions, start=1):
    print(f"Day {i}: ${price:.2f}")
