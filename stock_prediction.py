import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np


stock = yf.download("AAPL", start="1982-12-12", end="2025-01-01")


if stock.empty:
    print("Error: No stock data found. Check stock symbol or internet connection.")
    exit()


print(stock.head())


plt.figure(figsize=(10, 5))
plt.plot(stock["Close"], label="AAPL Closing Price", color='blue')
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.title("Apple Stock Price Over Time")
plt.legend()
plt.show()


stock['Prediction'] = stock['Close'].shift(-30)  # Shift 'Close' price 30 days ahead


data = stock[['Close', 'Prediction']].dropna()


if data.empty:
    print("Error: Not enough data for prediction after shifting. Try a smaller shift value.")
    exit()

X = data[['Close']]
y = data['Prediction']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Data:", X_train.shape, "Testing Data:", X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")


last_price = np.array(stock["Close"].iloc[-1]).reshape(-1, 1)
future_prediction = model.predict(last_price)[0]

print(f"Predicted Stock Price for the next 30 days: ${future_prediction:.2f}")
