
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = {
    'Size (sq ft)': [800, 1000, 1200, 1500, 1800],
    'Price ($)': [150000, 180000, 200000, 240000, 275000]
}

df = pd.DataFrame(data)


X = df[['Size (sq ft)']]   # house sizes
y = df['Price ($)']        # house prices


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


mse = mean_squared_error(y, y_pred)
print("Actual Prices:", list(y))
print("Predicted Prices:", list(y_pred))
print("Mean Squared Error:", mse)


new_house_size = [[2000]]
predicted_price = model.predict(new_house_size)
print(f"\nPredicted Price for 2000 sq ft house: ${predicted_price[0]:.2f}")


plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, y_pred, color='red', label='Prediction Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($)')
plt.title('🏡 House Price Prediction')
plt.legend()
plt.show()


