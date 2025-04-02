from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load preprocessed data and model
from data_cleaning_preprocessing import preprocessed_data
import joblib

X_test_scaled = preprocessed_data["X_test_scaled"]
y_test = preprocessed_data["y_test"]
model = joblib.load("regression_model.pkl")

# Make predictions
y_pred_test = model.predict(X_test_scaled)

# Evaluate the model
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")

# Visualize predictions vs actual values
plt.scatter(y_test, y_pred_test, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid()
plt.show()
