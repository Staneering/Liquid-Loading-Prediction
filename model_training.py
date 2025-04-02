from sklearn.linear_model import LinearRegression
import joblib

# Load preprocessed data
from data_cleaning_preprocessing import preprocessed_data

X_train_scaled = preprocessed_data["X_train_scaled"]
y_train = preprocessed_data["y_train"]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, "regression_model.pkl")
