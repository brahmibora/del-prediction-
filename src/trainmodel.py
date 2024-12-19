from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model(X_train, y_train):
    """Train the model using Random Forest."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Load the trained model from a file."""
    return joblib.load(file_path)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
