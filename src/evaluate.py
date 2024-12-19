from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the trained model.
    :param model: The trained machine learning model.
    :param X_test: The features of the test dataset.
    :param y_test: The true labels of the test dataset.
    :return: Dictionary containing evaluation metrics.
    """
    # Predicting the target values using the test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    
    # Return the metrics in a dictionary
    metrics = {
        'Mean Squared Error': mse,
        'R-squared': r2
    }
    
    return metrics

def save_model(model, filename):
    """
    Save the trained model to a file.
    :param model: The trained model.
    :param filename: The file name to save the model.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Load a model from a file.
    :param filename: The file name of the saved model.
    :return: The loaded model.
    """
    model = joblib.load(filename)
    return model
