import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from a CSV file.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data by encoding categorical features and separating features and target.
    :param df: The DataFrame containing the dataset.
    :return: X (features) and y (target).
    """
    # Example: Encoding 'traffic_level' as a categorical feature
    df['traffic_level'] = df['traffic_level'].astype('category').cat.codes  # Encoding categorical features
    
    # Define the features (X) and target (y)
    X = df.drop('delivery_time_minutes', axis=1)  # All columns except the target
    y = df['delivery_time_minutes']  # Target variable

    return X, y

def split_data(X, y, test_size=0.2):
    """
    Split the data into training and testing sets.
    :param X: Features
    :param y: Target
    :param test_size: Proportion of the data to be used as test set.
    :return: Split data (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)
