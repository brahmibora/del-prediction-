from src.preprocess import load_data, preprocess_data, split_data
from src.trainmodel import train_model, save_model, load_model, evaluate_model

def main():
    # Load the data
    df = load_data('data/dataset.csv')
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the model
    save_model(model, 'saved models/delivery_time_model.pkl')
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
