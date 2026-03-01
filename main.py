from data_loader import load_and_clean
from engineering import preprocess_data
from model import train_and_evaluate

def run_pipeline():
    # 1. Load
    data = load_and_clean('Churn_Modelling.csv')
    
    # 2. Engineer
    processed_data = preprocess_data(data)
    
    # 3. Train
    model, X_test, y_test = train_and_evaluate(processed_data)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()