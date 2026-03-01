from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def train_and_evaluate(df):
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Using scale_pos_weight to handle the 20/80 churn imbalance
    model = XGBClassifier(scale_pos_weight=4, random_state=42) 
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("--- Model Performance ---")
    print(classification_report(y_test, y_pred))
    
    # Export the result output into a new csv file
    results_df = X_test.copy()
    results_df['Actual_Exited'] = y_test
    results_df['Predicted_Exited'] = y_pred
    
    output_filename = 'churn_predictions_output.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"Results exported successfully to {output_filename}")
    
    return model, X_test, y_test