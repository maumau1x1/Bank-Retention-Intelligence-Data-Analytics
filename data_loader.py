import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    # Drop columns that have zero predictive power
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=drop_cols)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Generate EDA Visualizations
    os.makedirs('plots', exist_ok=True)
    
    # 1. Churn Rate by Geography
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Geography', hue='Exited', data=df)
    plt.title('Churn Count by Geography')
    plt.savefig('plots/churn_by_geography.png')
    plt.close()
    
    # 2. Correlation Heatmap (numeric columns only)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    print("EDA Visualizations saved in 'plots' directory.")
    
    return df