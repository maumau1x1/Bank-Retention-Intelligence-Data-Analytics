import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Encode Gender (Binary)
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    # One-Hot Encode Geography (Categorical)
    if 'Geography' in df.columns:
        df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    
    # Feature Engineering: Balance-to-Salary Ratio
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    
    # Feature 2: Tenure to Age Ratio
    df['Tenure_Age_Ratio'] = df['Tenure'] / df['Age']
    
    # Feature 3: Active Member with Credit Card
    df['Active_With_Card'] = df['IsActiveMember'] * df['HasCrCard']
    
    # Feature 4: Age Groups
    bins_age = [0, 30, 45, 60, 100]
    labels_age = ['Young Adult', 'Adult', 'Senior', 'Elderly']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age)
    df = pd.get_dummies(df, columns=['Age_Group'], drop_first=True)
    
    # Feature 5: Credit Score Range
    bins_cs = [0, 580, 670, 740, 800, 900]
    labels_cs = ['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
    df['Credit_Score_Range'] = pd.cut(df['CreditScore'], bins=bins_cs, labels=labels_cs)
    df = pd.get_dummies(df, columns=['Credit_Score_Range'], drop_first=True)
    
    return df