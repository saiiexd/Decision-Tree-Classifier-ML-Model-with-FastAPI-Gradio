import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def generate_and_train():
    # Generate simple synthetic data
    # Features: Age (18-60), EstimatedSalary (20000 - 150000)
    # Target: Purchased (0 or 1)
    np.random.seed(42)
    n_samples = 200
    age = np.random.randint(18, 61, n_samples)
    salary = np.random.randint(20000, 150001, n_samples)
    
    # Simple logic: If age > 35 and salary > 50000, higher chance of buying
    purchased = ((age > 35) & (salary > 50000)).astype(int)
    # Add some noise
    noise = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    purchased = np.logical_xor(purchased, noise).astype(int)

    df = pd.DataFrame({
        'Age': age,
        'EstimatedSalary': salary,
        'Purchased': purchased
    })

    X = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'decision_tree_model.joblib')
    print("Model trained and saved as decision_tree_model.joblib")
    
    return model

def predict(age, salary):
    if not os.path.exists('decision_tree_model.joblib'):
        generate_and_train()
    
    model = joblib.load('decision_tree_model.joblib')
    prediction = model.predict([[age, salary]])
    return int(prediction[0])

if __name__ == "__main__":
    generate_and_train()
