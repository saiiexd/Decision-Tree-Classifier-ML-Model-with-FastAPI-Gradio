import pandas as pd
import numpy as np

def generate_data(n_samples=250):
    """
    Generates a synthetic dataset for student pass/fail prediction.
    Includes non-linear boundaries and noise for realistic training.
    """
    np.random.seed(42)
    
    # Features
    study_hours = np.random.uniform(0, 12, n_samples)
    attendance = np.random.uniform(0, 100, n_samples)
    previous_score = np.random.uniform(0, 100, n_samples)
    
    # Complex non-linear logic
    # Interaction between study hours and attendance is key
    # Even if you study many hours, low attendance might hurt you (and vice-versa)
    score = (
        (study_hours * 5) + 
        (attendance * 0.4) + 
        (previous_score * 0.2) +
        (study_hours * attendance * 0.1) # Interaction term
    )
    
    # Threshold with some sigmoid-like probability
    # Base pass/fail based on score > 85
    prob = 1 / (1 + np.exp(-(score - 85) / 10))
    pass_status = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_score': previous_score,
        'pass_status': pass_status
    })
    
    return df

if __name__ == "__main__":
    df = generate_data()
    print(f"Generated {len(df)} rows.")
    print(df.head())
    df.to_csv("student_data.csv", index=False)
