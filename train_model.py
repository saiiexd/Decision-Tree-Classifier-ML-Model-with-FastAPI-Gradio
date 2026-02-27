import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import data_generation
import os

def train():
    # 1. Generate Data
    df = data_generation.generate_data(300)
    
    X = df[['study_hours', 'attendance', 'previous_score']]
    y = df['pass_status']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Decision Tree Tuning (max_depth 3-5)
    model = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=5)
    model.fit(X_train, y_train)
    
    # 4. Accuracy Check
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Training ---")
    print(f"Accuracy: {acc:.2f}")
    
    # 5. Feature Importance
    importances = model.feature_importances_
    features = X.columns
    feat_importance_dict = dict(zip(features, importances))
    print("Feature Importances:")
    for feat, imp in feat_importance_dict.items():
        print(f" - {feat}: {imp:.4f}")
    
    # 6. Save Model and Metadata
    metadata = {
        "feature_importance": feat_importance_dict,
        "accuracy": acc
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    # 7. Visualize Tree
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=features, class_names=['Fail', 'Pass'], filled=True, rounded=True)
    plt.title("Decision Tree Visualization")
    plt.savefig("tree_plot.png")
    plt.close()
    print("Decision tree plot saved as tree_plot.png")

if __name__ == "__main__":
    train()
