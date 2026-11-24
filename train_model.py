import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import os

def train():
    print("Loading training dataset...")
    data_path = os.path.join("datasets", "Training.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Create sorted disease mapping
    diseases = sorted(df['prognosis'].unique())
    disease_to_index = {d: i for i, d in enumerate(diseases)}

    X = df.drop('prognosis', axis=1)
    y = df['prognosis'].map(disease_to_index)

    print("Training Support Vector Classifier (SVC)...")
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    predictions = clf.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"Training Accuracy: {acc * 100:.2f}%")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "svc.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model successfully saved to {model_path}!")

if __name__ == "__main__":
    train()
