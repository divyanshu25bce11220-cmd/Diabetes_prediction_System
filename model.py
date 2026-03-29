# ================================
# Advanced Disease Prediction Model
# ================================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv("diabetes.csv", header=None)

data.columns = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"
]

# -------------------------------
# CLEAN DATA
# -------------------------------
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    data[col] = data[col].replace(0, data[col].mean())

# -------------------------------
# SPLIT DATA
# -------------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# SCALING (IMPORTANT FOR ACCURACY)
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# TRAIN MULTIPLE MODELS
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC()
}

accuracies = {}
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.2f}")

    if acc > best_score:
        best_score = acc
        best_model = model
        best_pred = pred

print("\nBest Model Selected ✅")

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_test, best_pred)

# -------------------------------
# SAVE EVERYTHING
# -------------------------------
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(accuracies, open("accuracies.pkl", "wb"))
pickle.dump(cm, open("cm.pkl", "wb"))

print("\nFiles saved: model.pkl, scaler.pkl, accuracies.pkl, cm.pkl ✅")



