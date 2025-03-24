import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

##Load the dataset

df = pd.read_csv("sepsis_balanced.csv")

##Define feature columns and target variable

features = ["HeartRate", "BloodPressure", "RespiratoryRate", "WBC_Count",
"Temperature", "Lactate", "SpO2", "PlateletCount", "Glucose", "CRP"]
target = "Sepsis"

##Split the dataset into features and labels

X = df[features]
y = df[target]

##Handle class imbalance using SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

##Standardize the features

scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

##Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

##Train a RandomForest model with balanced class weights

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

##Predict on the test set

y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

Adjust decision threshold dynamically to optimize recall

threshold = 0.4  # Lower threshold to detect more positive cases
y_pred_adjusted = (y_probs > threshold).astype(int)

##Evaluate the model

print("Adjusted Accuracy:", accuracy_score(y_test, y_pred_adjusted))
print("Classification Report with Adjusted Threshold:\n", classification_report(y_test, y_pred_adjusted))

##Feature Importance Analysis

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(np.array(features)[sorted_idx], feature_importance[sorted_idx], color="blue")
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Sepsis Prediction Model")
plt.show()

##Save the trained model and scaler

joblib.dump(model, "sepsis_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Updated model and scaler saved successfully!")
