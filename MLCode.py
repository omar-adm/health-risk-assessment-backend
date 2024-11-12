import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Load and preprocess the data
data = pd.read_csv('orgninal_clean_dataset_for_ML_951.csv')
X = data.drop('Diabetic', axis=1)
y = data['Diabetic']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

def preprocess_input(input_string):
    values = input_string.split(',')
    input_data = pd.DataFrame([values], columns=X.columns)
    
    # Convert categorical variables
    input_data['Gender'] = input_data['Gender'].map({'male': 0, 'female': 1})
    for col in ['Family_Diabetes', 'highBP', 'PhysicallyActive', 'Smoking', 'Alcohol', 'RegularMedicine', 'JunkFood', 'Stress', 'Pdiabetes']:
        input_data[col] = input_data[col].map({'yes': 1, 'no': 0})
    input_data['UriationFreq'] = input_data['UriationFreq'].map({'low': 0, 'high': 1})
    
    # Convert to float
    input_data = input_data.astype(float)
    
    return scaler.transform(input_data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_string>")
        sys.exit(1)

    input_string = sys.argv[1]
    input_scaled = preprocess_input(input_string)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Calculate performance metrics on the test set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Print results as comma-separated values
    print(f"{prediction},{probability},{accuracy},{sensitivity},{specificity}")