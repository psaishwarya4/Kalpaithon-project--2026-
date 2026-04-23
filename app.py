from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Training data
X = [
    [70, 0.02, 30],
    [90, 0.05, 45],
    [85, 0.04, 40],
    [95, 0.07, 50],
    [60, 0.01, 20],
    [100, 0.08, 55]
]

y_test = [0,0,0,1,0,1]

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X,y_test)
y_pred = model.predict(X)
accuracy = accuracy_score(y_test, y_pred)
st.write("Model Accuracy:", round(accuracy*100,2), "%")