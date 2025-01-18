import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# App title
st.title("Prediction App")

# Instructions
st.write("Input the values for the features to get a prediction.")

# Input fields for user data
# Replace 'Feature 1', 'Feature 2', ... with the actual names of your features
features = [
    st.number_input(f"Feature {i+1}", value=0.0)
    for i in range(len(model.feature_importances_))
]

# Button for prediction
if st.button("Predict"):
    # Make predictions
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.success(f"Predicted Value: {prediction[0]}")
