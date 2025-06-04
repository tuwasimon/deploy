import joblib
import streamlit as st
import numpy as np

# Load the trained model
model = joblib.load('deploy/tips.pkl')

# Set page config
st.set_page_config(page_title='Tip Classifier', page_icon='💸', layout='centered')

# Custom title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💡 Tip Classifier App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict the gender of a customer based on tipping behavior 💁‍♂️💁‍♀️</h4>", unsafe_allow_html=True)
st.markdown("---")

# Create input layout
col1, col2, col3 = st.columns(3)

with col1:
    total_bill = st.number_input('💵 Total Bill', min_value=0.0, step=0.1)

with col2:
    tip = st.number_input('💰 Tip Amount', min_value=0.0, step=0.1)

with col3:
    size = st.number_input('👥 Party Size', min_value=1, step=1)

# Predict Button
if st.button('🔍 Classify'):
    input_features = np.array([[total_bill, tip, size]])
    pred = model.predict(input_features)
    mapping = ['Male', 'Female']
    prediction = mapping[pred[0]]

    # Display result
    st.markdown("### 🎯 Prediction Result")
    st.success(f"Based on the input, the predicted gender is: **{prediction}**")

    # Optional: add confidence or more info here
