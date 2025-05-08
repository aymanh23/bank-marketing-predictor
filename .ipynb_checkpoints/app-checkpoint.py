import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('best_model.pkl')

st.title("Bank Marketing Subscription Prediction")

# === Mappings ===
education_mapping = {
    'Basic (4 years)': 0,
    'Basic (6 years)': 1,
    'Basic (9 years)': 2,
    'High School': 3,
    'Illiterate': 4,
    'Professional Course': 5,
    'University Degree': 6
}

poutcome_options = ['Success', 'Nonexistent']

# === Inputs ===
duration = st.number_input('Duration (seconds)', value=0)
nr_employed = st.number_input('Number Employed', value=0.0)
pdays = st.number_input('Days Since Last Contact (pdays)', value=999)
poutcome = st.radio('Previous Outcome', poutcome_options)
poutcome_success = int(poutcome == 'Success')
poutcome_nonexistent = int(poutcome == 'Nonexistent')
euribor3m = st.number_input('Euribor 3m', value=0.0)
emp_var_rate = st.number_input('Employment Variation Rate', value=0.0)
previous = st.number_input('Previous Contacts', value=0)
cons_price_idx = st.number_input('Consumer Price Index', value=0.0)
campaign = st.number_input('Campaign Contacts', value=0)

education = st.selectbox('Education', list(education_mapping.keys()))
education_encoded = education_mapping[education]

# === Build feature array ===
features = np.array([[
    duration,
    nr_employed,
    pdays,
    poutcome_success,
    euribor3m,
    emp_var_rate,
    previous,
    poutcome_nonexistent,
    cons_price_idx,
    campaign,
    education_encoded
]])

# === Predict ===
if st.button("Predict"):
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    st.write("📢 Prediction:", "✅ Subscribed" if prediction[0] == 1 else "❌ Not Subscribed")
    st.write(f"🔍 Confidence: {prob:.2%}")
