import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from streamlit_lottie import st_lottie

# --- Helper: Lottie Animation Loader ---


def load_lottie_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# --- Sidebar Page Navigation ---
st.markdown('''
    <style>
    div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] .css-1cpxqw2 {
        font-size: 1.25em !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSidebar"] .stSelectbox {
        font-size: 1.2em !important;
        padding: 0.7em 0.5em !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    </style>
''', unsafe_allow_html=True)
page = st.sidebar.selectbox(
    'Navigate', ['Home', 'Use Model', 'Help', 'Project Team'])

# --- Lottie Animations ---
hello_lottie = load_lottie_file('hello.json')
machine_lottie = load_lottie_file('machine.json')

# --- Helper: Categorical Options ---
job_options = [
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
    'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'
]
marital_options = ['single', 'married', 'divorced', 'unknown']
education_options = [
    'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
    'professional.course', 'university.degree', 'unknown'
]
default_options = ['no', 'yes', 'unknown']
housing_options = ['no', 'yes', 'unknown']
loan_options = ['no', 'yes', 'unknown']
contact_options = ['cellular', 'telephone']
month_options = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
]
day_of_week_options = ['mon', 'tue', 'wed', 'thu', 'fri']
poutcome_options = ['failure', 'nonexistent', 'success']

# --- Home Page ---
if page == 'Home':
    st.markdown('''
    <div style="display:flex;align-items:center;">
        <div style="flex:2;">
            <span style="font-size:3.2em;font-weight:900;line-height:1.1;">💰 Bank Marketing Subscription Prediction</span>
            <div style="margin-top:18px;font-size:1.35em;font-weight:600;">Welcome to the ADA442 Project!</div>
            <div style="margin-top:10px;font-size:1.15em;font-weight:400;max-width:700px;">
                This application predicts whether a bank client will subscribe to a term deposit after a marketing campaign, using a machine learning model trained on real-world data from a Portuguese bank.
            </div>
        </div>
        <div style="flex:1;text-align:right;">
    ''', unsafe_allow_html=True)
    if hello_lottie:
        st_lottie(hello_lottie, height=250, key="hello")
    else:
        st.markdown(
            '''<div style="font-size:80px; text-align:center;">🤖</div>''', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('''
    ### Model Used
    - **Decision Tree Classifier** (selected for best recall and F1 on the positive class)
    
    ### Model Performance (Test Set)
    - **F1 Score (Class 1):** 0.9633
    - **Recall (Class 1):** 1.0000
    - **Weighted F1 Score:** 0.9618
    
    ### How Was the Model Trained?
    - Data cleaning and preprocessing: handled missing values, encoded categorical variables, scaled numeric features, and performed feature engineering (including engineered features for previous contact, campaign interactions, etc.).
    - Multiple models were evaluated (Logistic Regression, KNN, Random Forest, Gradient Boosting, Decision Tree).
    - **Decision Tree** was chosen for its perfect recall (no missed subscribers) and strong F1 score for the positive class.
    - Hyperparameters were tuned using GridSearchCV (cross-validated F1-weighted score).
    - The final model is saved and deployed for instant predictions.
    
    *See the Help page for feature explanations, or try the Prediction page to use the model!*
    ''')
    st.markdown('---')

# --- Prediction Page ---
elif page == 'Use Model':
    st.markdown('<h1 style="text-align:center; font-weight:900;">Use the Model to Predict</h1>',
                unsafe_allow_html=True)
    if machine_lottie:
        st_lottie(machine_lottie, height=180, key="predicting-top")
    st.markdown(
        'Fill in the client and campaign details below, then click Predict!')
    with st.form('input_form'):
        st.subheader('Client Information')
        age = st.number_input('Age', min_value=18, max_value=100, value=35)
        job = st.selectbox('Job', job_options)
        marital = st.selectbox('Marital Status', marital_options)
        education = st.selectbox('Education', education_options)
        default = st.selectbox('Defaulted Credit?', default_options)
        housing = st.selectbox('Has Housing Loan?', housing_options)
        loan = st.selectbox('Has Personal Loan?', loan_options)

        st.subheader('Contact & Campaign')
        contact = st.selectbox('Contact Communication Type', contact_options)
        month = st.selectbox('Last Contact Month', month_options)
        day_of_week = st.selectbox(
            'Last Contact Day of Week', day_of_week_options)
        campaign = st.number_input(
            'Number of Contacts in this Campaign', min_value=1, max_value=100, value=1)
        previous = st.number_input(
            'Number of Previous Contacts', min_value=0, max_value=100, value=0)
        poutcome = st.selectbox(
            'Outcome of Previous Campaign', poutcome_options)
        pdays = st.number_input(
            'Days Since Last Contact (999 = never)', min_value=-1, max_value=999, value=999)
        duration = st.number_input(
            'Last Contact Duration (seconds)', min_value=0, max_value=5000, value=0)

        st.subheader('Economic Indicators')
        emp_var_rate = st.number_input(
            'Employment Variation Rate', value=1.1, format='%.2f')
        cons_price_idx = st.number_input(
            'Consumer Price Index', value=93.994, format='%.3f')
        cons_conf_idx = st.number_input(
            'Consumer Confidence Index', value=-36.4, format='%.1f')
        euribor3m = st.number_input(
            'Euribor 3 Month Rate', value=4.855, format='%.3f')
        nr_employed = st.number_input(
            'Number of Employees', value=5191.0, format='%.1f')

        submitted = st.form_submit_button('Predict')

    # --- Feature Engineering ---
    was_previously_contacted = int(pdays != 999)
    pdays_mod = pdays if pdays != 999 else -1
    campaign_previous_interaction = campaign * previous

    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
        'was_previously_contacted': was_previously_contacted,
        'pdays_mod': pdays_mod,
        'campaign_previous_interaction': campaign_previous_interaction,
        'duration': duration
    }
    input_df = pd.DataFrame([input_dict])

    if submitted:
        if not os.path.exists('model.pkl'):
            st.error(
                'Model file not found! Please ensure model.pkl is in the app directory.')
        else:
            with st.spinner('Predicting...'):
                model = joblib.load('model.pkl')
                pred = model.predict(input_df)[0]
            pred_label = 'Will Subscribe (Yes)' if pred == 1 or pred == 'yes' else 'Will Not Subscribe (No)'
            if pred == 1 or pred == 'yes':
                st.markdown(
                    '''<div style="background-color:#d4edda;padding:24px 0 24px 0;border-radius:12px;text-align:center;">
                        <span style="color:#155724;font-size:2.2em;font-weight:bold;">Prediction: Will Subscribe (Yes)</span>
                    </div>''',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '''<div style="background-color:#f8d7da;padding:24px 0 24px 0;border-radius:12px;text-align:center;">
                        <span style="color:#721c24;font-size:2.2em;font-weight:bold;">Prediction: Will Not Subscribe (No)</span>
                    </div>''',
                    unsafe_allow_html=True)

# --- Help Page ---
elif page == 'Help':
    st.title('Help: Feature Explanations')
    st.markdown('''
    Here you can find detailed explanations for each input feature required by the model:
    ''')
    st.markdown('''
- **age**: Age of the client (numeric)
- **job**: Type of job (categorical)
- **marital**: Marital status (categorical)
- **education**: Education level (categorical)
- **default**: Has credit in default? (yes, no, unknown)
- **housing**: Has housing loan? (yes, no, unknown)
- **loan**: Has personal loan? (yes, no, unknown)
- **contact**: Contact communication type (cellular, telephone)
- **month**: Last contact month of year
- **day_of_week**: Last contact day of the week
- **duration**: Last contact duration, in seconds
- **campaign**: Number of contacts performed during this campaign and for this client (numeric, includes last contact)
- **previous**: Number of contacts performed before this campaign and for this client
- **pdays**: Number of days that passed by after the client was last contacted from a previous campaign (999 means never contacted)
- **poutcome**: Outcome of the previous marketing campaign (failure, nonexistent, success)
- **emp.var.rate**: Employment variation rate (numeric)
- **cons.price.idx**: Consumer price index (numeric)
- **cons.conf.idx**: Consumer confidence index (numeric)
- **euribor3m**: Euribor 3 month rate (numeric)
- **nr.employed**: Number of employees (numeric)
- **was_previously_contacted**: Engineered feature: 1 if pdays != 999 else 0
- **pdays_mod**: Engineered feature: pdays if pdays != 999 else -1
- **campaign_previous_interaction**: Engineered feature: campaign * previous
    ''')
    st.markdown('---')
    st.info('If you have any questions, please contact the project team.')

# --- Project Team Page ---
elif page == 'Project Team':
    st.title('Project Team')
    st.markdown('''
    <div style="font-size:1.3em;">
    <ul>
        <li><b>Ayman Hamdan</b></li>
        <li><b>Ibrahim Yusuf</b></li>
        <li><b>Mervan Özgünül</b></li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)

# --- Footer ---
st.markdown('---')
st.caption('Group 16 | ADA442 Project | 2025')
