import streamlit as st
import pandas as pd
import joblib

model_package = joblib.load("customer_churn_rf_model.joblib")

model = model_package["model"]
feature_columns = model_package["feature_columns"]
threshold = model_package["threshold"]

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
        padding: 34px 34px;
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 28px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.22);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        color: #f8fafc;
        margin-bottom: 8px;
        letter-spacing: -0.8px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #cbd5e1;
        line-height: 1.7;
        max-width: 760px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        margin-top: 22px;
        margin-bottom: 6px;
        color: #f8fafc;
    }

    .section-desc {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 14px;
    }

    .icon-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: rgba(59, 130, 246, 0.14);
        border: 1px solid rgba(59, 130, 246, 0.22);
        margin-right: 8px;
    }

    [data-testid="stForm"] {
        border-radius: 24px;
        padding: 26px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.28);
        box-shadow: 0 14px 34px rgba(0,0,0,0.16);
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        height: 56px;
        border-radius: 16px;
        font-size: 17px;
        font-weight: 800;
        border: none;
        margin-top: 18px;
    }

    [data-testid="stExpander"] {
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.18);
    }

    .result-churn {
        padding: 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, #7f1d1d, #b91c1c, #dc2626);
        color: #ffffff;
        text-align: center;
        font-size: 32px;
        font-weight: 900;
        margin-top: 28px;
        box-shadow: 0 16px 38px rgba(220,38,38,0.28);
        border: 1px solid rgba(254,202,202,0.2);
    }

    .result-safe {
        padding: 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, #064e3b, #047857, #059669);
        color: #ffffff;
        text-align: center;
        font-size: 32px;
        font-weight: 900;
        margin-top: 28px;
        box-shadow: 0 16px 38px rgba(5,150,105,0.28);
        border: 1px solid rgba(167,243,208,0.2);
    }

    .result-note {
        margin-top: 14px;
        padding: 18px 20px;
        border-radius: 16px;
        background: rgba(148,163,184,0.10);
        color: #d1d5db;
        text-align: center;
        font-size: 16px;
        border: 1px solid rgba(148,163,184,0.16);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📊 Customer Churn Prediction</div>
        <div class="hero-subtitle">
            A professional machine learning application that analyzes telecom customer information
            and predicts whether the customer is likely to leave the company.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("customer_churn_form"):
    st.markdown(
        '<div class="section-title"><span class="icon-badge">👤</span>Customer Profile</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-desc">Basic customer identity, relationship status, and account duration.</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    with p2:
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with p3:
        tenure = st.number_input("Tenure in Months", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    st.markdown(
        '<div class="section-title"><span class="icon-badge">💼</span>Contract & Billing</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-desc">Subscription plan, billing method, and payment behavior.</div>',
        unsafe_allow_html=True
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

    with b2:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

    with b3:
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    st.markdown(
        '<div class="section-title"><span class="icon-badge">🌐</span>Internet Services</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-desc">Internet plan, security, backup, protection, support, and streaming services.</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3 = st.columns(3)

    with s1:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

    with s2:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with s3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    with st.expander("Additional Service Details"):
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    predict_button = st.form_submit_button("Analyze Customer Churn", use_container_width=True)

input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = 0

input_data.at[0, "gender"] = 1 if gender == "Male" else 0
input_data.at[0, "SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
input_data.at[0, "Partner"] = 1 if partner == "Yes" else 0
input_data.at[0, "Dependents"] = 1 if dependents == "Yes" else 0
input_data.at[0, "tenure"] = tenure
input_data.at[0, "PhoneService"] = 1 if phone_service == "Yes" else 0
input_data.at[0, "PaperlessBilling"] = 1 if paperless_billing == "Yes" else 0
input_data.at[0, "MonthlyCharges"] = monthly_charges
input_data.at[0, "TotalCharges"] = total_charges

if multiple_lines == "No phone service":
    input_data.at[0, "MultipleLines_No phone service"] = 1
elif multiple_lines == "Yes":
    input_data.at[0, "MultipleLines_Yes"] = 1

if internet_service == "Fiber optic":
    input_data.at[0, "InternetService_Fiber optic"] = 1
elif internet_service == "No":
    input_data.at[0, "InternetService_No"] = 1

if online_security == "No internet service":
    input_data.at[0, "OnlineSecurity_No internet service"] = 1
elif online_security == "Yes":
    input_data.at[0, "OnlineSecurity_Yes"] = 1

if online_backup == "No internet service":
    input_data.at[0, "OnlineBackup_No internet service"] = 1
elif online_backup == "Yes":
    input_data.at[0, "OnlineBackup_Yes"] = 1

if device_protection == "No internet service":
    input_data.at[0, "DeviceProtection_No internet service"] = 1
elif device_protection == "Yes":
    input_data.at[0, "DeviceProtection_Yes"] = 1

if tech_support == "No internet service":
    input_data.at[0, "TechSupport_No internet service"] = 1
elif tech_support == "Yes":
    input_data.at[0, "TechSupport_Yes"] = 1

if streaming_tv == "No internet service":
    input_data.at[0, "StreamingTV_No internet service"] = 1
elif streaming_tv == "Yes":
    input_data.at[0, "StreamingTV_Yes"] = 1

if streaming_movies == "No internet service":
    input_data.at[0, "StreamingMovies_No internet service"] = 1
elif streaming_movies == "Yes":
    input_data.at[0, "StreamingMovies_Yes"] = 1

if contract == "One year":
    input_data.at[0, "Contract_One year"] = 1
elif contract == "Two year":
    input_data.at[0, "Contract_Two year"] = 1

if payment_method == "Credit card (automatic)":
    input_data.at[0, "PaymentMethod_Credit card (automatic)"] = 1
elif payment_method == "Electronic check":
    input_data.at[0, "PaymentMethod_Electronic check"] = 1
elif payment_method == "Mailed check":
    input_data.at[0, "PaymentMethod_Mailed check"] = 1

input_data = input_data[feature_columns]

if predict_button:
    churn_probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if churn_probability >= threshold else 0

    if prediction == 1:
        st.markdown(
            '<div class="result-churn">Customer is likely to churn</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="result-note">Recommended action: contact this customer with retention support, service improvement, or a better plan offer.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-safe">Customer is not likely to churn</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="result-note">This customer appears stable based on the provided account and service information.</div>',
            unsafe_allow_html=True
        )