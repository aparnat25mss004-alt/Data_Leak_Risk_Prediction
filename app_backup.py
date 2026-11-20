
import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoder
@st.cache_resource
def load_model():
    model = joblib.load("models/risk_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

st.set_page_config(
    page_title="Data Leak Risk Prediction",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Data Leak Risk Prediction System")
st.subheader("Employee Activity Risk Analysis")

st.write("Analyze employee activities and identify potential data leak risks.")

st.divider()

# Employee details
st.header("👤 Employee Details")

col1, col2 = st.columns(2)

with col1:
    employee_id = st.text_input("Employee ID", placeholder="EMP001")

with col2:
    department = st.selectbox(
        "Department",
        ["IT", "HR", "Finance", "Sales", "Operations", "Other"]
    )

st.divider()

# Model input features
st.header("📊 Employee Activity Details")

col1, col2 = st.columns(2)

with col1:
    files_accessed = st.number_input(
        "Files Accessed",
        min_value=0,
        value=10
    )

    download_mb = st.number_input(
        "Download Size (MB)",
        min_value=0,
        value=100
    )

    sensitive_files_accessed = st.number_input(
        "Sensitive Files Accessed",
        min_value=0,
        value=0
    )

    external_sharing = st.selectbox(
        "External Sharing",
        ["No", "Yes"]
    )

with col2:
    usb_transfer = st.selectbox(
        "USB Transfer",
        ["No", "Yes"]
    )

    after_hours_activity = st.selectbox(
        "After-hours Activity",
        ["No", "Yes"]
    )

    failed_access_attempts = st.number_input(
        "Failed Access Attempts",
        min_value=0,
        value=0
    )

st.divider()

# Prediction
if st.button("🔍 Predict Data Leak Risk", use_container_width=True):

    if not employee_id.strip():
        st.warning("Please enter Employee ID.")

    else:
        employee_data = pd.DataFrame([{
            "files_accessed": files_accessed,
            "download_mb": download_mb,
            "sensitive_files_accessed": sensitive_files_accessed,
            "external_sharing": 1 if external_sharing == "Yes" else 0,
            "usb_transfer": 1 if usb_transfer == "Yes" else 0,
            "after_hours_activity": 1 if after_hours_activity == "Yes" else 0,
            "failed_access_attempts": failed_access_attempts
        }])

        try:
            prediction = model.predict(employee_data)
            risk_label = encoder.inverse_transform(prediction)[0]

            st.subheader("Prediction Result")

            st.write(f"Employee ID: **{employee_id}**")
            st.write(f"Department: **{department}**")

            st.success(f"Predicted Risk Level: {risk_label}")

            # Probability score, if model supports it
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(employee_data)[0]
                confidence = max(probabilities) * 100

                st.metric("Prediction Confidence", f"{confidence:.2f}%")

            st.subheader("Activity Summary")
            st.dataframe(employee_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.divider()

st.caption("Employee Data Leak Risk Prediction System | ML Project")
