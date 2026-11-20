
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Data Leak Risk Dashboard",
    page_icon="🔐",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("models/risk_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# ---------------- SESSION STATE ----------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ---------------- HEADER ----------------
st.title("🔐 Data Leak Risk Prediction System")
st.subheader("Employee Activity Monitoring & Risk Analytics")
st.write("Analyze employee activities and identify potential data leak risks.")

st.divider()

# ---------------- DASHBOARD METRICS ----------------
history = st.session_state.prediction_history

total_predictions = len(history)

high_risk_count = sum(
    "high" in str(item["Risk"]).strip().lower()
    for item in history
)

low_risk_count = sum(
    str(item["Risk"]).strip().lower() == "low"
    for item in history
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total_predictions)
col2.metric("High Risk Employees", high_risk_count)
col3.metric("Low Risk Employees", low_risk_count)

st.divider()

# ---------------- RISK ANALYTICS ----------------
st.header("📊 Risk Analytics Dashboard")

if history:

    history_df = pd.DataFrame(history)

    risk_data = pd.DataFrame({
        "Risk Level": ["High / Very High", "Low"],
        "Employees": [high_risk_count, low_risk_count]
    })

    st.subheader("Employee Risk Distribution")
    st.bar_chart(risk_data.set_index("Risk Level"))

    st.subheader("Prediction History")
    st.dataframe(history_df, use_container_width=True)

    # ---------------- DOWNLOAD REPORT ----------------
    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="employee_risk_prediction_report.csv",
        mime="text/csv"
    )

else:
    st.info("Prediction data will appear here after you analyze an employee.")

st.divider()

# ---------------- EMPLOYEE DETAILS ----------------
st.header("👤 Employee Details")

col1, col2 = st.columns(2)

with col1:
    employee_id = st.text_input(
        "Employee ID",
        placeholder="EMP001"
    )

with col2:
    department = st.selectbox(
        "Department",
        ["IT", "HR", "Finance", "Sales", "Operations", "Other"]
    )

st.divider()

# ---------------- ACTIVITY INPUT ----------------
st.header("📋 Employee Activity Details")

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

# ---------------- PREDICTION ----------------
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
            risk_label = str(encoder.inverse_transform(prediction)[0])

            # ---------------- SAVE HISTORY ----------------
            record = {
                "Employee ID": employee_id,
                "Department": department,
                "Risk": risk_label,
                "Files Accessed": files_accessed,
                "Download (MB)": download_mb,
                "Sensitive Files": sensitive_files_accessed,
                "External Sharing": external_sharing,
                "USB Transfer": usb_transfer,
                "After Hours": after_hours_activity,
                "Failed Attempts": failed_access_attempts
            }

            st.session_state.prediction_history.append(record)

            # ---------------- PREDICTION RESULT ----------------
            st.divider()
            st.header("🎯 Prediction Result")

            normalized_risk = risk_label.strip().lower()

            if "very high" in normalized_risk:

                st.error(f"🚨 VERY HIGH RISK: Employee {employee_id}")
                st.warning(
                    "Critical suspicious activity detected. "
                    "Immediate investigation is recommended."
                )

            elif "high" in normalized_risk:

                st.error(f"🚨 HIGH RISK: Employee {employee_id}")
                st.warning(
                    "Suspicious employee activity detected. "
                    "Further investigation may be required."
                )

            elif "medium" in normalized_risk:

                st.warning(f"⚠️ MEDIUM RISK: Employee {employee_id}")
                st.info("Moderate risk activity detected. Monitor employee activity.")

            else:

                st.success(f"🟢 LOW RISK: Employee {employee_id}")
                st.info("The model classified this employee's activity as low risk.")

            st.write(f"**Employee ID:** {employee_id}")
            st.write(f"**Department:** {department}")
            st.write(f"**Predicted Risk Level:** {risk_label}")

            # ---------------- RISK ANALYSIS ----------------
            st.subheader("🔎 Risk Analysis")

            risk_factors = []

            if external_sharing == "Yes":
                risk_factors.append("External data sharing detected")

            if usb_transfer == "Yes":
                risk_factors.append("Suspicious USB activity")

            if after_hours_activity == "Yes":
                risk_factors.append("Unusual after-hours activity")

            if sensitive_files_accessed > 5:
                risk_factors.append("High sensitive file access")

            if download_mb > 1000:
                risk_factors.append("Large data download detected")

            if failed_access_attempts > 3:
                risk_factors.append("Multiple failed access attempts")

            if files_accessed > 100:
                risk_factors.append("Unusually high file access")

            if risk_factors:

                for factor in risk_factors:
                    st.warning("⚠️ " + factor)

            else:
                st.success("No major suspicious activity indicators detected.")

            # ---------------- CONFIDENCE ----------------
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(employee_data)[0]
                confidence = max(probabilities) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            # ---------------- ACTIVITY SUMMARY ----------------
            st.subheader("📋 Activity Summary")
            st.dataframe(employee_data, use_container_width=True)

            # ---------------- ACTIVITY CHART ----------------
            st.subheader("📈 Employee Activity Chart")

            activity_chart = pd.DataFrame({
                "Activity": [
                    "Files Accessed",
                    "Download MB",
                    "Sensitive Files",
                    "External Sharing",
                    "USB Transfer",
                    "After-hours",
                    "Failed Attempts"
                ],
                "Value": [
                    files_accessed,
                    download_mb,
                    sensitive_files_accessed,
                    1 if external_sharing == "Yes" else 0,
                    1 if usb_transfer == "Yes" else 0,
                    1 if after_hours_activity == "Yes" else 0,
                    failed_access_attempts
                ]
            })

            st.bar_chart(activity_chart.set_index("Activity"))

        except Exception as e:

            st.error(f"Prediction error: {e}")

st.divider()

st.caption(
    "Employee Data Leak Risk Prediction System | Machine Learning Project"
)
