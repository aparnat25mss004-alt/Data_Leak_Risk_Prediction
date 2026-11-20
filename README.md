# 🔐 Data Leak Risk Prediction System

## 📌 Project Overview

The **Data Leak Risk Prediction System** is a Machine Learning-based application designed to analyze employee activities and identify potential data leakage risks.

It evaluates employee activity patterns such as file access, data downloads, sensitive file access, USB transfers, external sharing, after-hours activity, and failed access attempts to predict risk levels.

The application provides an interactive dashboard for analyzing employee behavior and identifying potentially suspicious activities.

## 🎯 Objectives

- Identify potentially risky employee activities.
- Predict employee data leak risk levels using Machine Learning.
- Detect suspicious behavioral patterns.
- Visualize employee activity and risk analysis.
- Provide an interactive dashboard for risk assessment.

## 🚀 Features

- Employee activity analysis
- Machine Learning-based risk prediction
- Low, Medium, and High risk classification
- Interactive Streamlit dashboard
- Prediction confidence display
- Risk indicator analysis
- Prediction history
- Data visualization and charts
- CSV export
- Docker containerization

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- FastAPI
- Joblib
- Docker
- Git & GitHub

## 📊 Machine Learning Input Features

The model analyzes the following employee activity features:

| Feature | Description |
|---|---|
| `files_accessed` | Number of files accessed |
| `download_mb` | Amount of data downloaded in MB |
| `sensitive_files_accessed` | Sensitive file access activity |
| `external_sharing` | Whether data was shared externally |
| `usb_transfer` | Whether USB data transfer occurred |
| `after_hours_activity` | Activity outside normal working hours |
| `failed_access_attempts` | Number of failed access attempts |

## 🏗️ Project Structure

```text
Data_Leak_Risk_Prediction/
│
├── app.py
├── app_backup.py
├── train.py
├── predict.py
├── confusion_matrix.py
├── model_comparison.py
├── mlflow_tracking.py
│
├── dataset/
│   └── employee_activity.csv
│
├── models/
│   ├── risk_model.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── data_check.py
│   └── preprocessing.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aparnat25mss004-alt/Data_Leak_Risk_Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Data_Leak_Risk_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

## 🐳 Running with Docker

### Build the Docker Image

```bash
docker build -t data-leak-risk-app .
```

### Run the Docker Container

```bash
docker run -d -p 8501:8501 --name risk-dashboard data-leak-risk-app
```

### Access the Dashboard

```text
http://localhost:8501
```

### Stop the Container

```bash
docker stop risk-dashboard
```

### Remove the Container

```bash
docker rm risk-dashboard
```

## 📈 Application Workflow

1. Enter employee details.
2. Enter employee activity data.
3. Submit the activity information.
4. The Machine Learning model analyzes the data.
5. The system predicts the employee's risk level.
6. The dashboard displays the prediction and confidence.
7. Risk indicators help identify suspicious activities.
8. Prediction history and charts support further analysis.

## 🔒 Purpose

This project is developed for educational purposes to demonstrate the application of Machine Learning in employee activity monitoring and data leak risk assessment.

## 👩‍💻 Author

**Aparna**

## 📄 License

This project is intended for academic and educational use.
