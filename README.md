# **Machine Learning Platform for Model Exploration, Testing Web App**  

## 📌 Project Overview

This project is a **machine learning and deep learning web application** developed using **Flask and Streamlit**. It provides an interactive platform for dataset exploration, model evaluation, AutoML, and chatbot interaction, integrated with **MLOps & CI/CD pipelines**.

## 🚀 Features

### 🔹 User Authentication

- **Login, Register, Logout, Profile Page**
- **User data stored in AWS-hosted PostgreSQL**

### 🔹 Home Page

- Three primary options:
  1. **Explore Datasets**
  2. **Explore Models**
  3. **Models via Categories**

### 🔹 Dataset Page

- **Preloaded datasets**: 5 for classification, 5 for regression
- **User-uploaded dataset support**
- **Dataset visualization** using Seaborn (various plots)
- **Exploratory Data Analysis (EDA)**
- **Pretrained model results** displayed for each dataset

### 🔹 Model Page

- **Pretrained Model Results** on default datasets
- **AutoML Support** using TPOT & Optuna
- **Hyperparameter Tuning** with GridCV & RandomCV
- **Custom Model Exploration** with user-defined parameters
- **SHAP Analysis** for model interpretability
- **MLflow Integration** via Airflow (users register MLflow repo details)

### 🔹 Categories Page

- View **all model results on a custom dataset**
- Parameter selection & hyperparameter tuning
- **Comparison of all model parameters**
- **Confusion matrices for classification** & **regression graphs**
- **Model comparison feature (in progress)**

### 🔹 Chatbot Integration

- Available in **Flask & Streamlit (sidebar)**
- Uses **Groq API** for ChatGPT-like responses
- Integrated with **Wikipedia & YouTube search tools**
- **User sessions stored in AWS-hosted PostgreSQL**
- **Upcoming retrieval-based chatbot using pincode**

### 🔹 CI/CD Pipeline

- **Automated Deployment** via GitHub Actions, Docker, and AWS
- **Workflow:**
  1. **Commit to GitHub → Docker Image Created**
  2. **Push to Docker Hub & AWS ECR**
  3. **Deploy on AWS EC2**

### 🔹 Upcoming Enhancements

- **Retrieval-based chatbot using pincode**
- **Model comparison in the categories page**
- **Task scheduling & automation with Airflow**

## 🛠️ Tech Stack

- **Frontend & UI**: Flask, Streamlit, Seaborn (for visualization)
- **Backend**: Flask, PostgreSQL (AWS-hosted), SQLAlchemy
- **Machine Learning**: Scikit-learn, TensorFlow, PyTorch
- **AutoML**: TPOT, Optuna
- **MLOps**: MLflow, Apache Airflow
- **Deployment**: Docker, AWS (EC2, ECR, S3), GitHub Actions

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/ml-dl-webapp.git
cd ml-dl-webapp
```

### 2️⃣ Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Flask Backend

```bash
python app.py
```

### 5️⃣ Run Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

### 6️⃣ Access the Application

- **Flask Backend**: `http://localhost:5000`
- **Streamlit Frontend**: `http://localhost:8501`


## 📝 License

This project is licensed under the MIT License.

