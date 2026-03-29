# Diabetes_prediction_System
Early diabetes risk predictor using Pima Indians Diabetes Dataset. Implements Logistic Regression, Decision Tree &amp; SVM. Auto-selects best model with accuracy chart &amp; confusion matrix.  Interactive Streamlit web app. Built with Python, scikit-learn, Pandas.  #MachineLearning #HealthcareAI
# 🩺 Diabetes Prediction System using Machine Learning

**An interactive web application that predicts the risk of diabetes using patient health parameters.**

Built as part of the **Fundamentals in AI and ML** course project.

## ✨ Features

- **Real-time Diabetes Risk Prediction** via user-friendly Streamlit interface
- **Three Classification Models**:
  - Logistic Regression
  - Decision Tree
  - Support Vector Machine (SVM)
- **Automatic Best Model Selection** based on highest accuracy
- **Data Preprocessing**:
  - Replaced invalid zero values with column mean
  - Feature scaling using `StandardScaler`
  - 80-20 train-test split
- **Model Evaluation**:
  - Accuracy comparison bar chart
  - Confusion matrix visualization

## 📊 Dataset

**Pima Indians Diabetes Dataset**  
- 768 records  
- 8 medical features + binary Outcome (`0 = No Diabetes`, `1 = Diabetes`)

**Features:**
- Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** (models & metrics)
- **Streamlit** (web interface)
- **Pandas & NumPy** (data handling)
- **Matplotlib** (visualizations)
- **Pickle** (model serialization)

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/diabetes-prediction-system.git
cd diabetes-prediction-system

# 2. Install dependencies
pip install pandas numpy scikit-learn streamlit

# 3. Run the Streamlit app
python model.py
streamlit run app.py
