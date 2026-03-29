import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# LOAD FILES
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
accuracies = pickle.load(open("accuracies.pkl", "rb"))
cm = pickle.load(open("cm.pkl", "rb"))

st.set_page_config(page_title="Diabetes Predictor", page_icon="🧠")

st.title("🧠 Diabetes Prediction System")

# -------------------------------
# INPUT
# -------------------------------
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age", min_value=0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

# -------------------------------
# ACCURACY GRAPH
# -------------------------------
st.subheader("📊 Model Accuracy Comparison")

models = list(accuracies.keys())
scores = list(accuracies.values())

fig1, ax1 = plt.subplots()
ax1.bar(models, scores)
ax1.set_ylabel("Accuracy")
ax1.set_title("Model Comparison")

st.pyplot(fig1)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
st.subheader("📈 Confusion Matrix")

fig2, ax2 = plt.subplots()
ax2.imshow(cm)
ax2.set_title("Confusion Matrix")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax2.text(j, i, cm[i][j], ha="center", va="center")

st.pyplot(fig2)






