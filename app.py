import streamlit as st
import pandas as pd
import joblib

# Set application title and wide layout
st.set_page_config(page_title="Multi-Model Prediction Dashboard", layout="wide")
st.title("🤖 AIML Multi-Model Prediction Dashboard")

# Function to load classification models with caching
@st.cache_resource
def load_clf_artifacts():
    scaler = joblib.load("saved_models/scaler_clf.pkl")
    cols = joblib.load("saved_models/clf_columns.pkl")
    models = joblib.load("saved_models/all_clf_models.pkl")
    return scaler, cols, models

# Function to load regression models with caching
@st.cache_resource
def load_reg_artifacts():
    scaler = joblib.load("saved_models/scaler_reg.pkl")
    cols = joblib.load("saved_models/reg_columns.pkl")
    models = joblib.load("saved_models/all_reg_models.pkl")
    return scaler, cols, models

# Sidebar problem selection
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])

# --- CLASSIFICATION SECTION ---
if problem_type == "Classification":
    st.header("🩸 Diabetes Prediction System (Classification)")
    
    try:
        scaler, columns, all_models = load_clf_artifacts()
        
        # Select Algorithm from sidebar
        model_choice = st.sidebar.selectbox("Select Algorithm", list(all_models.keys()))
        selected_model = all_models[model_choice]
        
        st.subheader(f"Enter Inputs (Model Selected: {model_choice})")
        
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", value=1)
            glucose = st.number_input("Glucose", value=120)
            blood_pressure = st.number_input("Blood Pressure", value=70)
            skin_thickness = st.number_input("Skin Thickness", value=20)
            
        with col2:
            insulin = st.number_input("Insulin", value=79)
            bmi = st.number_input("BMI", value=25.0)
            dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
            age = st.number_input("Age", value=30)
            
        # Prediction logic for Classification
        if st.button("Predict Outcome"):
            input_data = pd.DataFrame([{
                "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age
            }]).reindex(columns=columns, fill_value=0)
            
            # Scale input features and predict
            input_scaled = scaler.transform(input_data)
            prediction = selected_model.predict(input_scaled)[0]
            
            if prediction == 1:
                st.error("⚠️ **Prediction Result:** Diabetic Positive")
            else:
                st.success("✅ **Prediction Result:** Normal / Healthy")
                
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure the 'saved_models' folder exists in your VS Code project directory.")

# --- REGRESSION SECTION ---
elif problem_type == "Regression":
    st.header("💳 Insurance Expense Prediction (Regression)")
    
    try:
        scaler, columns, all_models = load_reg_artifacts()
        
        # Select Algorithm from sidebar
        model_choice = st.sidebar.selectbox("Select Algorithm", list(all_models.keys()))
        selected_model = all_models[model_choice]
        
        st.subheader(f"Enter Inputs (Model Selected: {model_choice})")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", value=30)
            sex = st.selectbox("Sex", ["male", "female"])
            bmi = st.number_input("BMI", value=25.0)
            
        with col2:
            children = st.number_input("Children", value=0)
            smoker = st.selectbox("Smoker", ["yes", "no"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
            
        # Prediction logic for Regression
        if st.button("Predict Insurance Cost"):
            input_data = pd.DataFrame([{
                "age": age, "sex": sex, "bmi": bmi,
                "children": children, "smoker": smoker, "region": region
            }])
            
            # Encode categorical inputs and scale features
            input_df = pd.get_dummies(input_data).reindex(columns=columns, fill_value=0)
            input_scaled = scaler.transform(input_df)
            
            prediction = selected_model.predict(input_scaled)[0]
            st.success(f"💰 **Predicted Insurance Cost:** ${prediction:,.2f}")
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure the 'saved_models' folder exists in your VS Code project directory.")