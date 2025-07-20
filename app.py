import streamlit as st
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import joblib

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor with XGBoost")

# Step 1: Load Data
data = pd.read_csv("car data.xls")

st.subheader("📂 Raw Dataset")
st.dataframe(data.head())

# Step 2: Preprocess
st.subheader("🔧 Data Preprocessing")
current_year = datetime.datetime.now().year
data['Age'] = current_year - data['Year']
data.drop('Year', axis=1, inplace=True)

Q1 = data['Selling_Price'].quantile(0.25)
Q3 = data['Selling_Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Selling_Price'] >= lower_bound) & (data['Selling_Price'] <= upper_bound)]

data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
data['Transmission'] = data['Transmission'].map({'Manual': 0, 'Automatic': 1})

st.write("✅ Cleaned and Encoded Data")
st.dataframe(data.head())

# Visualization
st.write("📊 Selling Price Distribution After Removing Outliers")
fig, ax = plt.subplots()
sns.boxplot(data['Selling_Price'], ax=ax)
st.pyplot(fig)

# Step 3: Model Loading and Evaluation
X = data.drop(['Car_Name', 'Selling_Price'], axis=1)
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def load_model():
    return joblib.load("car_price_predictor.pkl")

model = load_model()

y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Performance")
st.metric("R² Score", f"{r2*100:.2f} %")
st.metric("RMSE", f"{rmse:.2f} Lakhs")

# Step 4: Prediction Interface
st.subheader("🧮 Predict Car Selling Price")

with st.form("prediction_form"):
    Present_Price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0)
    Kms_Driven = st.number_input("Kms Driven", min_value=0, value=30000)
    Owner = st.selectbox("Owner (0 = First, 1 = Second, 2 = Third)", [0, 1, 2])
    Age = st.slider("Car Age (in years)", 0, 30, 5)
    Fuel_Type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    Seller_Type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    Transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    submit = st.form_submit_button("Predict Price")

    if submit:
        input_dict = {
            'Present_Price': Present_Price,
            'Kms_Driven': Kms_Driven,
            'Owner': Owner,
            'Age': Age,
            'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[Fuel_Type],
            'Seller_Type': {'Dealer': 0, 'Individual': 1}[Seller_Type],
            'Transmission': {'Manual': 0, 'Automatic': 1}[Transmission]
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[X.columns.tolist()]  # Ensure correct column order

        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")

