import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
# model_path = hf_hub_download(repo_id="siaese/tourism-package-prediction", filename="best_tourism_project_model_v1.joblib")

# Load the model
# model = joblib.load(model_path)
# model = joblib.load("best_tourism_project_model_v1.joblib")

# Try HF first, fallback to local
# try:
#     # from huggingface_hub import hf_hub_download
#     model_path = hf_hub_download(
#         repo_id="siaese/tourism-package-prediction", 
#         filename="best_tourism_project_model_v1.joblib",
#         # repo_type="dataset"
#     )
#     model = joblib.load(model_path)
#     st.sidebar.success("✅ Model loaded from Hugging Face")
# except:
#     # Fallback to local model
#     model = joblib.load("best_tourism_project_model_v1.joblib")
#     st.sidebar.info("📁 Model loaded locally")

# ✅ Download model from Hugging Face Hub (caches automatically)
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="siaese/tourism-package-prediction", 
            filename="best_tourism_project_model_v1.joblib"
            # repo_type omitted - works for model/dataset/space repos
        )
        model = joblib.load(model_path)
        st.sidebar.success("✅ Model loaded from Hugging Face Hub")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ HF Download failed: {str(e)}")
        st.sidebar.info("📁 Please place 'best_tourism_project_model_v1.joblib' locally")
        st.stop()

model = load_model()

# Streamlit UI for Tourism Package Prediction
st.title("🧳 Tourism Package Prediction App")
st.write("**Visit with Us** - Predict whether a customer will purchase a tourism package based on their profile.")
st.write("Kindly enter customer details below to get a personalized recommendation.")

# # Collect user input
# CreditScore = st.number_input("Credit Score (customer's credit score)", min_value=300, max_value=900, value=650)
# Geography = st.selectbox("Geography (country where the customer resides)", ["France", "Germany", "Spain"])
# Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
# Tenure = st.number_input("Tenure (number of years the customer has been with the bank)", value=12)
# Balance = st.number_input("Account Balance (customer’s account balance)", min_value=0.0, value=10000.0)
# NumOfProducts = st.number_input("Number of Products (number of products the customer has with the bank)", min_value=1, value=1)
# HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"])
# IsActiveMember = st.selectbox("Is Active Member?", ["Yes", "No"])
# EstimatedSalary = st.number_input("Estimated Salary (customer’s estimated salary)", min_value=0.0, value=50000.0)

# Collect user input based on your tourism dataset columns
Age = st.number_input("Age", min_value=18, max_value=80, value=35)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=20, value=3)
ProductPitched = st.selectbox("ProductPitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Previous Trips", min_value=0, max_value=20, value=1)
Passport = st.selectbox("Has Passport?", [0, 1])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Owns Car?", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input(
    "Monthly Income", 
    min_value=15000.0, 
    max_value=40000.0, 
    value=20993.0, 
    step=100.0,
    help="Typical range: 17k-35k based on dataset"
)

st.markdown("---")

# Convert categorical inputs to match model training
# input_data = pd.DataFrame([{
#     'CreditScore': CreditScore,
#     'Geography': Geography,
#     'Age': Age,
#     'Tenure': Tenure,
#     'Balance': Balance,
#     'NumOfProducts': NumOfProducts,
#     'HasCrCard': 1 if HasCrCard == "Yes" else 0,
#     'IsActiveMember': 1 if IsActiveMember == "Yes" else 0,
#     'EstimatedSalary': EstimatedSalary
# }])

# Construct input DataFrame matching model expected features
input_data = pd.DataFrame({
    'Age': [Age],
    'TypeofContact': [TypeofContact],
    'CityTier': [CityTier],
    'DurationOfPitch': [DurationOfPitch],
    'Occupation': [Occupation],
    'Gender': [Gender],
    'NumberOfPersonVisiting': [NumberOfPersonVisiting],
    'NumberOfFollowups': [NumberOfFollowups],
    'ProductPitched': [ProductPitched],
    'PreferredPropertyStar': [PreferredPropertyStar],
    'MaritalStatus': [MaritalStatus],
    'NumberOfTrips': [NumberOfTrips],
    'Passport': [Passport],
    'PitchSatisfactionScore': [PitchSatisfactionScore],
    'OwnCar': [OwnCar],
    'NumberOfChildrenVisiting': [NumberOfChildrenVisiting],
    'Designation': [Designation],
    'MonthlyIncome': [MonthlyIncome]
})

# Set the classification threshold
classification_threshold = 0.45

# Predict button
# if st.button("Predict"):
#     prediction_proba = model.predict_proba(input_data)[0, 1]
#     prediction = (prediction_proba >= classification_threshold).astype(int)
#     result = "churn" if prediction == 1 else "not churn"
#     st.write(f"Based on the information provided, the customer is likely to {result}.")
if st.button("🎯 Predict Package Purchase", type="primary"):
    try:
        prediction_proba = model.predict_proba(input_data)[0, 1]
        prediction = (prediction_proba >= classification_threshold).astype(int)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.metric("Purchase Likelihood", f"{prediction_proba:.1%}")
        with col2:
            if prediction == 1:
                st.success("✅ **Package will be PURCHASED!**")
                st.balloons()
            else:
                st.warning("❌ **Package unlikely to be purchased**")
        with col3:
            st.metric("Decision Threshold", f"{classification_threshold:.1%}")
        st.subheader("💡 Recommendation")
        if prediction == 1:
            st.info("✅ **Target this customer aggressively** with personalized offers")
        else:
            st.info("🔄 **Nurture with follow-ups** to improve conversion chances")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Make sure the model file is available and columns match training data.")

with st.expander("📊 Input Data Preview"):
    st.dataframe(input_data)

with st.sidebar:
    st.info("**Dataset Columns Used:**")
    st.write("- Age, MonthlyIncome, CityTier")
    st.write("- TypeofContact, Occupation, Gender")
    st.write("- NumberOfPersonVisiting, NumberOfFollowups")
    st.write("- PreferredPropertyStar, MaritalStatus")
    st.write("- NumberOfTrips, Passport, OwnCar")
    st.write("- PitchSatisfactionScore, NumberOfChildrenVisiting")
    st.write("- Designation, ProductPitched")
