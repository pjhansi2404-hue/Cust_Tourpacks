import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# Define the Hugging Face model repository ID
HF_MODEL_REPO_ID = "pjhansi2404/Cust_Tourpacks_Model"
MODEL_FILENAME = "logistic_regression_model.pkl"

@st.cache_resource
def load_model():
    """Downloads the model from Hugging Face Hub and loads it."""
    try:
        # Download the model file from Hugging Face Hub
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")
st.title("Wellness Tourism Package Purchase Prediction")

st.markdown("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Input fields for features
with st.form("prediction_form"):
    st.header("Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=30)
        typeofcontact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Large Business'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=10, value=1)
    
    with col2:
        preferredpropertystar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
        maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=1)
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        owncar = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

    with col3:
        designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director'])
        monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)
        pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        productpitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
        numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Map categorical inputs to numerical values (ensure consistent encoding with training)
        # Assuming 'Company Invited': 0, 'Self Inquiry': 1 based on alphabetical order for LabelEncoder
        typeofcontact_encoded = 0 if typeofcontact == 'Company Invited' else 1

        # Create a DataFrame for prediction
        input_df = pd.DataFrame({
            'Age': [age],
            'Type': [typeofcontact_encoded], # Using 'Type' as in prep.py
            'CityTier': [citytier],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [numberofpersonvisiting],
            'PreferredPropertyStar': [preferredpropertystar],
            'MaritalStatus': [maritalstatus],
            'NumberOfTrips': [numberoftrips],
            'Passport': [passport],
            'OwnCar': [owncar],
            'NumberOfChildrenVisiting': [numberofchildrenvisiting],
            'Designation': [designation],
            'MonthlyIncome': [monthlyincome],
            'PitchSatisfactionScore': [pitchsatisfactionscore],
            'ProductPitched': [productpitched],
            'NumberOfFollowups': [numberoffollowups],
            'DurationOfPitch': [durationofpitch]
        })

        # One-hot encode categorical features similar to how they were handled during training
        # Note: This is a simplified approach. In a real scenario, you'd save the fitted encoder/pipeline.
        categorical_cols = [
            'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched'
        ]
        
        # For consistent columns, create dummy columns for all possible categories seen during training
        # This is a critical step to ensure the app's input DataFrame has the same columns as the training data
        # A robust solution would involve saving the ColumnTransformer or OneHotEncoder fitted on training data
        # For this example, we'll manually create a comprehensive set of columns.
        all_possible_occupations = ['Salaried', 'Freelancer', 'Small Business', 'Large Business']
        all_possible_genders = ['Male', 'Female']
        all_possible_marital_statuses = ['Single', 'Married', 'Divorced']
        all_possible_designations = ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director']
        all_possible_product_pitched = ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King']

        # Create dummy variables for all categories, then select only the ones present in our input_df
        for col in categorical_cols:
            if col == 'Occupation':
                for occ in all_possible_occupations:
                    input_df[f'{col}_{occ}'] = (input_df[col] == occ).astype(int)
            elif col == 'Gender':
                for gen in all_possible_genders:
                    input_df[f'{col}_{gen}'] = (input_df[col] == gen).astype(int)
            elif col == 'MaritalStatus':
                for ms in all_possible_marital_statuses:
                    input_df[f'{col}_{ms}'] = (input_df[col] == ms).astype(int)
            elif col == 'Designation':
                for des in all_possible_designations:
                    input_df[f'{col}_{des}'] = (input_df[col] == des).astype(int)
            elif col == 'ProductPitched':
                for pp in all_possible_product_pitched:
                    input_df[f'{col}_{pp}'] = (input_df[col] == pp).astype(int)
        
        input_df = input_df.drop(columns=categorical_cols)

        # Ensure all columns expected by the model are present, fill missing with 0 if necessary
        # This is a placeholder. A robust solution would involve inspecting the model's expected features.
        # For demonstration, we assume the model expects these specific one-hot encoded columns.
        # In a real MLOps pipeline, the feature names from training should be explicitly saved.
        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [
            'Age', 'Type', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
            'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
            'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch',
            'Occupation_Freelancer', 'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business',
            'Gender_Female', 'Gender_Male',
            'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',
            'Designation_AVP', 'Designation_Director', 'Designation_Executive', 'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP',
            'ProductPitched_Basic', 'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe'
        ]
        
        # Align columns of input_df with expected_columns from training data
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]

        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)[:, 1]

            st.subheader("Prediction Results")
            if prediction[0] == 1:
                st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba[0]:.2f})")
            else:
                st.info(f"The customer is not likely to purchase the package. (Probability: {prediction_proba[0]:.2f})")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


st.markdown("---")
st.markdown("Developed by Your Name/Team for Visit with Us")
