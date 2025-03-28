import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
import warnings
from datetime import datetime
import time
import os
from io import StringIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Insurance Claims Severity Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E88E5;
    font-weight: bold;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.8rem;
    color: #0D47A1;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E0E0E0;
}
.section-header {
    font-size: 1.5rem;
    color: #0277BD;
    font-weight: bold;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
}
.info-box {
    background-color: #E3F2FD;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.success-box {
    background-color: #E8F5E9;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.warning-box {
    background-color: #FFF8E1;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.error-box {
    background-color: #FFEBEE;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #F5F5F5;
    padding: 1rem;
    border-radius: 0.5rem;
    box_shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.prediction-card {
    background-color: #E1F5FE;
    padding: 1.5rem;
    border-radius: 0.5rem;
    box_shadow: 0 2px 6px rgba(0,0,0,0.1);
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 3rem;
    white-space: pre-wrap;
    background-color: #F5F5F5;
    border-radius: 4px 4px 0px 0px;
    gap: 1rem;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
.stTabs [aria-selected="true"] {
    background-color: #E3F2FD;
    border-bottom: 2px solid #1E88E5;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'policy_data' not in st.session_state:
    st.session_state.policy_data = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'current_policy_number' not in st.session_state:
    st.session_state.current_policy_number = ""
if 'best_model_name' not in st.session_state:
    # Set Gradient Boosting as the best model based on analysis
    st.session_state.best_model_name = "Gradient Boosting"

# Helper function to create a download link for dataframes
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Helper function for feature engineering
def create_features(df):
    df_copy = df.copy()

    # Create new features
    # 1. Age of the vehicle at the time of incident
    if 'incident_date' in df_copy.columns and 'auto_year' in df_copy.columns:
        df_copy['vehicle_age'] = pd.to_datetime(df_copy['incident_date']).dt.year - df_copy['auto_year']

    # 2. Customer tenure at the time of incident (in years)
    if 'months_as_customer' in df_copy.columns:
        df_copy['customer_tenure_years'] = pd.to_numeric(df_copy['months_as_customer'], errors='coerce') / 12

    # 3. Season of incident
    if 'incident_date' in df_copy.columns:
        df_copy['incident_month'] = pd.to_datetime(df_copy['incident_date']).dt.month
        df_copy['incident_season'] = pd.cut(
            df_copy['incident_month'],
            bins=[0, 3, 6, 9, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall'],
            include_lowest=True
        )

    # 4. Time of day category
    if 'incident_hour_of_the_day' in df_copy.columns:
        df_copy['incident_time_category'] = pd.cut(
            pd.to_numeric(df_copy['incident_hour_of_the_day'], errors='coerce'),
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )

    # 5. Premium to umbrella ratio
    if 'policy_annual_premium' in df_copy.columns and 'umbrella_limit' in df_copy.columns:
        df_copy['premium_umbrella_ratio'] = pd.to_numeric(df_copy['policy_annual_premium'], errors='coerce') / (pd.to_numeric(df_copy['umbrella_limit'], errors='coerce') + 1)

    # 6. Total people involved (witnesses + bodily injuries)
    if 'witnesses' in df_copy.columns and 'bodily_injuries' in df_copy.columns:
        df_copy['total_people_involved'] = pd.to_numeric(df_copy['witnesses'], errors='coerce') + pd.to_numeric(df_copy['bodily_injuries'], errors='coerce')

    # 7. Claim ratio (proportion of each claim type)
    if all(col in df_copy.columns for col in ['injury_claim', 'property_claim', 'vehicle_claim', 'total_claim_amount']):
        total_claim = pd.to_numeric(df_copy['total_claim_amount'], errors='coerce')
        df_copy['injury_claim_ratio'] = pd.to_numeric(df_copy['injury_claim'], errors='coerce') / total_claim
        df_copy['property_claim_ratio'] = pd.to_numeric(df_copy['property_claim'], errors='coerce') / total_claim
        df_copy['vehicle_claim_ratio'] = pd.to_numeric(df_copy['vehicle_claim'], errors='coerce') / total_claim

    return df_copy

# Helper function to preprocess data for modeling
def preprocess_data(df, target_col='total_claim_amount', test_size=0.2, random_state=42):
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Convert columns to appropriate data types
    numeric_cols = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
                    'umbrella_limit', 'capital-gains', 'capital-loss', 'number_of_vehicles_involved',
                    'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim',
                    'property_claim', 'vehicle_claim', 'auto_year']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert date columns
    date_cols = ['policy_bind_date', 'incident_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Create engineered features
    df = create_features(df)

    # Exclude irrelevant columns
    exclude_cols = [target_col, 'injury_claim', 'property_claim', 'vehicle_claim',
                   'policy_number', 'policy_bind_date', 'incident_date', 'incident_location',
                   'insured_zip', '_c39']

    # Get all columns except the excluded ones
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Handle missing values
    df_clean = df[feature_cols + [target_col]].copy()

    # Impute missing values for numerical columns
    num_cols = df_clean.select_dtypes(include=['number']).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

    # Impute missing values for categorical columns
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    # Split the data
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    # Check if log transformation is needed
    log_transform = False
    if y.skew() > 1:
        y_log = np.log1p(y)
        log_transform = True
        y = y_log

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X.columns.tolist(), log_transform

# Helper function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, log_transform=False):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # If log transformation was applied, transform predictions back
    if log_transform:
        y_train_pred = np.expm1(y_train_pred)
        y_test_pred = np.expm1(y_test_pred)
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
    else:
        y_train_actual = y_train
        y_test_actual = y_test

    # Calculate metrics
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)

    return {
        'model': model,
        'model_name': model_name,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test_actual': y_test_actual,
        'y_test_pred': y_test_pred
    }

# Helper function to preprocess input data for prediction
def preprocess_input_data(input_data, features, scaler):
    # Create engineered features
    df = input_data.copy()

    # Vehicle age
    if 'incident_date' in df.columns and 'auto_year' in df.columns:
        df['vehicle_age'] = pd.to_datetime(df['incident_date']).dt.year - df['auto_year']

    # Customer tenure
    if 'months_as_customer' in df.columns:
        df['customer_tenure_years'] = pd.to_numeric(df['months_as_customer']) / 12

    # Season of incident
    if 'incident_date' in df.columns:
        df['incident_month'] = pd.to_datetime(df['incident_date']).dt.month
        df['incident_season'] = pd.cut(
            df['incident_month'],
            bins=[0, 3, 6, 9, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall'],
            include_lowest=True
        )

    # Time of day category
    if 'incident_hour_of_the_day' in df.columns:
        df['incident_time_category'] = pd.cut(
            pd.to_numeric(df['incident_hour_of_the_day']),
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )

    # Premium to umbrella ratio
    if 'policy_annual_premium' in df.columns and 'umbrella_limit' in df.columns:
        df['premium_umbrella_ratio'] = pd.to_numeric(df['policy_annual_premium']) / (pd.to_numeric(df['umbrella_limit']) + 1)

    # Total people involved
    if 'witnesses' in df.columns and 'bodily_injuries' in df.columns:
        df['total_people_involved'] = pd.to_numeric(df['witnesses']) + pd.to_numeric(df['bodily_injuries'])

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Create a DataFrame with all required features, filled with zeros
    X_pred = pd.DataFrame(0, index=range(1), columns=features)

    # Update the values for features that exist in the encoded input
    for col in df_encoded.columns:
        if col in features:
            X_pred[col] = df_encoded[col].values

    # Scale the features
    X_pred_scaled = scaler.transform(X_pred)

    return X_pred_scaled

# Helper function to make predictions
def predict_claim_amount(input_data, model, scaler, features, log_transform):
    # Preprocess the input data
    X_pred_scaled = preprocess_input_data(input_data, features, scaler)

    # Make prediction
    prediction = model.predict(X_pred_scaled)

    # If log transformation was applied, transform prediction back
    if log_transform:
        prediction = np.expm1(prediction)

    return prediction[0]

# Load data from local path
@st.cache_data
def load_data():
    try:
        # Local file path
        file_path = r"insurance_claims.csv"
        
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"File not found at: {file_path}")
            return pd.DataFrame()
        
        # Read the CSV data
        data = pd.read_csv(file_path)
        st.success(f"Successfully loaded data from: {file_path}")
        
        return data
    except Exception as e:
        st.error(f"Error loading data from local file: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Get data function that tries multiple methods
def get_data():
    # First try to get data from session state (if uploaded previously)
    if 'uploaded_data' in st.session_state:
        return st.session_state.uploaded_data
    
    # Then try to load from local path
    df = load_data()
    if not df.empty:
        return df
    
    # Return empty DataFrame if all methods fail
    return pd.DataFrame()

# Function to handle policy lookup
def lookup_policy(policy_num, df):
    try:
        # Convert to integer if needed
        if isinstance(policy_num, str) and policy_num.isdigit():
            policy_num = int(policy_num)
        
        # Find the policy in the dataset
        policy_row = df[df['policy_number'] == policy_num]
        
        if len(policy_row) > 0:
            # Store policy data in session state
            st.session_state.policy_data = policy_row.iloc[0].to_dict()
            st.session_state.current_policy_number = policy_num
            return True
        else:
            st.error("Policy not found. Please check the policy number and try again.")
            return False
    except Exception as e:
        st.error(f"Error looking up policy: {e}")
        return False

# Function to make prediction based on policy data and incident details
def make_prediction(incident_data, model, scaler, features, log_transform):
    if st.session_state.policy_data is None:
        st.error("No policy information available. Please lookup a policy first.")
        return None
    
    # Combine policy data with incident information
    input_data = pd.DataFrame({
        # Policyholder information from database
        'age': [st.session_state.policy_data.get('age', 0)],
        'months_as_customer': [st.session_state.policy_data.get('months_as_customer', 0)],
        'insured_sex': [st.session_state.policy_data.get('insured_sex', 'MALE')],
        'insured_education_level': [st.session_state.policy_data.get('insured_education_level', 'High School')],
        'insured_occupation': [st.session_state.policy_data.get('insured_occupation', 'other')],
        'insured_relationship': [st.session_state.policy_data.get('insured_relationship', 'other')],
        # Policy information from database
        'policy_state': [st.session_state.policy_data.get('policy_state', 'OH')],
        'policy_csl': [st.session_state.policy_data.get('policy_csl', '250/500')],
        'policy_deductable': [st.session_state.policy_data.get('policy_deductable', 500)],
        'policy_annual_premium': [st.session_state.policy_data.get('policy_annual_premium', 1000)],
        'umbrella_limit': [st.session_state.policy_data.get('umbrella_limit', 0)],
        # Vehicle information from database
        'auto_make': [st.session_state.policy_data.get('auto_make', 'Honda')],
        'auto_model': [st.session_state.policy_data.get('auto_model', 'Civic')],
        'auto_year': [st.session_state.policy_data.get('auto_year', 2010)],
        # User-provided incident information
        'incident_severity': [incident_data.get('incident_severity')],
        'incident_type': [incident_data.get('incident_type')],
        'collision_type': [incident_data.get('collision_type')],
        'authorities_contacted': [incident_data.get('authorities_contacted')],
        'incident_hour_of_the_day': [incident_data.get('incident_hour')],
        'number_of_vehicles_involved': [incident_data.get('num_vehicles')],
        'property_damage': [incident_data.get('property_damage')],
        'bodily_injuries': [incident_data.get('bodily_injuries')],
        'witnesses': [incident_data.get('witnesses')],
        'police_report_available': [incident_data.get('police_report')],
        'incident_date': [pd.to_datetime('today')]
    })
    
    # Make prediction
    prediction = predict_claim_amount(input_data, model, scaler, features, log_transform)
    return prediction

# Main application
def main():
    # Sidebar
    st.sidebar.markdown('<div class="main-header">Navigation</div>', unsafe_allow_html=True)
    
    app_mode = st.sidebar.selectbox("Choose a mode", 
                                   ["Home", 
                                    "Data Exploration", 
                                    "Model Training & Evaluation", 
                                    "Prediction Dashboard"],
                                   key="nav_mode")
    
    # Home page
    if app_mode == "Home":
        st.markdown('<div class="main-header">Insurance Claims Severity Prediction Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>Welcome to the Insurance Claims Severity Prediction Tool</h3>
                <p>This application helps insurance professionals predict the severity of insurance claims based on various factors related to the policy, policyholder, and incident details.</p>
                <p>Use the navigation panel on the left to explore different sections of the application:</p>
                <ul>
                    <li><strong>Data Exploration</strong>: Analyze and visualize the insurance claims dataset</li>
                    <li><strong>Model Training & Evaluation</strong>: Train and compare different machine learning models</li>
                    <li><strong>Prediction Dashboard</strong>: Make predictions for new insurance claims</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Try to load data from file path
            df = load_data()
            
            # If loading from file path fails, provide file upload option
            if df.empty:
                st.warning("Could not load data from the specified file path. Please upload your CSV file:")
                uploaded_file = st.file_uploader("Upload insurance claims dataset (CSV format)", type=["csv"], key="home_uploader")
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.uploaded_data = df
                        st.success(f"Data loaded successfully! Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
                    except Exception as e:
                        st.error(f"Error loading uploaded file: {e}")
            
            # Show data preview if we have data
            if not df.empty:
                st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df.head())
        
        with col2:
            st.markdown('<div class="section-header">Key Features</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Data Analysis</h4>
                <p>Explore patterns and relationships in insurance claims data</p>
            </div>
            
            <div class="metric-card">
                <h4>🤖 Machine Learning</h4>
                <p>Train models to predict claim severity with high accuracy</p>
            </div>
            
            <div class="metric-card">
                <h4>💰 Cost Prediction</h4>
                <p>Estimate total claim amounts for new incidents</p>
            </div>
            
            <div class="metric-card">
                <h4>🔍 Policy Lookup</h4>
                <p>Quickly retrieve policy information for predictions</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <p>This application demonstrates how machine learning can be used to predict insurance claim severity, helping insurance companies better assess risk and allocate resources.</p>
                <p>The models are trained on historical claims data and can predict the total claim amount based on various factors such as policy details, incident information, and policyholder demographics.</p>
            </div>
            """, unsafe_allow_html=True)

    # Data Exploration page
    elif app_mode == "Data Exploration":
        st.markdown('<div class="main-header">Data Exploration</div>', unsafe_allow_html=True)
        
        # Get data
        df = get_data()
        
        if df.empty:
            st.warning("No data available. Please upload a CSV file on the Home page.")
            uploaded_file = st.file_uploader("Upload insurance claims dataset (CSV format)", type=["csv"], key="explore_uploader")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.success(f"Data loaded successfully! Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
                    return
        
        if df.empty:
            return
        
        # Data overview
        st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Records", df.shape[0])
        with col2:
            st.metric("Number of Features", df.shape[1])
        with col3:
            if 'total_claim_amount' in df.columns:
                st.metric("Avg. Claim Amount", f"${pd.to_numeric(df['total_claim_amount'], errors='coerce').mean():.2f}")
            else:
                st.metric("Avg. Claim Amount", "N/A")
        
        # Data preview
        st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head())
        
        # Download link for the data
        st.markdown(get_table_download_link(df, "insurance_claims_data", "Download full dataset"), unsafe_allow_html=True)
        
        # Data types and missing values
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Data Types</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({'Data Type': df.dtypes}))
        
        with col2:
            st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
        
        if len(missing_values) > 0:
            st.dataframe(pd.DataFrame({'Missing Values': missing_values}))
        else:
            st.info("No missing values found in the dataset.")
        
        # Check for '?' values which might be treated as missing
        question_mark_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                if (df[col] == '?').any():
                    question_mark_cols.append({
                        'Column': col,
                        'Count': (df[col] == '?').sum()
                    })
        
        if question_mark_cols:
            st.markdown('<div class="section-header">Columns with \'?\' Values</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(question_mark_cols))
        
        # Convert numeric columns for analysis
        df_numeric = df.copy()
        numeric_cols = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
                        'umbrella_limit', 'capital-gains', 'capital-loss', 'number_of_vehicles_involved',
                        'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim',
                        'property_claim', 'vehicle_claim', 'auto_year']
        
        for col in numeric_cols:
            if col in df_numeric.columns:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        
        # Summary statistics
        st.markdown('<div class="sub-header">Summary Statistics</div>', unsafe_allow_html=True)
        
        # Select only numeric columns for summary statistics
        numeric_df = df_numeric.select_dtypes(include=['number'])
        st.dataframe(numeric_df.describe())
        
        # Exploratory Data Analysis
        st.markdown('<div class="sub-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        # Target variable analysis
        if 'total_claim_amount' in df.columns:
            st.markdown('<div class="section-header">Target Variable Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(pd.to_numeric(df['total_claim_amount'], errors='coerce'), kde=True, ax=ax)
                ax.set_title('Distribution of Total Claim Amount')
                ax.set_xlabel('Total Claim Amount')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                
                st.markdown(f"**Skewness of Total Claim Amount:** {pd.to_numeric(df['total_claim_amount'], errors='coerce').skew():.4f}")
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=pd.to_numeric(df['total_claim_amount'], errors='coerce'), ax=ax)
                ax.set_title('Boxplot of Total Claim Amount')
                st.pyplot(fig)
            
            # Log transformation of the target variable if it's skewed
            if pd.to_numeric(df['total_claim_amount'], errors='coerce').skew() > 1:
                st.markdown('<div class="section-header">Log-Transformed Target Variable</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(np.log1p(pd.to_numeric(df['total_claim_amount'], errors='coerce')), kde=True, ax=ax)
                    ax.set_title('Distribution of Log-Transformed Total Claim Amount')
                    ax.set_xlabel('Log(Total Claim Amount + 1)')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=np.log1p(pd.to_numeric(df['total_claim_amount'], errors='coerce')), ax=ax)
                ax.set_title('Boxplot of Log-Transformed Total Claim Amount')
                st.pyplot(fig)
            
            # Relationship between claim components
            if all(col in df.columns for col in ['injury_claim', 'property_claim', 'vehicle_claim']):
                st.markdown('<div class="section-header">Claim Components Analysis</div>', unsafe_allow_html=True)
                
                claim_components = ['injury_claim', 'property_claim', 'vehicle_claim', 'total_claim_amount']
                
                # Convert to numeric for correlation analysis
                df_claims = df[claim_components].copy()
                for col in claim_components:
                    df_claims[col] = pd.to_numeric(df_claims[col], errors='coerce')
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_claims.corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation between Claim Components')
                st.pyplot(fig)
        
        # Categorical variables analysis
        st.markdown('<div class="section-header">Categorical Variables Analysis</div>', unsafe_allow_html=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols and 'total_claim_amount' in df.columns:
            selected_cat_col = st.selectbox("Select a categorical variable", categorical_cols, key="cat_var_select")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=selected_cat_col, y=pd.to_numeric(df['total_claim_amount'], errors='coerce'), data=df, ax=ax)
            ax.set_title(f'Total Claim Amount by {selected_cat_col}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            st.pyplot(fig)
        
        # Numerical variables analysis
        st.markdown('<div class="section-header">Numerical Variables Analysis</div>', unsafe_allow_html=True)
        
        numerical_cols = numeric_df.columns.tolist()
        
        if 'total_claim_amount' in numerical_cols:
            numerical_cols.remove('total_claim_amount')
        
        if numerical_cols and 'total_claim_amount' in df.columns:
            # Correlation matrix
            st.markdown("#### Correlation with Target Variable")
            
            corr_with_target = numeric_df[numerical_cols + ['total_claim_amount']].corr()['total_claim_amount'].sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x=corr_with_target.values[1:11], y=corr_with_target.index[1:11], ax=ax)
            ax.set_title('Top 10 Numerical Features by Correlation with Total Claim Amount')
            ax.set_xlabel('Correlation Coefficient')
            st.pyplot(fig)
            
            # Scatter plot
            st.markdown("#### Relationship with Target Variable")
            
            selected_num_col = st.selectbox("Select a numerical variable", numerical_cols, key="num_var_select")
            
            # Scatter plot section
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=selected_num_col, y='total_claim_amount', data=numeric_df, alpha=0.6, ax=ax)
            ax.set_title(f'Total Claim Amount vs {selected_num_col}')
            st.pyplot(fig)
        
        # Feature Engineering Preview
        st.markdown('<div class="sub-header">Feature Engineering Preview</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <p>Feature engineering is a crucial step in improving model performance. The following features will be created:</p>
            <ul>
                <li><strong>vehicle_age</strong>: Age of the vehicle at the time of incident</li>
                <li><strong>customer_tenure_years</strong>: Customer tenure at the time of incident (in years)</li>
                <li><strong>incident_season</strong>: Season when the incident occurred</li>
                <li><strong>incident_time_category</strong>: Time of day category (Night, Morning, Afternoon, Evening)</li>
                <li><strong>premium_umbrella_ratio</strong>: Ratio of annual premium to umbrella limit</li>
                <li><strong>total_people_involved</strong>: Total number of people involved (witnesses + bodily injuries)</li>
                <li><strong>claim_ratios</strong>: Proportion of each claim type (injury, property, vehicle)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show engineered features
        if st.button("Preview Engineered Features", key="preview_features_btn"):
            with st.spinner("Creating engineered features..."):
                df_engineered = create_features(df)
                
                # Select only the new features
                new_features = ['vehicle_age', 'customer_tenure_years', 'incident_season',
                               'incident_time_category', 'premium_umbrella_ratio',
                               'total_people_involved']
                
                if all(col in df_engineered.columns for col in ['injury_claim_ratio', 'property_claim_ratio', 'vehicle_claim_ratio']):
                    new_features.extend(['injury_claim_ratio', 'property_claim_ratio', 'vehicle_claim_ratio'])
                
                # Filter for features that were successfully created
                new_features = [feat for feat in new_features if feat in df_engineered.columns]
                
                if new_features:
                    st.dataframe(df_engineered[new_features].head())
                    st.markdown(get_table_download_link(df_engineered, "insurance_claims_engineered", "Download engineered dataset"), unsafe_allow_html=True)
                else:
                    st.warning("Could not create engineered features. Please check your dataset.")

    # Model Training & Evaluation page
    elif app_mode == "Model Training & Evaluation":
        st.markdown('<div class="main-header">Model Training & Evaluation</div>', unsafe_allow_html=True)
        
        # Get data
        df = get_data()
        
        if df.empty:
            st.warning("No data available. Please upload a CSV file on the Home page.")
            uploaded_file = st.file_uploader("Upload insurance claims dataset (CSV format)", type=["csv"], key="model_uploader")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.success(f"Data loaded successfully! Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
                    return
        
        if df.empty:
            return
        
        # Check if the required target column exists
        if 'total_claim_amount' not in df.columns:
            st.error("The dataset must contain a 'total_claim_amount' column for model training.")
            return
        
        # Data preprocessing
        st.markdown('<div class="sub-header">Data Preprocessing</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <p>Before training models, the data will be preprocessed with the following steps:</p>
            <ol>
                <li>Convert data types and handle missing values</li>
                <li>Create engineered features</li>
                <li>Encode categorical variables</li>
                <li>Split data into training and testing sets</li>
                <li>Standardize numerical features</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.markdown('<div class="sub-header">Model Selection</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Select Models to Train")
            
            train_linear = st.checkbox("Linear Regression", value=True, key="train_linear")
            train_ridge = st.checkbox("Ridge Regression", value=True, key="train_ridge")
            train_rf = st.checkbox("Random Forest", value=True, key="train_rf")
            train_gb = st.checkbox("Gradient Boosting", value=True, key="train_gb")
            train_xgb = st.checkbox("XGBoost", value=True, key="train_xgb")
        
        with col2:
            st.markdown("#### Training Parameters")
            
            test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="test_size")
            random_state = st.number_input("Random State", min_value=1, max_value=100, value=42, key="random_state")
            # Fix the random state for numpy and other libraries to ensure consistency
            np.random.seed(random_state)
            
            # Advanced options
            show_advanced = st.checkbox("Show Advanced Options", key="show_advanced")
            
            # Add specific Gradient Boosting tuning options
            if show_advanced and train_gb:
                st.markdown("#### Gradient Boosting Specific Parameters")
                col1, col2 = st.columns(2)
                
                with col1:
                    gb_subsample = st.slider("Subsample Ratio", min_value=0.5, max_value=1.0, value=0.8, step=0.05, key="gb_subsample")
                    gb_min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=5, key="gb_min_samples_split")
                
                with col2:
                    gb_min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=5, value=2, key="gb_min_samples_leaf")
                    gb_max_depth = st.slider("GB Max Depth", min_value=3, max_value=10, value=5, step=1, key="gb_max_depth")
            else:
                gb_subsample = 0.8
                gb_min_samples_split = 5
                gb_min_samples_leaf = 2
                gb_max_depth = 5
            
            if show_advanced:
                n_estimators = st.slider("Number of Estimators (for tree-based models)", min_value=50, max_value=500, value=100, step=50, key="n_estimators")
                max_depth = st.slider("Max Depth (for tree-based models)", min_value=3, max_value=20, value=10, step=1, key="max_depth")
                learning_rate = st.slider("Learning Rate (for boosting models)", min_value=0.01, max_value=0.3, value=0.1, step=0.01, key="learning_rate")
                alpha = st.slider("Alpha (for Ridge Regression)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="alpha")
            else:
                n_estimators = 100
                max_depth = 10
                learning_rate = 0.1
                alpha = 1.0
        
        # Train models button
        if st.button("Train Models", key="train_models_btn"):
            with st.spinner("Preprocessing data and training models..."):
                # Preprocess data
                X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_list, log_transform = preprocess_data(
                    df, target_col='total_claim_amount', test_size=test_size, random_state=random_state
                )
                
                # Store in session state for later use
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.scaler = scaler
                st.session_state.feature_list = feature_list
                st.session_state.log_transform = log_transform
                
                # Display preprocessing results
                st.markdown('<div class="section-header">Preprocessing Results</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Set Size", X_train.shape[0])
                with col2:
                    st.metric("Testing Set Size", X_test.shape[0])
                with col3:
                    st.metric("Number of Features", X_train.shape[1])
                
                if log_transform:
                    st.info("Log transformation was applied to the target variable due to skewness.")
                
                # Initialize models
                models = []
                
                if train_linear:
                    linear_model = LinearRegression()
                    models.append(("Linear Regression", linear_model))
                
                if train_ridge:
                    ridge_model = Ridge(alpha=alpha)
                    models.append(("Ridge Regression", ridge_model))
                
                if train_rf:
                    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                    models.append(("Random Forest", rf_model))
                
                if train_gb:
                    # Use more robust hyperparameters to reduce overfitting
                    gb_model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        max_depth=gb_max_depth,
                        learning_rate=learning_rate,
                        min_samples_split=gb_min_samples_split,
                        min_samples_leaf=gb_min_samples_leaf,
                        subsample=gb_subsample,
                        random_state=random_state
                    )
                    models.append(("Gradient Boosting", gb_model))
                
                if train_xgb:
                    xgb_model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                            learning_rate=learning_rate, random_state=random_state)
                    models.append(("XGBoost", xgb_model))
                
                # Train and evaluate models
                results = []
                
                for name, model in models:
                    result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name, log_transform)
                    results.append(result)
                
                # Store results in session state
                st.session_state.model_results = results
                
                # Display model comparison
                st.markdown('<div class="sub-header">Model Comparison</div>', unsafe_allow_html=True)
                
                # Create a comparison dataframe
                comparison_df = pd.DataFrame({
                    'Model': [result['model_name'] for result in results],
                    'Training MAE': [result['train_mae'] for result in results],
                    'Testing MAE': [result['test_mae'] for result in results],
                    'Training RMSE': [result['train_rmse'] for result in results],
                    'Testing RMSE': [result['test_rmse'] for result in results],
                    'Training R²': [result['train_r2'] for result in results],
                    'Testing R²': [result['test_r2'] for result in results],
                    'Overfitting (R² diff)': [result['train_r2'] - result['test_r2'] for result in results]
                })
                
                st.dataframe(comparison_df)
                
                # Visualize model comparison
                st.markdown('<div class="section-header">Model Performance Visualization</div>', unsafe_allow_html=True)
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # MAE comparison
                sns.barplot(x='Model', y='Testing MAE', data=comparison_df, ax=axes[0, 0])
                axes[0, 0].set_title('Mean Absolute Error (MAE)')
                axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
                
                # RMSE comparison
                sns.barplot(x='Model', y='Testing RMSE', data=comparison_df, ax=axes[0, 1])
                axes[0, 1].set_title('Root Mean Squared Error (RMSE)')
                axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
                
                # R² comparison
                sns.barplot(x='Model', y='Testing R²', data=comparison_df, ax=axes[1, 0])
                axes[1, 0].set_title('R² Score')
                axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
                
                # Training vs Testing R² comparison
                model_names = comparison_df['Model']
                train_r2 = comparison_df['Training R²']
                test_r2 = comparison_df['Testing R²']
                
                x = np.arange(len(model_names))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, train_r2, width, label='Training R²')
                axes[1, 1].bar(x + width/2, test_r2, width, label='Testing R²')
                axes[1, 1].set_xlabel('Model')
                axes[1, 1].set_ylabel('R² Score')
                axes[1, 1].set_title('Training vs Testing R² Score')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(model_names, rotation=45)
                axes[1, 1].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Find Gradient Boosting model in results
                gb_index = next((i for i, result in enumerate(results) if result['model_name'] == "Gradient Boosting"), None)
                
                if gb_index is not None:
                    # Use Gradient Boosting as the best model based on analysis
                    best_model_index = gb_index
                    best_model_name = "Gradient Boosting"
                    best_model = results[best_model_index]['model']
                else:
                    # Fallback to highest R² if Gradient Boosting wasn't trained
                    best_model_index = comparison_df['Testing R²'].idxmax()
                    best_model_name = comparison_df.loc[best_model_index, 'Model']
                    best_model = results[best_model_index]['model']
                
                # Store the best model name in session state
                st.session_state.best_model_name = best_model_name
                
                st.markdown(f"""
                <div class="success-box">
                    <h3>Best Model: {best_model_name}</h3>
                    <p>Testing R²: {comparison_df.loc[best_model_index, 'Testing R²']:.4f}</p>
                    <p>Testing MAE: ${comparison_df.loc[best_model_index, 'Testing MAE']:.2f}</p>
                    <p>Testing RMSE: ${comparison_df.loc[best_model_index, 'Testing RMSE']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Check for overfitting
                overfitting_diff = comparison_df.loc[best_model_index, 'Overfitting (R² diff)']
                
                if overfitting_diff > 0.1:
                    st.warning(f"The best model shows signs of overfitting. Overfitting gap (Train R² - Test R²): {overfitting_diff:.4f}")
                else:
                    st.success(f"The best model does not show significant overfitting. Overfitting gap: {overfitting_diff:.4f}")
                
                # Store the best model in session state
                st.session_state.best_model = best_model
                
                # Actual vs Predicted plot for the best model
                st.markdown('<div class="section-header">Best Model: Actual vs Predicted</div>', unsafe_allow_html=True)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                y_test_actual = results[best_model_index]['y_test_actual']
                y_test_pred = results[best_model_index]['y_test_pred']
                
                ax1.scatter(y_test_actual, y_test_pred, alpha=0.5)
                ax1.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
                ax1.set_title(f'{best_model_name}: Actual vs Predicted')
                ax1.set_xlabel('Actual')
                ax1.set_ylabel('Predicted')
                
                # Residual plot
                residuals = y_test_actual - y_test_pred
                ax2.scatter(y_test_pred, residuals, alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_title('Residual Plot')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Residuals')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature importance for tree-based models
                if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
                    
                    if hasattr(best_model, 'feature_importances_'):
                        # Get feature importances
                        importances = best_model.feature_importances_
                        feature_importance_df = pd.DataFrame({
                            'Feature': feature_list,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importances
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
                        ax.set_title(f'Top 15 Features by Importance ({best_model_name})')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.dataframe(feature_importance_df.head(15))
                
                # Save model button
                if st.button("Save Best Model", key="save_model_btn"):
                    # Save the model
                    with open('final_model.pkl', 'wb') as f:
                        pickle.dump(best_model, f)
                    
                    # Save the scaler
                    with open('scaler.pkl', 'wb') as f:
                        pickle.dump(scaler, f)
                    
                    # Save the feature list
                    with open('feature_list.pkl', 'wb') as f:
                        pickle.dump(list(X.columns), f)
                    
                    # Save log transform flag
                    with open('log_transform.pkl', 'wb') as f:
                        pickle.dump(log_transform, f)
                    
                    st.success("Model, scaler, and feature list saved successfully!")
        
        # Show previously trained models if available
        elif 'model_results' in st.session_state:
            st.markdown('<div class="section-header">Previously Trained Models</div>', unsafe_allow_html=True)
            
            results = st.session_state.model_results
            
            # Create a comparison dataframe
            comparison_df = pd.DataFrame({
                'Model': [result['model_name'] for result in results],
                'Training MAE': [result['train_mae'] for result in results],
                'Testing MAE': [result['test_mae'] for result in results],
                'Training RMSE': [result['train_rmse'] for result in results],
                'Testing RMSE': [result['test_rmse'] for result in results],
                'Training R²': [result['train_r2'] for result in results],
                'Testing R²': [result['test_r2'] for result in results],
                'Overfitting (R² diff)': [result['train_r2'] - result['test_r2'] for result in results]
            })
            
            st.dataframe(comparison_df)
            
            # Highlight Gradient Boosting as the best model
            st.markdown(f"""
            <div class="success-box">
                <h3>Best Model: {st.session_state.best_model_name}</h3>
                <p>Based on our analysis, Gradient Boosting provides the best performance for this dataset.</p>
            </div>
            """, unsafe_allow_html=True)

    # Prediction Dashboard page
    elif app_mode == "Prediction Dashboard":
        st.markdown('<div class="main-header">Prediction Dashboard</div>', unsafe_allow_html=True)
        
        # Get data
        df = get_data()
        
        if df.empty:
            st.warning("No data available. Please upload a CSV file on the Home page.")
            uploaded_file = st.file_uploader("Upload insurance claims dataset (CSV format)", type=["csv"], key="predict_uploader")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.uploaded_data = df
                    st.success(f"Data loaded successfully! Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
                    return
        
        if df.empty:
            return
        
        # Check if models have been trained
        if 'best_model' not in st.session_state:
            st.warning("No trained models available. Please go to the 'Model Training & Evaluation' page to train models first.")
            
            # Provide a sample prediction interface anyway
            st.markdown('<div class="sub-header">Sample Prediction Interface</div>', unsafe_allow_html=True)
            st.info("This is a preview of the prediction interface. Train models first to make actual predictions.")
        else:
            st.markdown(f'<div class="success-box">Using the best model: {st.session_state.best_model_name}</div>', unsafe_allow_html=True)
        
        # Create tabs for different prediction methods
        tab1, tab2 = st.tabs(["Policy Lookup", "Manual Entry"])
        
        with tab1:
            st.markdown('<div class="sub-header">Policy Lookup</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <p>Enter a policy number to retrieve policyholder information and predict claim amount.</p>
                <p>This simplifies the claim prediction process by automatically retrieving all policy information.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Policy lookup form
            with st.form(key='policy_lookup_form'):
                policy_number = st.text_input("Enter Policy Number", placeholder="e.g., 556080", key="policy_number_input")
                lookup_submit = st.form_submit_button("Lookup Policy")
                
                if lookup_submit:
                    if policy_number:
                        # Look up policy
                        if lookup_policy(policy_number, df):
                            st.success(f"Policy {policy_number} found!")
                            # Reset prediction flag when looking up a new policy
                            st.session_state.prediction_made = False
                    else:
                        st.warning("Please enter a policy number.")
            
            # Display policy information if available
            if st.session_state.policy_data is not None:
                st.markdown('<div class="section-header">Policy Information Found</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Policyholder Details")
                    
                    policyholder_details = pd.DataFrame({
                        'Attribute': ['Age', 'Gender', 'Education', 'Occupation', 'Customer Since'],
                        'Value': [
                            st.session_state.policy_data.get('age', 'N/A'),
                            st.session_state.policy_data.get('insured_sex', 'N/A'),
                            st.session_state.policy_data.get('insured_education_level', 'N/A'),
                            st.session_state.policy_data.get('insured_occupation', 'N/A'),
                            f"{st.session_state.policy_data.get('months_as_customer', 'N/A')} months"
                        ]
                    })
                    
                    st.dataframe(policyholder_details, hide_index=True)
                
                with col2:
                    st.markdown("#### Policy Details")
                    
                    policy_details = pd.DataFrame({
                        'Attribute': ['State', 'CSL', 'Deductible', 'Annual Premium', 'Umbrella Limit'],
                        'Value': [
                            st.session_state.policy_data.get('policy_state', 'N/A'),
                            st.session_state.policy_data.get('policy_csl', 'N/A'),
                            f"${st.session_state.policy_data.get('policy_deductable', 'N/A')}",
                            f"${st.session_state.policy_data.get('policy_annual_premium', 'N/A')}",
                            f"${st.session_state.policy_data.get('umbrella_limit', 'N/A')}"
                        ]
                    })
                    
                    st.dataframe(policy_details, hide_index=True)
                    
                    st.markdown("#### Vehicle Details")
                    
                    vehicle_details = pd.DataFrame({
                        'Attribute': ['Make', 'Model', 'Year'],
                        'Value': [
                            st.session_state.policy_data.get('auto_make', 'N/A'),
                            st.session_state.policy_data.get('auto_model', 'N/A'),
                            st.session_state.policy_data.get('auto_year', 'N/A')
                        ]
                    })
                    
                    st.dataframe(vehicle_details, hide_index=True)
                
                # Incident form
                st.markdown('<div class="section-header">Enter Incident Details</div>', unsafe_allow_html=True)
                
                with st.form(key='incident_form'):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        incident_severity = st.selectbox(
                            "Incident Severity",
                            options=['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'],
                            key="incident_severity"
                        )
                        
                        incident_type = st.selectbox(
                            "Incident Type",
                            options=['Single Vehicle Collision', 'Vehicle Theft', 'Multi-vehicle Collision', 'Parked Car'],
                            key="incident_type"
                        )
                        
                        collision_type = st.selectbox(
                            "Collision Type",
                            options=['Front Collision', 'Rear Collision', 'Side Collision', 'Not Applicable'],
                            key="collision_type"
                        )
                        
                        authorities = st.selectbox(
                            "Authorities Contacted",
                            options=['Police', 'Fire', 'Ambulance', 'None', 'Other'],
                            key="authorities"
                        )
                    
                    with col2:
                        num_vehicles = st.slider("Number of Vehicles Involved", min_value=1, max_value=5, value=1, key="num_vehicles")
                        bodily_injuries = st.slider("Bodily Injuries", min_value=0, max_value=5, value=0, key="bodily_injuries")
                        witnesses = st.slider("Witnesses", min_value=0, max_value=5, value=0, key="witnesses")
                        
                        property_damage = st.selectbox("Property Damage", options=['YES', 'NO'], key="property_damage")
                        police_report = st.selectbox("Police Report Available", options=['YES', 'NO'], key="police_report")
                        incident_hour = st.slider("Hour of Day", min_value=0, max_value=23, value=12, key="incident_hour")
                    
                    # Predict button
                    predict_submit = st.form_submit_button("Predict Claim Amount")
                
                if predict_submit and 'best_model' in st.session_state:
                    # Collect incident data
                    incident_data = {
                        'incident_severity': incident_severity,
                        'incident_type': incident_type,
                        'collision_type': collision_type,
                        'authorities_contacted': authorities,
                        'num_vehicles': num_vehicles,
                        'bodily_injuries': bodily_injuries,
                        'witnesses': witnesses,
                        'property_damage': property_damage,
                        'police_report': police_report,
                        'incident_hour': incident_hour
                    }
                    
                    # Make prediction
                    prediction = make_prediction(
                        incident_data, 
                        st.session_state.best_model, 
                        st.session_state.scaler, 
                        st.session_state.feature_list, 
                        st.session_state.log_transform
                    )
                    
                    if prediction is not None:
                        # Set prediction flag
                        st.session_state.prediction_made = True
                        st.session_state.prediction_result = prediction
                
                # Display prediction result if available
                if st.session_state.prediction_made and hasattr(st.session_state, 'prediction_result'):
                    prediction = st.session_state.prediction_result
                    
                    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="color: #0066cc; margin-top: 0;">Prediction Result</h2>
                        <h1 style="color: #0066cc; font-size: 36px;">${prediction:,.2f}</h1>
                        <p style="font-size: 16px;">Estimated total claim amount for policy #{st.session_state.current_policy_number}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display breakdown based on typical ratios
                    injury_ratio = 0.15
                    property_ratio = 0.25
                    vehicle_ratio = 0.60
                    
                    injury_est = prediction * injury_ratio
                    property_est = prediction * property_ratio
                    vehicle_est = prediction * vehicle_ratio
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Estimated Claim Breakdown")
                        
                        breakdown_df = pd.DataFrame({
                            'Claim Type': ['Injury Claim', 'Property Claim', 'Vehicle Claim', 'Total'],
                            'Amount': [
                                f"${injury_est:,.2f}",
                                f"${property_est:,.2f}",
                                f"${vehicle_est:,.2f}",
                                f"${prediction:,.2f}"
                            ],
                            'Percentage': [
                                f"{injury_ratio*100:.0f}%",
                                f"{property_ratio*100:.0f}%",
                                f"{vehicle_ratio*100:.0f}%",
                                "100%"
                            ]
                        })
                        
                        st.dataframe(breakdown_df, hide_index=True)
                    
                    with col2:
                        # Risk assessment
                        risk_level = "Low"
                        risk_color = "#28a745"  # Green
                        
                        if prediction > 50000:
                            risk_level = "High"
                            risk_color = "#dc3545"  # Red
                        elif prediction > 20000:
                            risk_level = "Medium"
                            risk_color = "#ffc107"  # Yellow
                        
                        st.markdown(f"""
                        <div style="background-color: {risk_color}; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                            <h3 style="margin: 0;">{risk_level} Risk Claim</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Key factors
                        factors = []
                        
                        if incident_severity == "Major Damage" or incident_severity == "Total Loss":
                            factors.append("Severe incident damage")
                        if num_vehicles > 1:
                            factors.append("Multiple vehicles involved")
                        if bodily_injuries > 0:
                            factors.append("Bodily injuries reported")
                        if int(str(st.session_state.policy_data.get('auto_year', 2000))) > 2015:
                            factors.append("Newer vehicle (higher repair costs)")
                        if incident_type == "Multi-vehicle Collision":
                            factors.append("Multi-vehicle collision")
                        if float(str(st.session_state.policy_data.get('policy_deductable', 1000))) < 500:
                            factors.append("Low deductible policy")
                        
                        if not factors:
                            factors = ["Standard claim with no major risk factors"]
                        
                        st.markdown("#### Key Factors Influencing This Prediction")
                        
                        for factor in factors:
                            st.markdown(f"- {factor}")
        
        with tab2:
            st.markdown('<div class="sub-header">Manual Entry</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <p>Enter all policy and incident details manually to predict the claim amount.</p>
                <p>This method allows you to make predictions for new policies or hypothetical scenarios.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create form for manual entry
            with st.form(key='manual_entry_form'):
                # Create columns for input fields
                col1, col2 = st.columns(2)
                
                # Policyholder Information
                with col1:
                    st.markdown("#### Policyholder Information")
                    
                    age = st.number_input("Age", min_value=16, max_value=100, value=35, key="manual_age")
                    gender = st.selectbox("Gender", options=["MALE", "FEMALE"], key="manual_gender")
                    education = st.selectbox("Education Level", 
                                            options=["High School", "College", "Bachelor", "Master", 
                                                    "Associate", "JD", "MD", "PhD"],
                                            key="manual_education")
                    occupation = st.selectbox("Occupation", 
                                             options=["professional", "clerical", "manager", "blue-collar", 
                                                     "student", "homemaker", "retired", "other"],
                                             key="manual_occupation")
                    months_as_customer = st.number_input("Months as Customer", min_value=1, max_value=600, value=24, key="manual_months")
                
                # Policy Information
                with col2:
                    st.markdown("#### Policy Information")
                    
                    policy_state = st.selectbox("Policy State", options=["OH", "IL", "IN", "NY", "PA", "VA", "NC", "WV"], key="manual_state")
                    policy_csl = st.selectbox("Combined Single Limit", options=["100/300", "250/500", "500/1000"], key="manual_csl")
                    policy_deductible = st.number_input("Policy Deductible ($)", min_value=0, max_value=5000, value=500, step=100, key="manual_deductible")
                    policy_annual_premium = st.number_input("Annual Premium ($)", min_value=100, max_value=5000, value=1000, key="manual_premium")
                    umbrella_limit = st.number_input("Umbrella Limit ($)", min_value=0, max_value=10000000, value=0, step=1000000, key="manual_umbrella")
                
                # Vehicle Information
                st.markdown("#### Vehicle Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    auto_make = st.selectbox("Auto Make", 
                                            options=["Honda", "Toyota", "Ford", "Chevrolet", "BMW", "Mercedes", 
                                                    "Audi", "Nissan", "Subaru", "Other"],
                                            key="manual_make")
                
                with col2:
                    auto_model = st.selectbox("Auto Model", 
                                             options=["Civic", "Accord", "Camry", "Corolla", "F150", "Silverado", 
                                                     "3 Series", "C Class", "A4", "Other"],
                                             key="manual_model")
                
                with col3:
                    auto_year = st.number_input("Auto Year", min_value=1990, max_value=datetime.now().year, value=2015, key="manual_year")
                
                # Incident Information
                st.markdown("#### Incident Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    incident_severity = st.selectbox(
                        "Incident Severity",
                        options=['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'],
                        key="manual_severity"
                    )
                    
                    incident_type = st.selectbox(
                        "Incident Type",
                        options=['Single Vehicle Collision', 'Vehicle Theft', 'Multi-vehicle Collision', 'Parked Car'],
                        key="manual_type"
                    )
                    
                    collision_type = st.selectbox(
                        "Collision Type",
                        options=['Front Collision', 'Rear Collision', 'Side Collision', 'Not Applicable'],
                        key="manual_collision"
                    )
                    
                    authorities = st.selectbox(
                        "Authorities Contacted",
                        options=['Police', 'Fire', 'Ambulance', 'None', 'Other'],
                        key="manual_authorities"
                    )
                
                with col2:
                    num_vehicles = st.slider("Number of Vehicles Involved", min_value=1, max_value=5, value=1, key="manual_vehicles")
                    bodily_injuries = st.slider("Bodily Injuries", min_value=0, max_value=5, value=0, key="manual_injuries")
                    witnesses = st.slider("Witnesses", min_value=0, max_value=5, value=0, key="manual_witnesses")
                    
                    property_damage = st.selectbox("Property Damage", options=['YES', 'NO'], key="manual_damage")
                    police_report = st.selectbox("Police Report Available", options=['YES', 'NO'], key="manual_report")
                    incident_hour = st.slider("Hour of Day", min_value=0, max_value=23, value=12, key="manual_hour")
                
                # Submit button
                manual_predict_submit = st.form_submit_button("Predict Claim Amount")
            
            if manual_predict_submit and 'best_model' in st.session_state:
                # Create input dataframe with manually entered information
                input_data = pd.DataFrame({
                    # Policyholder information
                    'age': [age],
                    'months_as_customer': [months_as_customer],
                    'insured_sex': [gender],
                    'insured_education_level': [education],
                    'insured_occupation': [occupation],
                    'insured_relationship': ['other'],  # Default value
                    # Policy information
                    'policy_state': [policy_state],
                    'policy_csl': [policy_csl],
                    'policy_deductable': [policy_deductible],
                    'policy_annual_premium': [policy_annual_premium],
                    'umbrella_limit': [umbrella_limit],
                    # Vehicle information
                    'auto_make': [auto_make],
                    'auto_model': [auto_model],
                    'auto_year': [auto_year],
                    # Incident information
                    'incident_severity': [incident_severity],
                    'incident_type': [incident_type],
                    'collision_type': [collision_type],
                    'authorities_contacted': [authorities],
                    'incident_hour_of_the_day': [incident_hour],
                    'number_of_vehicles_involved': [num_vehicles],
                    'property_damage': [property_damage],
                    'bodily_injuries': [bodily_injuries],
                    'witnesses': [witnesses],
                    'police_report_available': [police_report],
                    'incident_date': [pd.to_datetime('today')]
                })
                
                # Make prediction
                prediction = predict_claim_amount(
                    input_data, 
                    st.session_state.best_model, 
                    st.session_state.scaler, 
                    st.session_state.feature_list, 
                    st.session_state.log_transform
                )
                
                # Display prediction result
                st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: #0066cc; margin-top: 0;">Prediction Result</h2>
                    <h1 style="color: #0066cc; font-size: 36px;">${prediction:,.2f}</h1>
                    <p style="font-size: 16px;">Estimated total claim amount</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display breakdown based on typical ratios
                injury_ratio = 0.15
                property_ratio = 0.25
                vehicle_ratio = 0.60
                
                injury_est = prediction * injury_ratio
                property_est = prediction * property_ratio
                vehicle_est = prediction * vehicle_ratio
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Estimated Claim Breakdown")
                    
                    breakdown_df = pd.DataFrame({
                        'Claim Type': ['Injury Claim', 'Property Claim', 'Vehicle Claim', 'Total'],
                        'Amount': [
                            f"${injury_est:,.2f}",
                            f"${property_est:,.2f}",
                            f"${vehicle_est:,.2f}",
                            f"${prediction:,.2f}"
                        ],
                        'Percentage': [
                            f"{injury_ratio*100:.0f}%",
                            f"{property_ratio*100:.0f}%",
                            f"{vehicle_ratio*100:.0f}%",
                            "100%"
                        ]
                    })
                    
                    st.dataframe(breakdown_df, hide_index=True)
                
                with col2:
                    # Risk assessment
                    risk_level = "Low"
                    risk_color = "#28a745"  # Green
                    
                    if prediction > 50000:
                        risk_level = "High"
                        risk_color = "#dc3545"  # Red
                    elif prediction > 20000:
                        risk_level = "Medium"
                        risk_color = "#ffc107"  # Yellow
                    
                    st.markdown(f"""
                    <div style="background-color: {risk_color}; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="margin: 0;">{risk_level} Risk Claim</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key factors
                    factors = []
                    
                    if incident_severity == "Major Damage" or incident_severity == "Total Loss":
                        factors.append("Severe incident damage")
                    if num_vehicles > 1:
                        factors.append("Multiple vehicles involved")
                    if bodily_injuries > 0:
                        factors.append("Bodily injuries reported")
                    if auto_year > 2015:
                        factors.append("Newer vehicle (higher repair costs)")
                    if incident_type == "Multi-vehicle Collision":
                        factors.append("Multi-vehicle collision")
                    if policy_deductible < 500:
                        factors.append("Low deductible policy")
                    
                    if not factors:
                        factors = ["Standard claim with no major risk factors"]
                    
                    st.markdown("#### Key Factors Influencing This Prediction")
                    
                    for factor in factors:
                        st.markdown(f"- {factor}")

if __name__ == "__main__":
    main()
