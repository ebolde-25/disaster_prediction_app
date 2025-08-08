"""
Configuration settings for the African Natural Disaster Prediction App
"""
import os

class Config:
    """Application configuration settings"""

    # Data paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    MODELS_DIR = os.path.join(PROJECT_DIR, "models")
    STATIC_DIR = os.path.join(PROJECT_DIR, "static")

    # Data files
    RAW_DATA_FILE = os.path.join(DATA_DIR, "cleanD_natural_disaster.csv")
    PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed", "processed_disaster_data.csv")
    MODEL_FILE = os.path.join(MODELS_DIR, "disaster_prediction_model.joblib")
    SCALER_FILE = os.path.join(MODELS_DIR, "feature_scaler.joblib")
    ENCODERS_FILE = os.path.join(MODELS_DIR, "label_encoders.joblib")

    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    # XGBoost hyperparameters (optimized from notebook analysis)
    XGBOOST_PARAMS = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0
    }

    # Target R² score from notebook
    EXPECTED_R2_SCORE = 0.59

    # African regions for the prediction system
    AFRICAN_REGIONS = [
        'Eastern Africa',
        'Western Africa', 
        'Middle Africa',
        'Northern Africa',
        'Southern Africa'
    ]

    # Common disaster types in Africa
    DISASTER_TYPES = [
        'Flood',
        'Drought', 
        'Epidemic',
        'Storm',
        'Earthquake',
        'Wildfire',
        'Landslide',
        'Volcanic activity',
        'Mass movement',
        'Extreme temperature'
    ]

    # Feature columns for model training
    FEATURE_COLUMNS = [
        'Year',
        'Total_Deaths',
        'Number_Injured', 
        'Number_Homeless',
        'Severity_Index',
        'Region_Encoded',
        'Disaster_Type_Encoded',
        'Start_Month'
    ]

    # Target column
    TARGET_COLUMN = 'Total_Affected'

    # Risk thresholds for categorization
    RISK_THRESHOLDS = {
        'low': 1000,
        'medium': 10000,
        'high': float('inf')
    }

    # Colors for risk categories
    RISK_COLORS = {
        'low': '#27ae60',
        'medium': '#f39c12', 
        'high': '#e74c3c'
    }

    # Streamlit configuration
    PAGE_CONFIG = {
        'page_title': 'African Natural Disaster Prediction',
        'page_icon': '🌍',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }

    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Regional disaster risk profiles (historical analysis)
REGIONAL_PROFILES = {
    'Eastern Africa': {
        'common_disasters': ['Drought', 'Flood', 'Epidemic'],
        'high_risk_months': [3, 4, 5, 10, 11, 12],
        'average_affected': 45000,
        'risk_multiplier': 1.2
    },
    'Western Africa': {
        'common_disasters': ['Flood', 'Epidemic', 'Drought'],
        'high_risk_months': [6, 7, 8, 9],
        'average_affected': 35000,
        'risk_multiplier': 1.0
    },
    'Middle Africa': {
        'common_disasters': ['Flood', 'Epidemic', 'Wildfire'],
        'high_risk_months': [5, 6, 7, 8, 9],
        'average_affected': 25000,
        'risk_multiplier': 0.8
    },
    'Northern Africa': {
        'common_disasters': ['Flood', 'Earthquake', 'Extreme temperature'],
        'high_risk_months': [1, 2, 3, 10, 11, 12],
        'average_affected': 20000,
        'risk_multiplier': 0.7
    },
    'Southern Africa': {
        'common_disasters': ['Drought', 'Flood', 'Storm'],
        'high_risk_months': [1, 2, 3, 11, 12],
        'average_affected': 30000,
        'risk_multiplier': 0.9
    }
}
