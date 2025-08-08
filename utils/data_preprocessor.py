"""
Data preprocessing utilities for African Natural Disaster Prediction
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Config, REGIONAL_PROFILES

class DisasterDataPreprocessor:
    """
    Comprehensive data preprocessing for African disaster prediction
    Handles data cleaning, feature engineering, and encoding
    """

    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.region_encoder = LabelEncoder()
        self.disaster_type_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format=self.config.LOG_FORMAT
        )
        self.logger = logging.getLogger(__name__)

    def load_and_clean_data(self, file_path=None):
        """
        Load and perform comprehensive cleaning of disaster data

        Args:
            file_path (str): Path to CSV file, defaults to config path

        Returns:
            pd.DataFrame: Cleaned disaster data
        """
        if file_path is None:
            file_path = self.config.RAW_DATA_FILE

        try:
            self.logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} records")

            # Filter for African regions only
            african_data = df[df['Region'].isin(self.config.AFRICAN_REGIONS)]
            self.logger.info(f"Filtered to {len(african_data)} African records")

            # Clean and standardize column names
            african_data = self._standardize_columns(african_data)

            # Handle missing values and data types
            african_data = self._clean_data_types(african_data)

            # Remove extreme outliers
            african_data = self._remove_outliers(african_data)

            self.logger.info(f"Final cleaned dataset: {len(african_data)} records")
            return african_data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _standardize_columns(self, df):
        """Standardize column names and create consistent features"""
        # Create standardized column mappings
        column_mapping = {
            'Total Deaths': 'Total_Deaths',
            'No Injured': 'Number_Injured', 
            'No Affected': 'Number_Affected',
            'No Homeless': 'Number_Homeless',
            'Total Affected': 'Total_Affected',
            'Start Month': 'Start_Month',
            'Start Day': 'Start_Day',
            'Start Year': 'Start_Year',
            'End Month': 'End_Month',
            'End Year': 'End_Year',
            'Disaster Type': 'Disaster_Type',
            'Disaster Subtype': 'Disaster_Subtype'
        }

        # Apply column mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        return df

    def _clean_data_types(self, df):
        """Clean data types and handle missing values"""

        # Define numeric columns that should be converted
        numeric_columns = [
            'Year', 'Total_Deaths', 'Number_Injured', 'Number_Affected',
            'Number_Homeless', 'Total_Affected', 'Start_Month', 'Start_Day'
        ]

        # Convert to numeric, handling errors
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill missing values with appropriate defaults
        fill_values = {
            'Total_Deaths': 0,
            'Number_Injured': 0,
            'Number_Homeless': 0,
            'Start_Month': 6,  # Mid-year default
            'Start_Day': 15    # Mid-month default
        }

        for col, fill_val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)

        # Handle Total_Affected - use sum of components if missing
        if 'Total_Affected' in df.columns and 'Number_Affected' in df.columns:
            # Use Total_Affected if available, otherwise use Number_Affected
            df['Total_Affected'] = df['Total_Affected'].fillna(df['Number_Affected'])
        elif 'Number_Affected' in df.columns:
            df['Total_Affected'] = df['Number_Affected']

        # Remove records with missing target variable
        required_columns = ['Total_Affected', 'Year', 'Region']
        df = df.dropna(subset=required_columns)

        # Ensure Total_Affected is positive
        df = df[df['Total_Affected'] > 0]

        return df

    def _remove_outliers(self, df):
        """Remove extreme outliers using IQR method"""
        if 'Total_Affected' not in df.columns:
            return df

        # Calculate IQR for Total_Affected
        Q1 = df['Total_Affected'].quantile(0.25)
        Q3 = df['Total_Affected'].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds (using 3*IQR for more liberal filtering)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # Filter outliers
        before_count = len(df)
        df = df[(df['Total_Affected'] >= lower_bound) & 
                (df['Total_Affected'] <= upper_bound)]
        after_count = len(df)

        self.logger.info(f"Removed {before_count - after_count} outliers")
        return df

    def feature_engineering(self, df):
        """
        Create additional features for better prediction accuracy

        Args:
            df (pd.DataFrame): Cleaned disaster data

        Returns:
            pd.DataFrame: Data with engineered features
        """
        self.logger.info("Starting feature engineering")

        # Create severity index
        df['Severity_Index'] = (
            df.get('Total_Deaths', 0) + 
            df.get('Number_Injured', 0) + 
            df.get('Number_Homeless', 0)
        ) / 3

        # Create decade feature
        df['Decade'] = (df['Year'] // 10) * 10

        # Seasonal features
        df['Start_Month'] = df.get('Start_Month', 6)
        df['Season'] = df['Start_Month'].apply(self._get_season)

        # Create regional risk multiplier
        df['Regional_Risk_Multiplier'] = df['Region'].map(
            {region: profile['risk_multiplier'] 
             for region, profile in REGIONAL_PROFILES.items()}
        ).fillna(1.0)

        # Monthly risk indicator
        df['High_Risk_Month'] = df.apply(
            lambda row: 1 if row['Start_Month'] in 
            REGIONAL_PROFILES.get(row['Region'], {}).get('high_risk_months', []) 
            else 0, axis=1
        )

        # Disaster frequency by region-year
        disaster_freq = df.groupby(['Region', 'Year']).size().reset_index(name='Annual_Disaster_Count')
        df = df.merge(disaster_freq, on=['Region', 'Year'], how='left')
        df['Annual_Disaster_Count'] = df['Annual_Disaster_Count'].fillna(1)

        # Population density proxy (based on historical averages)
        region_avg_affected = df.groupby('Region')['Total_Affected'].mean()
        df['Region_Avg_Impact'] = df['Region'].map(region_avg_affected)

        self.logger.info("Feature engineering completed")
        return df

    def _get_season(self, month):
        """Convert month to season"""
        if pd.isna(month):
            return 'Unknown'
        elif month in [12, 1, 2]:
            return 'Summer'  # Southern hemisphere summer
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'

    def encode_categorical_features(self, df):
        """
        Encode categorical features for machine learning

        Args:
            df (pd.DataFrame): DataFrame with categorical features

        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        self.logger.info("Encoding categorical features")

        # Fit and transform region encoder
        if 'Region' in df.columns:
            df['Region_Encoded'] = self.region_encoder.fit_transform(df['Region'])

        # Fit and transform disaster type encoder
        if 'Disaster_Type' in df.columns:
            # Handle missing disaster types
            df['Disaster_Type'] = df['Disaster_Type'].fillna('Unknown')
            df['Disaster_Type_Encoded'] = self.disaster_type_encoder.fit_transform(df['Disaster_Type'])
        else:
            # Create default disaster type encoding
            df['Disaster_Type'] = 'Unknown'
            df['Disaster_Type_Encoded'] = 0

        # Encode season
        season_mapping = {'Summer': 0, 'Autumn': 1, 'Winter': 2, 'Spring': 3, 'Unknown': 4}
        df['Season_Encoded'] = df.get('Season', 'Unknown').map(season_mapping).fillna(4)

        self.is_fitted = True
        return df

    def prepare_features(self, df):
        """
        Prepare final feature set for machine learning

        Args:
            df (pd.DataFrame): Processed DataFrame

        Returns:
            tuple: (X, y) features and target
        """
        # Define feature columns (expanded from original notebook)
        feature_columns = [
            'Year',
            'Total_Deaths', 
            'Number_Injured',
            'Number_Homeless',
            'Severity_Index',
            'Region_Encoded',
            'Disaster_Type_Encoded',
            'Start_Month',
            'Regional_Risk_Multiplier',
            'High_Risk_Month',
            'Annual_Disaster_Count',
            'Season_Encoded'
        ]

        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in df.columns:
                self.logger.warning(f"Feature column {col} missing, filling with default")
                if 'Encoded' in col:
                    df[col] = 0
                else:
                    df[col] = df[feature_columns].median().iloc[0] if len(df) > 0 else 0

        X = df[feature_columns].copy()
        y = df[self.config.TARGET_COLUMN].copy()

        # Store feature names for later use
        self.feature_names = feature_columns

        self.logger.info(f"Prepared features: {X.shape}, target: {y.shape}")
        return X, y

    def split_data(self, X, y, test_size=None, random_state=None):
        """
        Split data into training and testing sets

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test size ratio
            random_state (int): Random state for reproducibility

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.config.TEST_SIZE
        if random_state is None:
            random_state = self.config.RANDOM_STATE

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_encoders(self, filepath=None):
        """Save fitted encoders for later use"""
        if not self.is_fitted:
            raise ValueError("Encoders not fitted yet!")

        if filepath is None:
            filepath = self.config.ENCODERS_FILE

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        encoders = {
            'region_encoder': self.region_encoder,
            'disaster_type_encoder': self.disaster_type_encoder,
            'feature_names': self.feature_names
        }

        joblib.dump(encoders, filepath)
        self.logger.info(f"Encoders saved to {filepath}")

    def load_encoders(self, filepath=None):
        """Load previously fitted encoders"""
        if filepath is None:
            filepath = self.config.ENCODERS_FILE

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Encoders file not found: {filepath}")

        encoders = joblib.load(filepath)
        self.region_encoder = encoders['region_encoder']
        self.disaster_type_encoder = encoders['disaster_type_encoder'] 
        self.feature_names = encoders['feature_names']
        self.is_fitted = True

        self.logger.info(f"Encoders loaded from {filepath}")

    def process_full_pipeline(self, file_path=None, save_processed=True):
        """
        Run the complete preprocessing pipeline

        Args:
            file_path (str): Path to raw data file
            save_processed (bool): Whether to save processed data

        Returns:
            tuple: (X, y) processed features and target
        """
        self.logger.info("Starting full preprocessing pipeline")

        # Load and clean data
        df = self.load_and_clean_data(file_path)

        # Feature engineering
        df = self.feature_engineering(df)

        # Encode categorical features
        df = self.encode_categorical_features(df)

        # Prepare features and target
        X, y = self.prepare_features(df)

        # Save processed data if requested
        if save_processed:
            processed_path = self.config.PROCESSED_DATA_FILE
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            df.to_csv(processed_path, index=False)
            self.logger.info(f"Processed data saved to {processed_path}")

        # Save encoders
        self.save_encoders()

        self.logger.info("Preprocessing pipeline completed successfully")
        return X, y
