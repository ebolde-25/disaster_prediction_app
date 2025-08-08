"""
Real-time Disaster Impact Prediction Module
Handles single predictions and batch processing for the Streamlit app
"""
import numpy as np
import pandas as pd
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Config, REGIONAL_PROFILES
from models.model_trainer import DisasterPredictor as BasePredictor

class RealTimePredictor:
    """
    Real-time prediction interface for the Streamlit application
    Handles feature preparation and risk assessment
    """

    def __init__(self, model_path=None):
        self.config = Config()
        self.base_predictor = BasePredictor()
        self.region_mappings = self._create_region_mappings()
        self.disaster_mappings = self._create_disaster_mappings()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format=self.config.LOG_FORMAT
        )
        self.logger = logging.getLogger(__name__)

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def _create_region_mappings(self):
        """Create region name to encoding mappings"""
        return {region: idx for idx, region in enumerate(self.config.AFRICAN_REGIONS)}

    def _create_disaster_mappings(self):
        """Create disaster type to encoding mappings"""
        return {disaster: idx for idx, disaster in enumerate(self.config.DISASTER_TYPES)}

    def load_model(self, model_path=None):
        """Load the trained prediction model"""
        try:
            self.base_predictor.load_model(model_path)
            self.logger.info("Prediction model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def prepare_features(self, input_data):
        """
        Prepare features from user input for prediction

        Args:
            input_data (dict): Raw input from user interface

        Returns:
            pd.DataFrame: Prepared features for model
        """
        # Extract and validate inputs
        region = input_data.get('region', 'Eastern Africa')
        disaster_type = input_data.get('disaster_type', 'Flood')
        year = input_data.get('year', 2024)
        total_deaths = input_data.get('total_deaths', 0)
        injured = input_data.get('injured', 0)
        homeless = input_data.get('homeless', 0)
        month = input_data.get('month', 6)

        # Calculate derived features
        severity_index = (total_deaths + injured + homeless) / 3

        # Get regional risk multiplier
        regional_profile = REGIONAL_PROFILES.get(region, {})
        regional_risk_multiplier = regional_profile.get('risk_multiplier', 1.0)

        # Check if it's a high-risk month for the region
        high_risk_months = regional_profile.get('high_risk_months', [])
        high_risk_month = 1 if month in high_risk_months else 0

        # Estimate annual disaster count (simplified)
        annual_disaster_count = 2.5  # Average based on historical data

        # Season encoding
        season_encoded = self._get_season_encoded(month)

        # Prepare feature dictionary
        features = {
            'Year': year,
            'Total_Deaths': total_deaths,
            'Number_Injured': injured,
            'Number_Homeless': homeless,
            'Severity_Index': severity_index,
            'Region_Encoded': self.region_mappings.get(region, 0),
            'Disaster_Type_Encoded': self.disaster_mappings.get(disaster_type, 0),
            'Start_Month': month,
            'Regional_Risk_Multiplier': regional_risk_multiplier,
            'High_Risk_Month': high_risk_month,
            'Annual_Disaster_Count': annual_disaster_count,
            'Season_Encoded': season_encoded
        }

        # Convert to DataFrame
        features_df = pd.DataFrame([features])

        self.logger.info(f"Prepared features for prediction: {region}, {disaster_type}, {year}")
        return features_df

    def _get_season_encoded(self, month):
        """Encode month to season (Southern Hemisphere)"""
        season_mapping = {
            'Summer': 0, 'Autumn': 1, 'Winter': 2, 'Spring': 3, 'Unknown': 4
        }

        if month in [12, 1, 2]:
            season = 'Summer'
        elif month in [3, 4, 5]:
            season = 'Autumn'
        elif month in [6, 7, 8]:
            season = 'Winter'
        else:
            season = 'Spring'

        return season_mapping[season]

    def predict_disaster_impact(self, input_data):
        """
        Make a disaster impact prediction

        Args:
            input_data (dict): User input data

        Returns:
            dict: Prediction results with risk assessment
        """
        if not self.base_predictor.is_trained:
            raise ValueError("Model not loaded or trained!")

        # Prepare features
        features = self.prepare_features(input_data)

        # Make prediction
        prediction = self.base_predictor.predict(features)[0]
        prediction = max(0, int(prediction))  # Ensure non-negative integer

        # Calculate confidence based on model performance
        r2_score = self.base_predictor.training_metrics.get('r2_score', 0.59)
        confidence = min(95, max(60, r2_score * 100))  # Convert to percentage

        # Assess risk level
        risk_assessment = self._assess_risk_level(prediction, input_data)

        # Get regional context
        regional_context = self._get_regional_context(input_data['region'])

        result = {
            'predicted_affected': prediction,
            'confidence_level': round(confidence, 1),
            'risk_level': risk_assessment['level'],
            'risk_color': risk_assessment['color'],
            'risk_description': risk_assessment['description'],
            'regional_context': regional_context,
            'model_info': {
                'r2_score': r2_score,
                'model_type': 'XGBoost Regressor'
            }
        }

        self.logger.info(f"Prediction completed: {prediction:,} affected people, {risk_assessment['level']} risk")
        return result

    def _assess_risk_level(self, prediction, input_data):
        """Assess risk level based on prediction and context"""
        region = input_data.get('region', 'Eastern Africa')

        # Base thresholds
        if prediction < self.config.RISK_THRESHOLDS['low']:
            level = 'Low'
            color = self.config.RISK_COLORS['low']
            description = "Limited impact expected. Standard preparedness recommended."
        elif prediction < self.config.RISK_THRESHOLDS['medium']:
            level = 'Medium'
            color = self.config.RISK_COLORS['medium'] 
            description = "Moderate impact expected. Enhanced preparedness and monitoring advised."
        else:
            level = 'High'
            color = self.config.RISK_COLORS['high']
            description = "Significant impact expected. Emergency preparedness and rapid response planning critical."

        # Adjust based on regional factors
        regional_profile = REGIONAL_PROFILES.get(region, {})
        if input_data.get('month') in regional_profile.get('high_risk_months', []):
            if level == 'Low':
                level = 'Medium'
                color = self.config.RISK_COLORS['medium']
                description = "Elevated risk due to high-risk season for this region."

        return {
            'level': level,
            'color': color,
            'description': description
        }

    def _get_regional_context(self, region):
        """Get regional disaster context and recommendations"""
        profile = REGIONAL_PROFILES.get(region, {})

        context = {
            'common_disasters': profile.get('common_disasters', []),
            'high_risk_months': profile.get('high_risk_months', []),
            'average_affected': profile.get('average_affected', 30000),
            'recommendations': self._get_regional_recommendations(region)
        }

        return context

    def _get_regional_recommendations(self, region):
        """Get region-specific preparedness recommendations"""
        recommendations = {
            'Eastern Africa': [
                "Monitor drought conditions and food security",
                "Strengthen flood early warning systems",
                "Maintain emergency food and water reserves",
                "Coordinate with regional health organizations for epidemic preparedness"
            ],
            'Western Africa': [
                "Prepare for seasonal flooding during rainy season",
                "Maintain health surveillance for epidemic diseases",
                "Coordinate cross-border disaster response",
                "Strengthen community-based early warning systems"
            ],
            'Middle Africa': [
                "Monitor forest fire conditions during dry season",
                "Prepare flood response for major river basins",
                "Maintain health emergency response capacity",
                "Coordinate regional disaster information sharing"
            ],
            'Northern Africa': [
                "Monitor seismic activity and earthquake preparedness",
                "Prepare for extreme temperature events",
                "Maintain flood response for coastal and urban areas",
                "Coordinate Mediterranean regional response mechanisms"
            ],
            'Southern Africa': [
                "Prepare for cyclone season and coastal hazards",
                "Monitor drought conditions and agricultural impacts",
                "Maintain cross-border coordination mechanisms",
                "Strengthen urban flood management systems"
            ]
        }

        return recommendations.get(region, [
            "Maintain general disaster preparedness",
            "Monitor regional hazard conditions",
            "Coordinate with neighboring countries",
            "Prepare emergency response resources"
        ])

    def batch_predict(self, input_list):
        """
        Make predictions for multiple disaster scenarios

        Args:
            input_list (list): List of input dictionaries

        Returns:
            list: List of prediction results
        """
        results = []

        for i, input_data in enumerate(input_list):
            try:
                result = self.predict_disaster_impact(input_data)
                result['scenario_id'] = i + 1
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error predicting scenario {i+1}: {e}")
                results.append({
                    'scenario_id': i + 1,
                    'error': str(e),
                    'predicted_affected': 0
                })

        return results

    def get_prediction_explanation(self, input_data, prediction_result):
        """
        Generate explanation for the prediction

        Args:
            input_data (dict): Original input data
            prediction_result (dict): Prediction results

        Returns:
            dict: Explanation of the prediction
        """
        explanation = {
            'key_factors': [],
            'regional_factors': [],
            'seasonal_factors': [],
            'model_confidence': prediction_result['confidence_level']
        }

        # Key factors analysis
        severity_index = (input_data.get('total_deaths', 0) + 
                         input_data.get('injured', 0) + 
                         input_data.get('homeless', 0)) / 3

        if severity_index > 1000:
            explanation['key_factors'].append("High severity index indicates major disaster impact")
        elif severity_index > 100:
            explanation['key_factors'].append("Moderate severity index suggests significant impact")
        else:
            explanation['key_factors'].append("Low severity index indicates limited direct casualties")

        # Regional factors
        region = input_data.get('region')
        regional_profile = REGIONAL_PROFILES.get(region, {})

        explanation['regional_factors'].append(f"Historical average for {region}: {regional_profile.get('average_affected', 30000):,} people")
        explanation['regional_factors'].append(f"Regional risk multiplier: {regional_profile.get('risk_multiplier', 1.0)}")

        # Seasonal factors
        month = input_data.get('month', 6)
        if month in regional_profile.get('high_risk_months', []):
            explanation['seasonal_factors'].append(f"High-risk month for {region}")
        else:
            explanation['seasonal_factors'].append(f"Standard risk month for {region}")

        return explanation

    def model_status(self):
        """Get current model status"""
        return {
            'is_loaded': self.base_predictor.is_trained,
            'model_performance': self.base_predictor.training_metrics,
            'feature_importance': self.base_predictor.feature_importance.head(10).to_dict('records') if self.base_predictor.feature_importance is not None else []
        }
