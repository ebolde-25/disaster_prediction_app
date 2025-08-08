"""
XGBoost Model Training Module for African Natural Disaster Prediction
Implements the optimized hyperparameters from the research notebook
"""
import numpy as np
import pandas as pd
import joblib
import logging
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Config

class DisasterPredictor:
    """
    XGBoost-based disaster impact prediction model
    Uses optimized hyperparameters from notebook analysis
    """

    def __init__(self, config=None):
        self.config = config or Config()
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}
        self.is_trained = False

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format=self.config.LOG_FORMAT
        )
        self.logger = logging.getLogger(__name__)

    def create_model(self, custom_params=None):
        """
        Create XGBoost model with optimized hyperparameters

        Args:
            custom_params (dict): Optional custom parameters to override defaults

        Returns:
            XGBRegressor: Configured XGBoost model
        """
        # Start with optimized parameters from notebook
        params = self.config.XGBOOST_PARAMS.copy()

        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)

        self.logger.info(f"Creating XGBoost model with parameters: {params}")

        self.model = XGBRegressor(**params)
        return self.model

    def train_model(self, X, y, validation_split=True, cross_validate=True):
        """
        Train the XGBoost model with comprehensive evaluation

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            validation_split (bool): Whether to use train/validation split
            cross_validate (bool): Whether to perform cross-validation

        Returns:
            dict: Training metrics and results
        """
        self.logger.info("Starting model training")

        # Create model if not exists
        if self.model is None:
            self.create_model()

        # Log transform target to handle skewness (as done in notebook)
        y_log = np.log1p(y)
        self.logger.info("Applied log transformation to target variable")

        if validation_split:
            # Split data for validation
            X_train, X_test, y_train_log, y_test_log = train_test_split(
                X, y_log, 
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE
            )

            # Train model
            self.model.fit(X_train, y_train_log)

            # Make predictions
            y_pred_log = self.model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_test_original = np.expm1(y_test_log)

            # Calculate metrics
            mse = mean_squared_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            mae = mean_absolute_error(y_test_original, y_pred)
            rmse = np.sqrt(mse)

            # Store training metrics
            self.training_metrics = {
                'r2_score': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'training_date': datetime.now().isoformat()
            }

            self.logger.info(f"Model Performance - R²: {r2:.4f}, MSE: {mse:,.0f}, MAE: {mae:,.0f}")

        else:
            # Train on full dataset
            self.model.fit(X, y_log)
            self.logger.info("Model trained on full dataset")

        # Perform cross-validation if requested
        if cross_validate:
            cv_scores = cross_val_score(
                self.model, X, y_log, 
                cv=self.config.CV_FOLDS,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )

            cv_mean = -cv_scores.mean()
            cv_std = cv_scores.std()

            self.training_metrics['cv_rmse_mean'] = cv_mean
            self.training_metrics['cv_rmse_std'] = cv_std

            self.logger.info(f"Cross-validation RMSE: {cv_mean:.2f} (+/- {cv_std*2:.2f})")

        # Calculate feature importance
        self._calculate_feature_importance(X)

        self.is_trained = True
        self.logger.info("Model training completed successfully")

        return self.training_metrics

    def _calculate_feature_importance(self, X):
        """Calculate and store feature importance"""
        if self.model is None:
            return

        importance_scores = self.model.feature_importances_
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        self.logger.info("Feature importance calculated")

    def predict(self, X):
        """
        Make predictions with the trained model

        Args:
            X (pd.DataFrame or np.array): Features for prediction

        Returns:
            np.array: Predictions (inverse log-transformed)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        # Make prediction on log scale
        y_pred_log = self.model.predict(X)

        # Inverse transform to original scale
        y_pred = np.expm1(y_pred_log)

        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def predict_single(self, features_dict):
        """
        Make prediction for a single disaster scenario

        Args:
            features_dict (dict): Dictionary with feature values

        Returns:
            float: Predicted number of affected people
        """
        # Convert dictionary to DataFrame for consistency
        features_df = pd.DataFrame([features_dict])

        # Make prediction
        prediction = self.predict(features_df)[0]

        return max(0, int(prediction))  # Return as non-negative integer

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance on test data

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Calculate percentage errors
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        evaluation_metrics = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

        self.logger.info(f"Model Evaluation - R²: {r2:.4f}, RMSE: {rmse:,.0f}, MAPE: {mape:.2f}%")

        return evaluation_metrics

    def hyperparameter_tuning(self, X, y, param_grid=None):
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            param_grid (dict): Parameters to tune

        Returns:
            dict: Best parameters and cross-validation score
        """
        if param_grid is None:
            # Default parameter grid for tuning
            param_grid = {
                'n_estimators': [300, 500, 700],
                'learning_rate': [0.03, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }

        self.logger.info("Starting hyperparameter tuning")

        # Log transform target
        y_log = np.log1p(y)

        # Create base model
        base_model = XGBRegressor(
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid,
            cv=self.config.CV_FOLDS,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y_log)

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_

        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best CV score: {best_score:.4f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'grid_search': grid_search
        }

    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):
        """
        Plot feature importance

        Args:
            top_n (int): Number of top features to display
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Feature importance plot
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated yet!")

        plt.figure(figsize=figsize)

        # Get top N features
        top_features = self.feature_importance.head(top_n)

        # Create horizontal bar plot
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - XGBoost Model')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()

        return plt.gcf()

    def plot_predictions_vs_actual(self, X_test, y_test, figsize=(10, 8)):
        """
        Plot predictions vs actual values

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Scatter plot
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        # Make predictions
        y_pred = self.predict(X_test)

        plt.figure(figsize=figsize)

        # Create scatter plot
        plt.scatter(y_test, y_pred, alpha=0.6)

        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')

        # Calculate R²
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return plt.gcf()

    def save_model(self, filepath=None):
        """
        Save the trained model and associated data

        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        if filepath is None:
            filepath = self.config.MODEL_FILE

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'config_params': self.config.XGBOOST_PARAMS,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath=None):
        """
        Load a previously trained model

        Args:
            filepath (str): Path to the saved model
        """
        if filepath is None:
            filepath = self.config.MODEL_FILE

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load model data
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance')
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = model_data.get('is_trained', True)

        self.logger.info(f"Model loaded from {filepath}")

        # Log model performance if available
        if 'r2_score' in self.training_metrics:
            r2 = self.training_metrics['r2_score']
            self.logger.info(f"Loaded model R² score: {r2:.4f}")

    def get_model_summary(self):
        """
        Get a summary of the trained model

        Returns:
            dict: Model summary information
        """
        if not self.is_trained:
            return {"status": "Model not trained"}

        summary = {
            "model_type": "XGBoost Regressor",
            "training_status": "Trained",
            "hyperparameters": self.config.XGBOOST_PARAMS,
            "performance_metrics": self.training_metrics,
            "feature_count": len(self.feature_importance) if self.feature_importance is not None else "Unknown",
            "top_features": self.feature_importance.head(5).to_dict('records') if self.feature_importance is not None else []
        }

        return summary
