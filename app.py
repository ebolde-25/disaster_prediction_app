"""
African Natural Disaster Impact Prediction App
Main Streamlit application with interactive UI for disaster prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import logging
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from config.settings import Config, REGIONAL_PROFILES
    from utils.data_preprocessor import DisasterDataPreprocessor
    from models.model_trainer import DisasterPredictor
    from models.predictor import RealTimePredictor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure Streamlit
st.set_page_config(**Config.PAGE_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f4fd 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-result {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #27ae60 !important;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #fde2a9 100%);
        border-left-color: #f39c12 !important;
    }
    .risk-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f1c6cb 100%);
        border-left-color: #e74c3c !important;
    }
    .sidebar-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DisasterPredictionApp:
    """Main application class for the disaster prediction system"""

    def __init__(self):
        self.config = Config()
        self.predictor = RealTimePredictor()
        self.preprocessor = DisasterDataPreprocessor()
        self.load_or_train_model()

    def load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            # Try to load existing model
            if os.path.exists(self.config.MODEL_FILE):
                success = self.predictor.load_model(self.config.MODEL_FILE)
                if success:
                    st.session_state.model_status = "loaded"
                    return

            # If no model exists, train a new one
            st.session_state.model_status = "training"
            self.train_new_model()

        except Exception as e:
            st.session_state.model_status = "error"
            st.session_state.model_error = str(e)

    def train_new_model(self):
        """Train a new prediction model"""
        with st.spinner("🤖 Training disaster prediction model... This may take a few minutes."):
            try:
                # Process data
                X, y = self.preprocessor.process_full_pipeline()

                # Train model
                trainer = DisasterPredictor()
                metrics = trainer.train_model(X, y)

                # Save model
                trainer.save_model()

                # Load into predictor
                self.predictor.load_model()

                st.session_state.model_status = "trained"
                st.session_state.model_metrics = metrics

                st.success(f"✅ Model trained successfully! R² Score: {metrics.get('r2_score', 0):.3f}")

            except Exception as e:
                st.session_state.model_status = "error"
                st.session_state.model_error = str(e)
                st.error(f"Training failed: {e}")

    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🌍 African Natural Disaster Impact Predictor</h1>', 
                   unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 2rem; 
                    background: linear-gradient(90deg, #f8f9fa, #e9ecef); padding: 1rem; border-radius: 10px;">
            🎯 Predicting disaster impacts across African regions using advanced XGBoost machine learning<br>
            📊 R² Score: 0.59 | 🔬 Model: XGBoost Regressor | 📈 Features: 12 predictive factors
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render sidebar with prediction controls"""
        st.sidebar.markdown('<div class="sidebar-header">🎯 Disaster Prediction Parameters</div>', 
                           unsafe_allow_html=True)

        # Region selection
        region = st.sidebar.selectbox(
            "🌍 Select African Region",
            self.config.AFRICAN_REGIONS,
            help="Choose the African region for disaster prediction"
        )

        # Disaster type
        disaster_type = st.sidebar.selectbox(
            "⚡ Disaster Type",
            self.config.DISASTER_TYPES,
            help="Select the type of natural disaster"
        )

        # Time parameters
        col1, col2 = st.sidebar.columns(2)
        with col1:
            year = st.number_input("📅 Year", 2020, 2030, 2024)
        with col2:
            month = st.slider("📆 Month", 1, 12, 6)

        # Impact metrics
        st.sidebar.subheader("📊 Expected Impact Metrics")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            total_deaths = st.number_input("💀 Expected Deaths", 0, 10000, 50, step=10)
            injured = st.number_input("🤕 Expected Injuries", 0, 50000, 200, step=50)
        with col2:
            homeless = st.number_input("🏠 Expected Homeless", 0, 100000, 1000, step=100)

        # Regional information
        if region in REGIONAL_PROFILES:
            profile = REGIONAL_PROFILES[region]
            st.sidebar.info(f"""
            **{region} Profile:**
            - Common disasters: {', '.join(profile['common_disasters'][:2])}
            - Risk multiplier: {profile['risk_multiplier']}
            - High-risk months: {', '.join(map(str, profile['high_risk_months'][:4]))}
            """)

        return {
            'region': region,
            'disaster_type': disaster_type,
            'year': year,
            'month': month,
            'total_deaths': total_deaths,
            'injured': injured,
            'homeless': homeless
        }

    def make_prediction(self, inputs):
        """Make disaster impact prediction"""
        try:
            result = self.predictor.predict_disaster_impact(inputs)
            return result
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

    def render_prediction_section(self, inputs):
        """Render prediction interface and results"""
        col1, col2 = st.columns([3, 1])

        with col1:
            predict_button = st.button(
                "🎯 Predict Disaster Impact", 
                type="primary", 
                use_container_width=True,
                help="Generate prediction based on input parameters"
            )

            if predict_button:
                if st.session_state.get('model_status') == 'error':
                    st.error(f"Model error: {st.session_state.get('model_error', 'Unknown error')}")
                    return

                with st.spinner("🔮 Generating prediction..."):
                    result = self.make_prediction(inputs)

                    if result:
                        # Store result in session state
                        st.session_state.last_prediction = result
                        st.session_state.last_inputs = inputs

                        # Determine risk class for styling
                        risk_class = f"risk-{result['risk_level'].lower()}"

                        st.markdown(f"""
                        <div class="prediction-result {risk_class}">
                            <h2 style="margin-bottom: 1rem;">📈 Prediction Results</h2>
                            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                <h1 style="color: {result['risk_color']}; margin: 0; font-size: 2.5rem;">
                                    {result['predicted_affected']:,} people
                                </h1>
                                <span style="margin-left: 1rem; font-size: 1.2rem; opacity: 0.8;">
                                    expected to be affected
                                </span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                                <div><strong>Region:</strong> {inputs['region']}</div>
                                <div><strong>Disaster:</strong> {inputs['disaster_type']}</div>
                                <div><strong>Year:</strong> {inputs['year']}</div>
                                <div><strong>Confidence:</strong> {result['confidence_level']}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Risk assessment
                        risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[result['risk_level']]

                        st.markdown(f"""
                        <div style="background: {result['risk_color']}20; padding: 1.5rem; 
                                    border-radius: 15px; margin: 1rem 0; border-left: 5px solid {result['risk_color']};">
                            <h3 style="color: {result['risk_color']}; margin-bottom: 0.5rem;">
                                {risk_icon} {result['risk_level']} Risk Level
                            </h3>
                            <p style="margin: 0; font-size: 1.1rem;">{result['risk_description']}</p>
                        </div>
                        """, unsafe_allow_html=True)

        with col2:
            # Input summary
            st.subheader("📋 Input Summary")
            summary_data = {
                "Region": inputs['region'],
                "Disaster": inputs['disaster_type'],
                "Year": inputs['year'],
                "Month": inputs['month'],
                "Deaths": inputs['total_deaths'],
                "Injured": inputs['injured'],
                "Homeless": inputs['homeless']
            }

            for key, value in summary_data.items():
                st.metric(key, value)

    def render_analytics_dashboard(self):
        """Render analytics and visualizations"""
        st.header("📊 Analytics Dashboard")

        try:
            # Try to load processed data
            if os.path.exists(self.config.PROCESSED_DATA_FILE):
                df = pd.read_csv(self.config.PROCESSED_DATA_FILE)
            else:
                # Load and process raw data
                df = self.preprocessor.load_and_clean_data()
                df = self.preprocessor.feature_engineering(df)

            if len(df) == 0:
                st.warning("No data available for analytics")
                return

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_disasters = len(df)
                st.metric("📈 Total Disasters", f"{total_disasters:,}")

            with col2:
                total_affected = df['Total_Affected'].sum()
                st.metric("👥 Total Affected", f"{total_affected:,.0f}")

            with col3:
                avg_affected = df['Total_Affected'].mean()
                st.metric("📊 Avg. per Disaster", f"{avg_affected:,.0f}")

            with col4:
                regions_covered = df['Region'].nunique()
                st.metric("🌍 Regions Covered", regions_covered)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Regional impact analysis
                regional_data = df.groupby('Region')['Total_Affected'].agg(['sum', 'count', 'mean']).reset_index()
                regional_data.columns = ['Region', 'Total_Affected', 'Disaster_Count', 'Avg_Affected']

                fig1 = px.bar(
                    regional_data, 
                    x='Region', 
                    y='Total_Affected',
                    title="Total People Affected by African Region",
                    color='Total_Affected',
                    color_continuous_scale='Reds',
                    text='Total_Affected'
                )
                fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig1.update_xaxes(tickangle=45)
                fig1.update_layout(height=500)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Temporal trends
                if 'Year' in df.columns:
                    yearly_data = df.groupby('Year')['Total_Affected'].agg(['sum', 'count']).reset_index()
                    yearly_data.columns = ['Year', 'Total_Affected', 'Disaster_Count']

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=yearly_data['Year'],
                        y=yearly_data['Total_Affected'],
                        mode='lines+markers',
                        name='Total Affected',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))

                    fig2.update_layout(
                        title="Disaster Impact Trends Over Time",
                        xaxis_title="Year",
                        yaxis_title="Total People Affected",
                        height=500
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Disaster type analysis
            if 'Disaster_Type' in df.columns:
                disaster_stats = df.groupby('Disaster_Type').agg({
                    'Total_Affected': ['sum', 'count', 'mean'],
                    'Total_Deaths': 'sum'
                }).round(0)

                disaster_stats.columns = ['Total_Affected', 'Count', 'Avg_Affected', 'Total_Deaths']
                disaster_stats = disaster_stats.reset_index().sort_values('Total_Affected', ascending=False)

                fig3 = px.treemap(
                    disaster_stats.head(10),
                    path=['Disaster_Type'],
                    values='Total_Affected',
                    color='Total_Deaths',
                    color_continuous_scale='Viridis',
                    title="Disaster Types by Impact (Top 10)"
                )
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)

            # Monthly risk analysis
            if 'Start_Month' in df.columns:
                monthly_data = df.groupby('Start_Month')['Total_Affected'].agg(['sum', 'count', 'mean']).reset_index()
                monthly_data.columns = ['Month', 'Total_Affected', 'Count', 'Avg_Affected']

                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_data['Month_Name'] = monthly_data['Month'].apply(lambda x: month_names[int(x)-1] if 1 <= x <= 12 else 'Unknown')

                fig4 = px.bar(
                    monthly_data,
                    x='Month_Name',
                    y='Count',
                    title="Disaster Frequency by Month",
                    color='Avg_Affected',
                    color_continuous_scale='Blues'
                )
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading analytics data: {e}")
            st.info("Analytics will be available after the model is trained with data")

    def render_model_performance(self):
        """Render model performance metrics and information"""
        st.header("🤖 Model Performance & Information")

        # Model status
        model_status = st.session_state.get('model_status', 'unknown')

        if model_status == 'error':
            st.error(f"Model Error: {st.session_state.get('model_error', 'Unknown error')}")
            if st.button("🔄 Retry Model Training"):
                st.session_state.model_status = 'training'
                st.experimental_rerun()
            return

        # Performance metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-container">
                <h4>🎯 Model Accuracy</h4>
                <h2 style="color: #1f77b4;">R² = 0.59</h2>
                <p>Explains 59% of variance in disaster impact</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-container">
                <h4>🧠 Algorithm</h4>
                <h3 style="color: #2ca02c;">XGBoost</h3>
                <p>Gradient Boosting with optimized hyperparameters</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-container">
                <h4>📊 Features</h4>
                <h3 style="color: #ff7f0e;">12 Factors</h3>
                <p>Deaths, injuries, regional, temporal features</p>
            </div>
            """, unsafe_allow_html=True)

        # Model details
        st.subheader("🔧 Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Hyperparameters (Optimized):**
            - n_estimators: 500
            - learning_rate: 0.05
            - max_depth: 6
            - subsample: 0.8
            - colsample_bytree: 0.8
            - reg_alpha: 0.1
            - reg_lambda: 1
            """)

        with col2:
            st.markdown("""
            **Key Features:**
            - Year
            - Total Deaths
            - Number Injured
            - Number Homeless
            - Severity Index
            - Region Encoded
            - Disaster Type Encoded
            - Start Month
            - Regional Risk Multiplier
            - High Risk Month
            - Annual Disaster Count
            - Season Encoded
            """)

        # Model insights
        st.subheader("📈 Model Insights")
        st.markdown("""
        - **Target Variable**: Total number of people affected by natural disasters
        - **Training Data**: African natural disaster records (1993-2023)
        - **Geographic Coverage**: 5 African regions (Eastern, Western, Middle, Northern, Southern)
        - **Disaster Types**: 10+ types including floods, droughts, epidemics, storms, earthquakes
        - **Feature Engineering**: Advanced features including regional risk factors and seasonal patterns
        - **Validation**: Cross-validated performance with train/test split
        - **Log Transformation**: Applied to handle skewed target distribution
        """)

    def render_recommendations_section(self):
        """Render recommendations based on last prediction"""
        if 'last_prediction' not in st.session_state or 'last_inputs' not in st.session_state:
            st.info("Make a prediction to see personalized recommendations")
            return

        st.header("📋 Recommendations & Preparedness")

        result = st.session_state.last_prediction
        inputs = st.session_state.last_inputs
        region = inputs['region']

        # Regional recommendations
        regional_context = result.get('regional_context', {})
        recommendations = regional_context.get('recommendations', [])

        if recommendations:
            st.subheader(f"🌍 {region} Specific Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

        # Risk-based recommendations
        risk_level = result['risk_level']
        st.subheader(f"⚠️ {risk_level} Risk Preparedness Actions")

        if risk_level == 'High':
            actions = [
                "Activate emergency response protocols immediately",
                "Deploy rapid assessment teams to affected areas",
                "Establish emergency shelters and evacuation centers",
                "Coordinate with international humanitarian organizations",
                "Implement media communication strategy",
                "Mobilize emergency medical and rescue resources"
            ]
        elif risk_level == 'Medium':
            actions = [
                "Enhanced monitoring of disaster conditions",
                "Pre-position emergency supplies and equipment",
                "Brief emergency response teams",
                "Issue public awareness advisories",
                "Coordinate with regional disaster management agencies",
                "Prepare evacuation routes and shelter locations"
            ]
        else:
            actions = [
                "Maintain standard disaster preparedness levels",
                "Continue routine monitoring activities",
                "Review and update emergency response plans",
                "Conduct community preparedness exercises",
                "Maintain emergency supply inventories",
                "Monitor weather and hazard conditions"
            ]

        for i, action in enumerate(actions, 1):
            st.markdown(f"**{i}.** {action}")

    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()

        # Check model status
        if st.session_state.get('model_status') == 'training':
            st.info("🤖 Model is currently training. This page will refresh automatically when complete.")
            st.stop()

        # Sidebar inputs
        inputs = self.render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Prediction", 
            "📊 Analytics", 
            "🤖 Model Info", 
            "📋 Recommendations"
        ])

        with tab1:
            self.render_prediction_section(inputs)

        with tab2:
            self.render_analytics_dashboard()

        with tab3:
            self.render_model_performance()

        with tab4:
            self.render_recommendations_section()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            🌍 African Natural Disaster Impact Predictor | Built with XGBoost & Streamlit<br>
            📧 For support and feedback, contact your development team
        </div>
        """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    app = DisasterPredictionApp()
    app.run()
