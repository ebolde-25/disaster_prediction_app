# 🌍 African Natural Disaster Impact Predictor

A comprehensive machine learning application for predicting the number of people affected by natural disasters across African regions using advanced XGBoost algorithms.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red.svg)
![XGBoost](https://img.shields.io/badge/xgboost-2.0.3-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This application leverages historical disaster data from 1993-2023 to predict disaster impacts across five African regions:
- **Eastern Africa** (Kenya, Tanzania, Ethiopia, Uganda, etc.)
- **Western Africa** (Nigeria, Ghana, Senegal, Mali, etc.)
- **Middle Africa** (Cameroon, Chad, Central African Republic, etc.)
- **Northern Africa** (Egypt, Algeria, Sudan, Morocco, etc.)
- **Southern Africa** (South Africa, Zimbabwe, Botswana, etc.)

### 🔬 Model Performance
- **Algorithm**: XGBoost Regressor with hyperparameter optimization
- **R² Score**: 0.59 (explains 59% of variance)
- **Features**: 12 predictive factors including deaths, injuries, regional risk multipliers
- **Training Data**: 30+ years of African disaster records

## 🚀 Features

### 🎯 Disaster Impact Prediction
- Real-time prediction interface
- Support for 10+ disaster types (floods, droughts, epidemics, storms, etc.)
- Risk level assessment (Low/Medium/High)
- Confidence intervals for predictions

### 📊 Analytics Dashboard
- Regional disaster impact analysis
- Temporal trends and patterns
- Disaster type distribution
- Monthly risk analysis

### 🤖 Model Information
- Hyperparameter details
- Feature importance analysis
- Performance metrics
- Cross-validation results

### 📋 Preparedness Recommendations
- Region-specific action plans
- Risk-based preparedness strategies
- Seasonal considerations
- Emergency response guidelines

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Step 1: Clone/Download the Application
```bash
# If using Git
git clone <repository-url>
cd disaster_prediction_app

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n disaster_pred python=3.9
conda activate disaster_pred

# Using venv
python -m venv disaster_pred
source disaster_pred/bin/activate  # On Windows: disaster_pred\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
disaster_prediction_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── 
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration settings
├── 
├── data/
│   ├── cleanD_natural_disaster.csv # Raw disaster data
│   └── processed/
│       └── processed_disaster_data.csv
├── 
├── models/
│   ├── __init__.py
│   ├── model_trainer.py           # XGBoost model training
│   ├── predictor.py               # Real-time prediction engine
│   ├── disaster_prediction_model.joblib  # Trained model (auto-generated)
│   └── label_encoders.joblib      # Feature encoders (auto-generated)
├── 
├── utils/
│   ├── __init__.py
│   └── data_preprocessor.py       # Data preprocessing utilities
└── 
└── static/                        # Static assets (if any)
```

## 🎮 Usage Guide

### Making Predictions

1. **Select Parameters** in the sidebar:
   - Choose African region
   - Select disaster type
   - Set year and month
   - Input expected casualties (deaths, injuries, homeless)

2. **Generate Prediction**:
   - Click "🎯 Predict Disaster Impact"
   - View prediction results with confidence level
   - Assess risk level and recommendations

3. **Analyze Results**:
   - Review regional context
   - Check seasonal risk factors
   - Read preparedness recommendations

### Understanding Risk Levels

- **🟢 Low Risk** (<1,000 affected): Standard preparedness
- **🟡 Medium Risk** (1,000-10,000 affected): Enhanced monitoring
- **🔴 High Risk** (>10,000 affected): Emergency preparedness

### Analytics Dashboard

- **Regional Analysis**: Compare disaster impacts across regions
- **Temporal Trends**: View disaster patterns over time
- **Disaster Types**: Analyze frequency and impact by disaster type
- **Monthly Patterns**: Identify high-risk seasons

## 🔧 Model Details

### Hyperparameters (Optimized)
```python
{
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42
}
```

### Key Features
1. **Year** - Temporal trend factor
2. **Total Deaths** - Direct casualty impact
3. **Number Injured** - Injury-based impact
4. **Number Homeless** - Displacement impact
5. **Severity Index** - Composite severity measure
6. **Region Encoded** - Geographic risk factor
7. **Disaster Type Encoded** - Disaster-specific patterns
8. **Start Month** - Seasonal risk factor
9. **Regional Risk Multiplier** - Historical regional risk
10. **High Risk Month** - Regional seasonal patterns
11. **Annual Disaster Count** - Frequency factor
12. **Season Encoded** - Seasonal classification

### Data Processing Pipeline
1. **Data Cleaning**: Handle missing values, outliers
2. **Feature Engineering**: Create derived features
3. **Encoding**: Transform categorical variables
4. **Scaling**: Normalize feature distributions
5. **Validation**: Train/test split with cross-validation

## 📈 Performance Metrics

- **R² Score**: 0.59 (Good predictive capability)
- **Cross-Validation**: 5-fold validation
- **Target Transformation**: Log transformation for skewed data
- **Feature Importance**: XGBoost built-in importance scoring

## 🌍 Regional Profiles

### Eastern Africa
- **Common Disasters**: Drought, Flood, Epidemic
- **High-Risk Months**: Mar-May, Oct-Dec
- **Risk Multiplier**: 1.2
- **Average Impact**: 45,000 people

### Western Africa
- **Common Disasters**: Flood, Epidemic, Drought
- **High-Risk Months**: Jun-Sep
- **Risk Multiplier**: 1.0
- **Average Impact**: 35,000 people

### Middle Africa
- **Common Disasters**: Flood, Epidemic, Wildfire
- **High-Risk Months**: May-Sep
- **Risk Multiplier**: 0.8
- **Average Impact**: 25,000 people

### Northern Africa
- **Common Disasters**: Flood, Earthquake, Extreme Temperature
- **High-Risk Months**: Jan-Mar, Oct-Dec
- **Risk Multiplier**: 0.7
- **Average Impact**: 20,000 people

### Southern Africa
- **Common Disasters**: Drought, Flood, Storm
- **High-Risk Months**: Jan-Mar, Nov-Dec
- **Risk Multiplier**: 0.9
- **Average Impact**: 30,000 people

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the correct directory
cd disaster_prediction_app
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**2. Model Training Issues**
- Ensure the CSV data file is in the `data/` folder
- Check that you have sufficient RAM (4GB+ recommended)
- Verify Python version compatibility (3.8+)

**3. Streamlit Issues**
```bash
# Clear Streamlit cache
streamlit cache clear
# Restart the application
streamlit run app.py --server.address 0.0.0.0
```

**4. Memory Issues**
- Close other applications
- Use a machine with more RAM
- Consider reducing the dataset size for testing

### Getting Help

1. Check the console output for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure the data file is properly formatted
4. Contact the development team for support

## 📊 Data Sources

- **Primary Dataset**: EM-DAT International Disaster Database
- **Coverage**: African natural disasters (1993-2023)
- **Records**: 1000+ disaster events
- **Variables**: 40+ features including casualties, economic impact, geographic data

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time data integration
- [ ] Mobile-responsive design
- [ ] Multi-language support
- [ ] API endpoints for external integration
- [ ] Advanced visualization options
- [ ] Historical comparison tools
- [ ] Export functionality (PDF reports)
- [ ] Email alert system
- [ ] Batch prediction capabilities

### Model Improvements
- [ ] Ensemble methods (Random Forest + XGBoost)
- [ ] Deep learning integration
- [ ] Time series forecasting
- [ ] Satellite data integration
- [ ] Weather pattern analysis
- [ ] Economic impact prediction

## 🤝 Contributing

We welcome contributions to improve the disaster prediction system:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team & Acknowledgments

### Development Team
- **Data Scientists**: XGBoost model development and optimization
- **Software Engineers**: Streamlit application and infrastructure
- **Domain Experts**: Disaster management and regional expertise

### Acknowledgments
- EM-DAT Database for comprehensive disaster data
- XGBoost developers for the excellent ML framework
- Streamlit team for the amazing web app framework
- African disaster management organizations for regional insights

## 📞 Contact & Support

For questions, suggestions, or support:
- **Email**: [your-email@domain.com]
- **GitHub Issues**: [Repository Issues Page]
- **Documentation**: This README and in-app help

---

## 🎯 Quick Start Commands

```bash
# Setup (one-time)
pip install -r requirements.txt

# Run the application
streamlit run app.py

# Access the app
# Open browser to: http://localhost:8501
```

---

**Built with ❤️ for African disaster preparedness and response**

*Last updated: January 2024*
