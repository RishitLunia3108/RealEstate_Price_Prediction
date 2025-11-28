# 🏠 Real Estate Price Prediction - Streamlit App

An intelligent web application for predicting real estate prices in Ahmedabad using advanced machine learning models.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Models-green?style=for-the-badge)

## 🚀 Quick Start

### Method 1: Using the Launcher (Recommended)
```bash
python launch_app.py
```

### Method 2: Direct Streamlit Command
```bash
streamlit run app.py
```

### Method 3: Install and Run
```bash
pip install -r requirements_streamlit.txt
streamlit run app.py
```

## 📋 Prerequisites

- Python 3.7 or higher
- Internet connection (for initial package downloads)

## 🛠️ Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd modular_price_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Launch the application**
   ```bash
   python launch_app.py
   ```

## 🎯 Features

### 🔮 Price Prediction
- Interactive form for property details input
- Real-time price prediction using ML models
- Input validation and error handling
- Price range estimation with confidence intervals
- Property analysis and market positioning

### 📊 Data Dashboard
- Interactive data exploration
- Market overview with key metrics
- Price distribution analysis
- BHK configuration insights
- Locality comparison tools
- Filterable property listings

### 🏆 Model Performance
- Model accuracy metrics and comparisons
- R² scores, RMSE, and MAE values
- Feature importance analysis
- Cross-validation results
- Performance visualizations

### 📈 Market Insights
- Market segmentation analysis
- Locality-wise price trends
- Investment recommendations
- Furnishing impact on pricing
- Value for money analysis

### ℹ️ About & Documentation
- Project methodology and technical details
- Dataset statistics and model information
- Technology stack overview
- Performance benchmarks

## 📊 Screenshots

### Price Prediction Interface
![Price Prediction](images/streamlit_prediction.png)

### Data Dashboard
![Dashboard](images/streamlit_dashboard.png)

### Model Performance
![Model Performance](images/streamlit_models.png)

## 🔧 Configuration

The app uses several configuration files:

- **`streamlit_config.py`**: App configuration and settings
- **`streamlit_utils.py`**: Utility functions for data processing
- **`requirements_streamlit.txt`**: Python package dependencies

### Customization

You can customize the app by modifying `streamlit_config.py`:

```python
# App Configuration
APP_CONFIG = {
    "title": "Your Custom Title",
    "icon": "🏠",
    "layout": "wide"
}

# Add custom localities
LOCALITY_CONFIG = {
    "premium_localities": ['Your', 'Premium', 'Areas'],
    "default_locality": "Your Default Area"
}
```

## 📁 File Structure

```
modular_price_prediction/
├── app.py                          # Main Streamlit application
├── streamlit_utils.py               # Utility functions
├── streamlit_config.py              # Configuration settings
├── launch_app.py                    # App launcher script
├── requirements_streamlit.txt       # Python dependencies
├── README_STREAMLIT.md             # This file
│
├── data/                           # Data files
│   ├── cleaned_data.csv            # Processed real estate data
│   └── ahmedabad_real_estate_data.csv # Raw scraped data
│
├── models/                         # Trained ML models
│   ├── best_price_prediction_model.pkl
│   ├── feature_scaler.pkl
│   ├── furnishing_encoder.pkl
│   └── locality_encoder.pkl
│
└── images/                         # Generated visualizations
    ├── correlation_matrix.png
    ├── model_comparison.png
    └── ...
```

## 🎛️ Usage Guide

### 1. Price Prediction

1. Navigate to "🔮 Price Prediction"
2. Fill in property details:
   - **Area**: Enter property area in square feet
   - **BHK**: Select bedroom configuration
   - **Furnishing**: Choose furnishing status
   - **Locality**: Select or enter locality name
3. Click "🔍 Predict Price"
4. View prediction with confidence range and insights

### 2. Data Exploration

1. Go to "📊 Data Dashboard"
2. View market overview metrics
3. Explore interactive charts
4. Use filters to analyze specific segments
5. Compare different localities

### 3. Model Analysis

1. Visit "🏆 Model Performance"
2. Review model accuracy metrics
3. Examine feature importance
4. Compare different algorithms

### 4. Market Insights

1. Access "📈 Market Insights"
2. Review key market trends
3. Explore investment recommendations
4. Analyze locality performance

## 🚨 Troubleshooting

### Common Issues

1. **"Models not loaded" error**
   ```bash
   # Train models first
   python main.py
   ```

2. **Missing packages**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Port already in use**
   ```bash
   streamlit run app.py --server.port=8502
   ```

4. **Data not loading**
   - App will automatically use sample data if real data is unavailable
   - Run the main pipeline first: `python main.py`

### System Requirements Check

```bash
python launch_app.py --check
```

### Install Dependencies

```bash
python launch_app.py --install
```

## 🌐 Deployment

### Local Deployment
The app runs locally on `http://localhost:8501` by default.

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click

#### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI or GitHub integration

#### Docker
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements_streamlit.txt .
   RUN pip install -r requirements_streamlit.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

## 📊 Performance Metrics

- **Model Accuracy**: 89.1% R² Score
- **Average Prediction Error**: ±₹0.61 Cr MAE
- **App Loading Time**: < 3 seconds
- **Prediction Time**: < 1 second
- **Supported Properties**: 1000+ in dataset

## 🔐 Data Privacy

- No personal data is stored or transmitted
- All processing happens locally or on your deployment
- Property details are only used for prediction
- No data is shared with third parties

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational and demonstration purposes.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the configuration files
3. Ensure all dependencies are installed
4. Verify data and model files exist

## 🎉 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)
- Data processing with [pandas](https://pandas.pydata.org/)

---

**Happy Predicting! 🏠💰**