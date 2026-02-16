# 🌾 ETo Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A web-based application for predicting **Reference Evapotranspiration (ETo)** using a deep Artificial Neural Network with advanced missing value imputation.

## ✨ Features

- 🧠 **Deep Learning Model**: 5-layer ANN (256→128→64→32 neurons)
- 🔮 **Smart Imputation**: RandomForest-based missing value estimation
- 📊 **High Accuracy**: R² = 0.9863, RMSE = 0.230 mm/day
- 🎯 **Flexible Input**: Works with 2-6 parameters
- 🌐 **Web Interface**: Beautiful Streamlit UI
- 📱 **Responsive Design**: Works on desktop and mobile
- 💡 **Quick Examples**: Pre-filled test scenarios

## 🎯 Input Parameters

The model accepts the following parameters (minimum 2 required):

| Parameter | Description | Range | Unit |
|-----------|-------------|-------|------|
| **n** | Sunshine hours | 0-15 | hours |
| **Tmax** | Maximum temperature | -10 to 50 | °C |
| **Tmin** | Minimum temperature | -20 to 40 | °C |
| **RHmax** | Maximum relative humidity | 0-100 | % |
| **RHmin** | Minimum relative humidity | 0-100 | % |
| **u** | Wind speed at 2m height | 0-10 | m/s |

**Output**: Reference Evapotranspiration (ETo) in mm/day

## 📊 Model Performance

### Training Details
- **Dataset**: 7,665 samples from Ludhiana, Punjab (2000-2020)
- **Split**: 70% Train / 15% Validation / 15% Test
- **Architecture**: Deep Neural Network with 5 hidden layers
- **Imputation**: IterativeImputer with RandomForest estimator

### Metrics

| Set | RMSE | MAE | R² Score |
|-----|------|-----|----------|
| Training | 0.224 | 0.161 | 0.9873 |
| Validation | 0.230 | 0.166 | 0.9862 |
| Test | 0.230 | 0.165 | 0.9863 |

### Accuracy by Parameters Provided

| Parameters | Accuracy Level | Typical Error |
|-----------|----------------|---------------|
| 2 params | Fair | ±0.3 mm/day |
| 3 params | Good | ±0.25 mm/day |
| 4 params | Very Good | ±0.20 mm/day |
| 5-6 params | Excellent | ±0.15 mm/day |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/eto-predictor.git
   cd eto-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## 📦 Project Structure

```
eto-predictor/
├── app.py                      # Streamlit web application
├── eto_ann_model_training.py   # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── Model Files (Required for app)
│   ├── eto_ann_model.pkl      # Trained ANN model (1.1 MB)
│   ├── smart_imputer.pkl      # RandomForest imputer (40 MB)
│   ├── scaler_X.pkl           # Input feature scaler
│   ├── scaler_y.pkl           # Output scaler
│   └── feature_names.pkl      # Feature names list
│
└── Data (Optional - for retraining)
    └── et-for-ludhiana-1.xlsx # Original training data
```

## 🌐 Live Demo

**Deployed on Streamlit Cloud**: [https://eto-predictor.streamlit.app/]

## 💻 Usage

### Web Interface

1. Enter at least 2 weather parameters
2. Click "Predict ETo"
3. View the predicted evapotranspiration value
4. Check estimated values for missing parameters

### Python API

```python
from eto_ann_model_training import predict_eto

# Example 1: All parameters
result = predict_eto({
    'n': 8.0,
    'Tmax (°C)': 32.0,
    'Tmin (°C)': 18.0,
    'RHmax': 85,
    'RHmin': 45,
    'u ': 1.5
})
print(f"ETo: {result:.2f} mm/day")

# Example 2: Only temperature
result = predict_eto({
    'Tmax (°C)': 30.0,
    'Tmin (°C)': 20.0
})
print(f"ETo: {result:.2f} mm/day")
```

## 🔄 Retraining the Model

To retrain the model with your own data:

1. **Prepare your data** in the same format as `et-for-ludhiana-1.xlsx`
2. **Run the training script**:
   ```bash
   python eto_ann_model_training.py
   ```
3. **New model files** will be generated automatically

## 📈 ETo Interpretation Guide

| ETo Range (mm/day) | Condition | Irrigation Need |
|-------------------|-----------|-----------------|
| 0 - 2 | Low | Minimal |
| 2 - 4 | Moderate | Normal |
| 4 - 6 | High | Increased |
| 6+ | Very High | Heavy |

### Factors Affecting ETo

- **Temperature** (Most important) ⭐⭐⭐⭐⭐
- **Solar Radiation** (Sunshine hours) ⭐⭐⭐⭐
- **Humidity** ⭐⭐⭐
- **Wind Speed** ⭐⭐

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn
- **Deep Learning**: MLPRegressor (Neural Network)
- **Imputation**: IterativeImputer with RandomForest
- **Data Processing**: pandas, numpy
- **Model Persistence**: pickle

## 📚 Scientific Background

Reference Evapotranspiration (ETo) represents the evapotranspiration rate from a reference surface (well-watered grass). It's a key parameter in:

- 🌾 Irrigation scheduling
- 💧 Water resource management
- 🌱 Crop water requirement estimation
- 🌤️ Hydrological modeling

### Standard Method
The **FAO Penman-Monteith equation** is the standard method for computing ETo, requiring:
- Solar radiation
- Air temperature
- Humidity
- Wind speed

### Our Approach
This project uses **Machine Learning** to predict ETo from available weather parameters, with the ability to work with incomplete data through intelligent imputation.

## 🎯 Use Cases

1. **Agriculture**: Irrigation planning and scheduling
2. **Water Management**: Estimating water demand
3. **Research**: Climate impact studies
4. **Smart Farming**: Integration with IoT weather stations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<div align="center">
Made with ❤️ for Agriculture and Water Management

**[🌐 Live Demo](https://eto-predictor.streamlit.app/)**
</div>
