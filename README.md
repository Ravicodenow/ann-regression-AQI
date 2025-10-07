# Air Quality Index Prediction System

A comprehensive single-file Streamlit application for predicting Air Quality Index (AQI) using Artificial Neural Networks (ANN) with integrated model training and prediction capabilities.

## 🌟 Features

- **🔮 Single Prediction**: Interactive form for individual AQI predictions
- **📊 Batch Prediction**: Upload CSV files for multiple predictions
- **🤖 Model Training**: Built-in ANN model training with real-time progress
- **📈 Performance Analysis**: Model metrics and architecture visualization
- **💡 Real-time Results**: Instant predictions with health impact assessment
- **📋 Interactive Visualizations**: Gauge charts, histograms, and time series plots
- **🛡️ Fallback System**: Rule-based prediction when TensorFlow is unavailable
- **📥 Export Results**: Download prediction results as CSV

## 📊 Technology Stack

- **Application**: Streamlit (Single File Architecture)
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Model Storage**: Joblib

## 🗂️ Project Structure

```
AirPredictionApp/
├── streamlit_app.py         # Complete Streamlit application (MAIN FILE)
├── streamlit_requirements.txt # Streamlit app dependencies
├── run_streamlit.sh         # Quick start script
├── AirQualityUCI.csv       # Dataset (required for training)
├── models/                 # Auto-created for trained models
│   ├── aqi_model.h5        # Trained TensorFlow model
│   ├── scaler.pkl          # Feature scaler
│   └── feature_columns.pkl # Feature column names
├── backend/                # Original backend (legacy)
├── frontend/               # Original frontend (legacy)
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- macOS/Linux (script written for Unix-like systems)

### ⚡ Super Easy Startup (Recommended)

```bash
# Clone or navigate to the project directory
cd AirPredictionApp

# Run the single command to start everything
./run_streamlit.sh
```

The script will automatically:
1. Create a virtual environment (if needed)
2. Install all dependencies
3. Start the Streamlit application
4. Open your browser to http://localhost:8501

### 🔧 Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r streamlit_requirements.txt

# Start the application
streamlit run streamlit_app.py
```

### 📱 Using the Application

1. **Single Prediction**: 
   - Navigate to "🔮 Single Prediction"
   - Fill in the air quality measurements
   - Click "🔮 Predict AQI" to get instant results

2. **Batch Prediction**:
   - Go to "📊 Batch Prediction"
   - Upload a CSV file with air quality data
   - View results and download predictions

3. **Train Your Own Model**:
   - Visit "🤖 Train Model"
   - Ensure `AirQualityUCI.csv` is in the directory
   - Adjust training parameters
   - Start training and monitor progress

4. **View Performance**:
   - Check "📈 Model Performance" for model details

## 🎯 Key Features

### 🔮 Single Prediction Page
- Interactive form with all air quality parameters
- Real-time AQI prediction with health impact
- Beautiful gauge chart visualization
- Categorical health recommendations

### 📊 Batch Prediction Page
- Upload CSV files (original or processed format)
- Automatic format detection and preprocessing
- Batch processing with progress tracking
- Statistical summaries and visualizations
- Export results as CSV

### 🤖 Model Training Page
- Train custom ANN models
- Adjustable hyperparameters
- Real-time training progress
- Performance metrics and plots
- Automatic model saving

### 📈 Performance Analysis
- Model architecture details
- Input feature descriptions
- Performance metrics display

## 📋 Input Data Format

The application accepts multiple CSV formats:

### Original AirQualityUCI.csv Format
```csv
Date,Time,CO(GT),PT08.S1(CO),NMHC(GT),C6H6(GT),PT08.S2(NMHC),NOx(GT),PT08.S3(NOx),NO2(GT),PT08.S4(NO2),PT08.S5(O3),T,RH,AH
10/03/2004,18.00.00,2.6,1360,150,11.9,1046,166,1056,113,1692,1268,13.6,48.9,0.7578
```

### Processed Format
```csv
CO_GT,PT08_S1_CO,NMHC_GT,C6H6_GT,PT08_S2_NMHC,NOx_GT,PT08_S3_NOx,NO2_GT,PT08_S4_NO2,PT08_S5_O3,T,RH,AH,Hour,Day,Month
2.6,1360,150,11.9,1046,166,1056,113,1692,1268,13.6,48.9,0.7578,18,10,3
```

## 🧠 Model Architecture

### Deep Neural Network (ANN)
```
Input Layer (17 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (30%)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Dense Layer (16 neurons, ReLU)
    ↓
Output Layer (1 neuron, Linear)
```

### Input Features
- **Pollutant Concentrations**: CO(GT), NMHC(GT), C6H6(GT), NOx(GT), NO2(GT)
- **Sensor Responses**: PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)
- **Environmental**: Temperature (T), Relative Humidity (RH), Absolute Humidity (AH)
- **Temporal**: Hour, Day, Month, DayOfWeek

### Fallback System
If TensorFlow is not available, the application automatically uses a rule-based prediction system ensuring continued functionality.

## 🎨 AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0-50 | Good | Air quality is satisfactory |
| 51-100 | Moderate | Acceptable for most people |
| 101-150 | Unhealthy for Sensitive Groups | Sensitive individuals may be affected |
| 151-200 | Unhealthy | Everyone may experience health effects |
| 201-300 | Very Unhealthy | Health warnings for everyone |
| 301-500 | Hazardous | Emergency conditions |

## 🔧 Advanced Configuration

### Custom Model Training
- Adjust epochs (50-200)
- Configure batch size (16, 32, 64, 128)
- Set test/train split ratio
- Monitor training progress in real-time

### Data Preprocessing
- Automatic handling of missing values (-200 indicators)
- Feature scaling with StandardScaler
- Time-based feature extraction
- Multi-format CSV support

## 📊 Visualizations

### Single Prediction
- **Gauge Chart**: Interactive AQI meter with color-coded zones
- **Category Display**: Health impact with color-coded recommendations

### Batch Prediction
- **Histogram**: Distribution of predicted AQI values
- **Pie Chart**: Category distribution breakdown
- **Time Series**: AQI trends over sequence/time
- **Summary Statistics**: Min, max, average AQI values

### Model Training
- **Loss Curves**: Training and validation loss over epochs
- **Scatter Plot**: Predicted vs actual values
- **Metrics Dashboard**: R², MAE, RMSE in real-time

## 🚨 Troubleshooting

### Common Issues

1. **TensorFlow Not Available**
   - The app will automatically use rule-based prediction
   - Install TensorFlow: `pip install tensorflow`

2. **Dataset Not Found**
   - Ensure `AirQualityUCI.csv` is in the project directory
   - Download from UCI ML Repository if missing

3. **CSV Upload Issues**
   - Check file format (semicolon or comma separated)
   - Ensure required columns are present
   - Verify no special characters in data

4. **Model Training Fails**
   - Check available memory
   - Reduce batch size or epochs
   - Ensure dataset is properly formatted

### Performance Tips
- For large datasets, use smaller batch sizes
- Monitor memory usage during training
- Use the batch prediction for multiple samples
- Export results for further analysis

## 🔄 Migration from Original Structure

This single-file app replaces the original FastAPI backend + Streamlit frontend architecture:

### What's Included
- ✅ All prediction functionality
- ✅ Model training capabilities  
- ✅ Batch processing
- ✅ Interactive visualizations
- ✅ Model persistence
- ✅ Health impact assessment

### What's Changed
- 🔄 Single file instead of client-server architecture
- 🔄 Integrated model training (no separate script needed)
- 🔄 Direct data processing (no API calls)
- 🔄 Simplified deployment and setup

### Legacy Files
The original `backend/` and `frontend/` directories are preserved for reference but are no longer needed for the single-file app.

## 📈 Performance Metrics

Expected model performance:
- **R² Score**: ~0.85
- **Mean Absolute Error**: ~15-20 AQI units
- **Root Mean Square Error**: ~20-25 AQI units

*Actual metrics will be displayed after model training*

---

## 🌟 Getting Started Summary

1. **Quick Start**: Run `./run_streamlit.sh`
2. **Access App**: Open http://localhost:8501
3. **Make Predictions**: Use the single prediction form
4. **Batch Process**: Upload CSV files for bulk predictions
5. **Train Model**: Use your data to create custom models
6. **Analyze**: View model performance and metrics

The application is designed to be user-friendly while providing powerful machine learning capabilities for air quality analysis.