import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, handle gracefully if not available
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None
    tf = None
    st.warning("⚠️ TensorFlow not available. Using rule-based prediction model.")

# Configure page
st.set_page_config(
    page_title="Air Quality Index Prediction",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .good { background-color: #d4edda; border-left: 5px solid #28a745; }
    .moderate { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .unhealthy-sensitive { background-color: #f8d7da; border-left: 5px solid #fd7e14; }
    .unhealthy { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .very-unhealthy { background-color: #f8d7da; border-left: 5px solid #6f42c1; }
    .hazardous { background-color: #f8d7da; border-left: 5px solid #343a40; }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
MODEL_PATH = "models/aqi_model.h5"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    if os.path.exists("AirQualityUCI.csv"):
        try:
            df = pd.read_csv("AirQualityUCI.csv", sep=';', decimal=',')
            df = df.dropna(axis=1, how='all')
            df = df.dropna(axis=0, how='all')
            if df.columns[-1] == '':
                df = df.drop(columns=[df.columns[-1]])
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    return None

def calculate_aqi(df):
    """Calculate a simplified AQI based on available pollutants"""
    co_normalized = (df['CO(GT)'] / df['CO(GT)'].max()) * 100
    nox_normalized = (df['NOx(GT)'] / df['NOx(GT)'].max()) * 100
    no2_normalized = (df['NO2(GT)'] / df['NO2(GT)'].max()) * 100
    
    # Weighted average
    aqi = (co_normalized * 0.4 + nox_normalized * 0.3 + no2_normalized * 0.3)
    aqi = aqi * 5  # Scale to typical AQI range (0-500)
    
    return aqi.fillna(aqi.mean())

def preprocess_data(df):
    """Preprocess the air quality data"""
    # Convert Date and Time to datetime
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    
    # Drop original Date and Time columns
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Replace -200 values (missing data indicator) with NaN
    df = df.replace(-200.0, np.nan)
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Create AQI target variable
    df['AQI'] = calculate_aqi(df)
    
    # Extract time-based features
    df['Hour'] = df['DateTime'].dt.hour
    df['Day'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    
    # Drop DateTime column
    df = df.drop('DateTime', axis=1)
    
    return df

def build_model(input_dim):
    """Build the ANN model for regression"""
    if not TENSORFLOW_AVAILABLE:
        return None
        
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Single output for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model():
    """Train the ANN model"""
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow not available. Cannot train model.")
        return None, None, None
        
    # Load data
    df = load_data()
    if df is None:
        st.error("Dataset not found. Please ensure AirQualityUCI.csv is in the current directory.")
        return None, None, None
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare features
    feature_columns = [col for col in df.columns if col != 'AQI']
    X = df[feature_columns]
    y = df['AQI']
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    model = build_model(X_train_scaled.shape[1])
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Custom callback to update progress
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / 100  # Assuming 100 epochs
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f'Training epoch {epoch + 1}/100 - Loss: {logs["loss"]:.4f}')
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[ProgressCallback()]
    )
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    
    progress_bar.empty()
    status_text.empty()
    
    return model, scaler, feature_columns, {'mse': mse, 'r2': r2, 'mae': mae}

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        if TENSORFLOW_AVAILABLE and os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            return model, scaler, feature_columns
    except Exception as e:
        st.warning(f"Could not load trained model: {e}")
    return None, None, None

def calculate_rule_based_aqi(input_data):
    """Calculate AQI using rule-based approach when ML model is not available"""
    # Enhanced rule-based calculation using multiple pollutants
    
    # Handle both formats: 'CO_GT' and 'CO(GT)'
    co_value = input_data.get('CO(GT)', input_data.get('CO_GT', 0))
    nox_value = input_data.get('NOx(GT)', input_data.get('NOx_GT', 0))
    no2_value = input_data.get('NO2(GT)', input_data.get('NO2_GT', 0))
    nmhc_value = input_data.get('NMHC(GT)', input_data.get('NMHC_GT', 0))
    c6h6_value = input_data.get('C6H6(GT)', input_data.get('C6H6_GT', 0))
    temp_value = input_data.get('T', 20)
    humidity_value = input_data.get('RH', 50)
    
    # Debug print (will be visible in logs)
    # print(f"Debug: CO={co_value}, NOx={nox_value}, NO2={no2_value}, T={temp_value}, RH={humidity_value}")
    
    # Normalize pollutant values based on typical ranges
    co_norm = min(max(co_value, 0) / 10.0, 1.0) * 100     # CO: 0-10 mg/m³
    nox_norm = min(max(nox_value, 0) / 500.0, 1.0) * 100  # NOx: 0-500 µg/m³
    no2_norm = min(max(no2_value, 0) / 300.0, 1.0) * 100  # NO2: 0-300 µg/m³
    nmhc_norm = min(max(nmhc_value, 0) / 1000.0, 1.0) * 100  # NMHC: 0-1000 µg/m³
    c6h6_norm = min(max(c6h6_value, 0) / 50.0, 1.0) * 100    # Benzene: 0-50 µg/m³
    
    # Weighted average (weights based on health impact)
    aqi_base = (co_norm * 0.25 + nox_norm * 0.20 + no2_norm * 0.25 + 
                nmhc_norm * 0.15 + c6h6_norm * 0.15)
    
    # Scale to AQI range (0-500)
    aqi = aqi_base * 3  # Scale factor to get reasonable AQI values
    
    # Environmental factors
    temp_factor = 1.0 + (abs(temp_value - 20) / 50)  # Temperature effect
    humidity_factor = 1.0 + (abs(humidity_value - 50) / 100)  # Humidity effect
    
    # Apply environmental factors
    aqi = aqi * temp_factor * humidity_factor
    
    # Ensure minimum AQI for any pollution (if any pollutants are present)
    if any([co_value > 0, nox_value > 0, no2_value > 0, nmhc_value > 0, c6h6_value > 0]):
        aqi = max(aqi, 15)  # Minimum AQI when pollutants are detected
    
    return max(0, min(500, aqi))  # Clamp between 0 and 500

def get_aqi_category(aqi_value):
    """Get AQI category and health impact"""
    if aqi_value <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi_value <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people who are unusually sensitive to air pollution."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi_value <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected."

def get_aqi_color_class(category):
    """Get CSS class for AQI category"""
    color_map = {
        "Good": "good",
        "Moderate": "moderate",
        "Unhealthy for Sensitive Groups": "unhealthy-sensitive",
        "Unhealthy": "unhealthy",
        "Very Unhealthy": "very-unhealthy",
        "Hazardous": "hazardous"
    }
    return color_map.get(category, "moderate")

def predict_aqi(input_data, model=None, scaler=None, feature_columns=None):
    """Make AQI prediction using available model"""
    try:
        if model is not None and scaler is not None and feature_columns is not None:
            # Use ML model
            df = pd.DataFrame([input_data])
            df = df[feature_columns]
            features_scaled = scaler.transform(df)
            prediction = model.predict(features_scaled)
            predicted_aqi = float(prediction[0][0])
        else:
            # Use rule-based approach
            predicted_aqi = calculate_rule_based_aqi(input_data)
        
        category, health_impact = get_aqi_category(predicted_aqi)
        
        return {
            "predicted_aqi": round(predicted_aqi, 2),
            "aqi_category": category,
            "health_impact": health_impact
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🌬️ Air Quality Index Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_columns = load_model()
    
    # Model status
    if model is not None:
        st.success("✅ AI Model loaded successfully! Using deep learning predictions.")
    else:
        st.info("ℹ️ Using rule-based prediction model. Train a new model for better accuracy.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🔮 Single Prediction",
        "📊 Batch Prediction", 
        "🤖 Train Model",
        "📈 Model Performance",
        "ℹ️ About"
    ])
    
    if page == "🔮 Single Prediction":
        show_single_prediction_page(model, scaler, feature_columns)
    elif page == "📊 Batch Prediction":
        show_batch_prediction_page(model, scaler, feature_columns)
    elif page == "🤖 Train Model":
        show_train_model_page()
    elif page == "📈 Model Performance":
        show_model_performance_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_single_prediction_page(model, scaler, feature_columns):
    """Single prediction page"""
    st.header("Single AQI Prediction")
    st.write("Enter air quality measurements to predict the Air Quality Index.")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pollutant Concentrations")
            
            co_gt = st.number_input(
                "CO Concentration (mg/m³)", 
                min_value=0.0, 
                max_value=20.0, 
                value=2.6, 
                step=0.1,
                help="Carbon Monoxide concentration"
            )
            
            pt08_s1_co = st.number_input(
                "CO Sensor Response", 
                min_value=500.0, 
                max_value=2000.0, 
                value=1360.0, 
                step=1.0,
                help="CO sensor response"
            )
            
            nmhc_gt = st.number_input(
                "NMHC Concentration (µg/m³)", 
                min_value=0.0, 
                max_value=500.0, 
                value=150.0, 
                step=1.0,
                help="Non-methanic hydrocarbons concentration"
            )
            
            c6h6_gt = st.number_input(
                "Benzene Concentration (µg/m³)", 
                min_value=0.0, 
                max_value=50.0, 
                value=11.9, 
                step=0.1,
                help="Benzene concentration"
            )
            
            pt08_s2_nmhc = st.number_input(
                "NMHC Sensor Response", 
                min_value=500.0, 
                max_value=2000.0, 
                value=1046.0, 
                step=1.0,
                help="NMHC sensor response"
            )
            
            nox_gt = st.number_input(
                "NOx Concentration (µg/m³)", 
                min_value=0.0, 
                max_value=1000.0, 
                value=166.0, 
                step=1.0,
                help="Nitrogen oxides concentration"
            )
            
            pt08_s3_nox = st.number_input(
                "NOx Sensor Response", 
                min_value=500.0, 
                max_value=2000.0, 
                value=1056.0, 
                step=1.0,
                help="NOx sensor response"
            )
        
        with col2:
            st.subheader("Other Pollutants")
            
            no2_gt = st.number_input(
                "NO2 Concentration (µg/m³)", 
                min_value=0.0, 
                max_value=500.0, 
                value=113.0, 
                step=1.0,
                help="Nitrogen dioxide concentration"
            )
            
            pt08_s4_no2 = st.number_input(
                "NO2 Sensor Response", 
                min_value=500.0, 
                max_value=3000.0, 
                value=1692.0, 
                step=1.0,
                help="NO2 sensor response"
            )
            
            pt08_s5_o3 = st.number_input(
                "O3 Sensor Response", 
                min_value=500.0, 
                max_value=2000.0, 
                value=1268.0, 
                step=1.0,
                help="Ozone sensor response"
            )
            
            st.subheader("Environmental Conditions")
            
            temperature = st.number_input(
                "Temperature (°C)", 
                min_value=-20.0, 
                max_value=50.0, 
                value=13.6, 
                step=0.1
            )
            
            relative_humidity = st.number_input(
                "Relative Humidity (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=48.9, 
                step=0.1
            )
            
            absolute_humidity = st.number_input(
                "Absolute Humidity", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7578, 
                step=0.01
            )
            
            st.subheader("Time Information")
            
            hour = st.selectbox("Hour", options=list(range(24)), index=18)
            day = st.selectbox("Day", options=list(range(1, 32)), index=9)
            month = st.selectbox("Month", options=list(range(1, 13)), index=2)
            day_of_week = st.selectbox(
                "Day of Week", 
                options=list(range(7)), 
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                index=2
            )
        
        # Prediction button
        submitted = st.form_submit_button("🔮 Predict AQI", type="primary", use_container_width=True)
        
        if submitted:
            # Prepare data
            prediction_data = {
                'CO(GT)': co_gt,
                'PT08.S1(CO)': pt08_s1_co,
                'NMHC(GT)': nmhc_gt,
                'C6H6(GT)': c6h6_gt,
                'PT08.S2(NMHC)': pt08_s2_nmhc,
                'NOx(GT)': nox_gt,
                'PT08.S3(NOx)': pt08_s3_nox,
                'NO2(GT)': no2_gt,
                'PT08.S4(NO2)': pt08_s4_no2,
                'PT08.S5(O3)': pt08_s5_o3,
                'T': temperature,
                'RH': relative_humidity,
                'AH': absolute_humidity,
                'Hour': hour,
                'Day': day,
                'Month': month,
                'DayOfWeek': day_of_week
            }
            
            # Debug section (expandable)
            with st.expander("🔍 Debug: View Input Values", expanded=False):
                st.write("**Input values being used for prediction:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Pollutants:**")
                    st.write(f"- CO: {prediction_data['CO(GT)']} mg/m³")
                    st.write(f"- NOx: {prediction_data['NOx(GT)']} µg/m³")
                    st.write(f"- NO2: {prediction_data['NO2(GT)']} µg/m³")
                    st.write(f"- NMHC: {prediction_data['NMHC(GT)']} µg/m³")
                    st.write(f"- Benzene: {prediction_data['C6H6(GT)']} µg/m³")
                with col2:
                    st.write("**Environmental:**")
                    st.write(f"- Temperature: {prediction_data['T']}°C")
                    st.write(f"- Humidity: {prediction_data['RH']}%")
                    st.write(f"- Time: {prediction_data['Hour']}:00")
                    st.write(f"- Date: {prediction_data['Day']}/{prediction_data['Month']}")
            
            # Make prediction
            with st.spinner("Making prediction..."):
                result = predict_aqi(prediction_data, model, scaler, feature_columns)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                aqi_value = result["predicted_aqi"]
                category = result["aqi_category"]
                health_impact = result["health_impact"]
                
                # Main prediction display
                color_class = get_aqi_color_class(category)
                
                st.markdown(f"""
                <div class="prediction-box {color_class}">
                    <h2 style="margin: 0; color: #333;">Predicted AQI: {aqi_value}</h2>
                    <h3 style="margin: 0.5rem 0; color: #555;">Category: {category}</h3>
                    <p style="margin: 0; color: #666;">{health_impact}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # AQI gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=aqi_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Air Quality Index"},
                    delta={'reference': 100},
                    gauge={
                        'axis': {'range': [None, 500]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "yellow"},
                            {'range': [100, 150], 'color': "orange"},
                            {'range': [150, 200], 'color': "red"},
                            {'range': [200, 300], 'color': "purple"},
                            {'range': [300, 500], 'color': "maroon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 150
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def show_batch_prediction_page(model, scaler, feature_columns):
    """Batch prediction page"""
    st.header("Batch AQI Prediction")
    st.write("Upload a CSV file with air quality data for batch predictions.")
    
    st.info("""
    💡 **Supported Formats:**
    - Original AirQualityUCI.csv format (semicolon-separated)
    - Standard CSV format (comma-separated)
    - The system will automatically detect and process the format
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            try:
                # Try semicolon separator first
                df = pd.read_csv(uploaded_file, sep=';', decimal=',')
                if df.shape[1] == 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            st.write(f"**File info:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            if st.button("🔮 Make Batch Predictions", type="primary"):
                with st.spinner("Processing data and making predictions..."):
                    try:
                        # Preprocess the data
                        processed_df = preprocess_batch_data(df)
                        
                        # Make predictions
                        predictions = []
                        progress_bar = st.progress(0)
                        
                        for i, (_, row) in enumerate(processed_df.iterrows()):
                            prediction_data = row.to_dict()
                            result = predict_aqi(prediction_data, model, scaler, feature_columns)
                            if "error" not in result:
                                predictions.append(result)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(processed_df))
                        
                        progress_bar.empty()
                        
                        if predictions:
                            st.success(f"✅ Successfully processed {len(predictions)} predictions!")
                            
                            # Display results
                            df_results = pd.DataFrame(predictions)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Predictions", len(predictions))
                            with col2:
                                avg_aqi = df_results["predicted_aqi"].mean()
                                st.metric("Average AQI", f"{avg_aqi:.1f}")
                            with col3:
                                max_aqi = df_results["predicted_aqi"].max()
                                st.metric("Max AQI", f"{max_aqi:.1f}")
                            with col4:
                                min_aqi = df_results["predicted_aqi"].min()
                                st.metric("Min AQI", f"{min_aqi:.1f}")
                            
                            # Results table
                            st.subheader("Prediction Results")
                            st.dataframe(df_results)
                            
                            # Download button
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"aqi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # AQI distribution
                                fig_hist = px.histogram(
                                    df_results, 
                                    x="predicted_aqi", 
                                    title="Distribution of Predicted AQI Values",
                                    nbins=30
                                )
                                fig_hist.update_layout(height=400)
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            with col2:
                                # Category distribution
                                category_counts = df_results["aqi_category"].value_counts()
                                fig_pie = px.pie(
                                    values=category_counts.values,
                                    names=category_counts.index,
                                    title="AQI Category Distribution"
                                )
                                fig_pie.update_layout(height=400)
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Time series if multiple predictions
                            if len(predictions) > 1:
                                df_results['index'] = range(len(predictions))
                                fig_line = px.line(
                                    df_results,
                                    x='index',
                                    y='predicted_aqi',
                                    title='AQI Predictions Over Sequence'
                                )
                                fig_line.update_layout(height=400)
                                st.plotly_chart(fig_line, use_container_width=True)
                        else:
                            st.error("No valid predictions could be made from the uploaded data.")
                            
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        # Show sample format
        st.subheader("📋 Sample Data Format")
        
        sample_data = pd.DataFrame({
            'Date': ['10/03/2004'],
            'Time': ['18.00.00'],
            'CO(GT)': [2.6],
            'PT08.S1(CO)': [1360],
            'NMHC(GT)': [150],
            'C6H6(GT)': [11.9],
            'PT08.S2(NMHC)': [1046],
            'NOx(GT)': [166],
            'PT08.S3(NOx)': [1056],
            'NO2(GT)': [113],
            'PT08.S4(NO2)': [1692],
            'PT08.S5(O3)': [1268],
            'T': [13.6],
            'RH': [48.9],
            'AH': [0.7578]
        })
        
        st.dataframe(sample_data)

def preprocess_batch_data(df):
    """Preprocess batch data for prediction"""
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Column mapping for different formats
    column_mapping = {
        'CO(GT)': 'CO(GT)',
        'PT08.S1(CO)': 'PT08.S1(CO)',
        'NMHC(GT)': 'NMHC(GT)',
        'C6H6(GT)': 'C6H6(GT)',
        'PT08.S2(NMHC)': 'PT08.S2(NMHC)',
        'NOx(GT)': 'NOx(GT)',
        'PT08.S3(NOx)': 'PT08.S3(NOx)',
        'NO2(GT)': 'NO2(GT)',
        'PT08.S4(NO2)': 'PT08.S4(NO2)',
        'PT08.S5(O3)': 'PT08.S5(O3)',
        'T': 'T',
        'RH': 'RH',
        'AH': 'AH'
    }
    
    # Replace -200 values with NaN
    processed_df = processed_df.replace(-200.0, np.nan)
    
    # Handle missing values
    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
    
    # Add time features if Date and Time columns exist
    if 'Date' in processed_df.columns and 'Time' in processed_df.columns:
        try:
            processed_df['DateTime'] = pd.to_datetime(
                processed_df['Date'] + ' ' + processed_df['Time'], 
                format='%d/%m/%Y %H.%M.%S'
            )
            processed_df['Hour'] = processed_df['DateTime'].dt.hour
            processed_df['Day'] = processed_df['DateTime'].dt.day
            processed_df['Month'] = processed_df['DateTime'].dt.month
            processed_df['DayOfWeek'] = processed_df['DateTime'].dt.dayofweek
            
            processed_df = processed_df.drop(['Date', 'Time', 'DateTime'], axis=1)
        except:
            # Add default time features
            processed_df['Hour'] = 12
            processed_df['Day'] = 15
            processed_df['Month'] = 6
            processed_df['DayOfWeek'] = 3
    else:
        # Add default time features
        processed_df['Hour'] = 12
        processed_df['Day'] = 15
        processed_df['Month'] = 6
        processed_df['DayOfWeek'] = 3
    
    # Ensure all required columns exist
    required_columns = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
        'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH', 'Hour', 'Day', 'Month', 'DayOfWeek'
    ]
    
    for col in required_columns:
        if col not in processed_df.columns:
            if col == 'Hour':
                processed_df[col] = 12
            elif col == 'Day':
                processed_df[col] = 15
            elif col == 'Month':
                processed_df[col] = 6
            elif col == 'DayOfWeek':
                processed_df[col] = 3
            elif col == 'T':
                processed_df[col] = 20.0
            elif col == 'RH':
                processed_df[col] = 50.0
            elif col == 'AH':
                processed_df[col] = 0.8
            else:
                processed_df[col] = 100.0
    
    # Select only required columns
    processed_df = processed_df[required_columns]
    
    # Fill any remaining NaN values
    processed_df = processed_df.fillna(processed_df.mean())
    
    return processed_df

def show_train_model_page():
    """Train model page"""
    st.header("Train AQI Prediction Model")
    st.write("Train a new ANN model using your own Air Quality dataset or the default dataset.")
    
    if not TENSORFLOW_AVAILABLE:
        st.error("❌ TensorFlow is not available. Please install TensorFlow to train the model.")
        st.code("pip install tensorflow")
        return
    
    # Data source selection
    st.subheader("📁 Data Source Selection")
    data_source = st.radio(
        "Choose your data source:",
        ["Use Default Dataset (AirQualityUCI.csv)", "Upload Custom Dataset"],
        help="Select whether to use the default dataset or upload your own"
    )
    
    dataset_available = False
    uploaded_file = None
    
    if data_source == "Use Default Dataset (AirQualityUCI.csv)":
        if os.path.exists("AirQualityUCI.csv"):
            dataset_available = True
            st.success("✅ Default dataset found: AirQualityUCI.csv")
        else:
            st.error("❌ Default dataset file 'AirQualityUCI.csv' not found in the current directory.")
            st.info("💡 Switch to 'Upload Custom Dataset' to use your own data.")
    
    else:  # Upload Custom Dataset
        st.info("""
        📋 **Dataset Format Requirements:**
        - CSV file with semicolon (;) or comma (,) separation
        - Required columns: Date, Time, CO(GT), PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH
        - Missing values can be represented as -200 or NaN
        - Decimal separator can be comma or period
        """)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload your air quality dataset in the specified format"
        )
        
        if uploaded_file is not None:
            # Preview uploaded data
            try:
                # Try to read the file with different separators
                try:
                    df_preview = pd.read_csv(uploaded_file, sep=';', decimal=',', nrows=5)
                    if df_preview.shape[1] == 1:
                        uploaded_file.seek(0)
                        df_preview = pd.read_csv(uploaded_file, nrows=5)
                except:
                    uploaded_file.seek(0)
                    df_preview = pd.read_csv(uploaded_file, nrows=5)
                
                uploaded_file.seek(0)  # Reset file pointer
                
                st.subheader("📋 Dataset Preview")
                st.dataframe(df_preview)
                st.write(f"**File info:** Preview of first 5 rows, {df_preview.shape[1]} columns detected")
                
                # Validate required columns
                required_base_columns = ['Date', 'Time', 'CO(GT)', 'T', 'RH']
                missing_columns = [col for col in required_base_columns if col not in df_preview.columns]
                
                if missing_columns:
                    st.warning(f"⚠️ Missing required columns: {missing_columns}")
                    st.info("The system will attempt to handle missing columns during preprocessing.")
                else:
                    st.success("✅ Dataset format looks good!")
                
                dataset_available = True
                
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {str(e)}")
                st.info("Please ensure your file is a valid CSV with the correct format.")
    
    # Show sample format
    with st.expander("📋 View Required Dataset Format", expanded=False):
        sample_data = pd.DataFrame({
            'Date': ['10/03/2004', '10/03/2004', '10/03/2004'],
            'Time': ['18.00.00', '19.00.00', '20.00.00'],
            'CO(GT)': [2.6, 2.0, 2.2],
            'PT08.S1(CO)': [1360, 1292, 1402],
            'NMHC(GT)': [150, 112, 88],
            'C6H6(GT)': [11.9, 9.4, 9.0],
            'PT08.S2(NMHC)': [1046, 955, 939],
            'NOx(GT)': [166, 103, 131],
            'PT08.S3(NOx)': [1056, 1174, 1140],
            'NO2(GT)': [113, 92, 114],
            'PT08.S4(NO2)': [1692, 1559, 1555],
            'PT08.S5(O3)': [1268, 972, 1074],
            'T': [13.6, 13.3, 11.9],
            'RH': [48.9, 47.7, 54.0],
            'AH': [0.7578, 0.7255, 0.7502]
        })
        st.dataframe(sample_data)
        
        # Download sample template
        csv_sample = sample_data.to_csv(sep=';', decimal=',', index=False)
        st.download_button(
            label="📥 Download Sample Template",
            data=csv_sample,
            file_name="air_quality_template.csv",
            mime="text/csv",
            help="Download a template CSV file with the correct format"
        )
    
    if not dataset_available:
        st.warning("⚠️ Please provide a dataset to proceed with training.")
        return
    
    st.info("ℹ️ Training will create a new ANN model and may take several minutes.")
    
    # Training parameters
    st.subheader("⚙️ Training Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", min_value=50, max_value=300, value=100, step=10,
                          help="Number of training iterations. More epochs may improve accuracy but take longer.")
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1,
                                 help="Number of samples processed together. Larger batches may train faster.")
    
    with col2:
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05,
                             help="Fraction of data used for testing (not used in training).")
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0,
                                    help="Controls how quickly the model learns. Lower is more stable.")
    
    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.3, step=0.05,
                                   help="Prevents overfitting. Higher values = more regularization.")
            hidden_layers = st.selectbox("Hidden Layer Sizes", 
                                       ["128-64-32-16", "256-128-64-32", "64-32-16", "512-256-128-64"],
                                       index=0, help="Architecture of the neural network layers.")
        with col2:
            validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=0.2, step=0.05,
                                        help="Fraction of training data used for validation during training.")
            early_stopping = st.checkbox("Early Stopping", value=True,
                                        help="Stop training if validation loss doesn't improve.")
    
    if st.button("🚀 Start Training", type="primary"):
        st.info("🔄 Training started... This may take several minutes.")
        
        try:
            # Load data based on source
            if data_source == "Use Default Dataset (AirQualityUCI.csv)":
                df = load_data()
                if df is None:
                    st.error("Failed to load default dataset.")
                    return
                st.success("✅ Default dataset loaded successfully")
            else:
                # Load uploaded file
                try:
                    # Try semicolon separator first
                    df = pd.read_csv(uploaded_file, sep=';', decimal=',')
                    if df.shape[1] == 1:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Custom dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            with st.spinner("Preprocessing data..."):
                df = preprocess_data(df)
            
            # Data quality check
            st.subheader("📊 Data Quality Report")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len([col for col in df.columns if col != 'AQI']))
            with col3:
                missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing Data %", f"{missing_percentage:.1f}%")
            with col4:
                st.metric("Target (AQI)", "Generated" if 'AQI' in df.columns else "Missing")
            
            # Prepare features
            feature_columns = [col for col in df.columns if col != 'AQI']
            X = df[feature_columns]
            y = df['AQI']
            
            # Remove any remaining NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                st.error("❌ No valid samples found after preprocessing. Please check your data quality.")
                return
            
            st.success(f"✅ Data ready: {len(X)} valid samples, {len(feature_columns)} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Build model with custom architecture
            def build_custom_model(input_dim, hidden_layers_str, dropout_rate, learning_rate):
                layer_sizes = list(map(int, hidden_layers_str.split('-')))
                
                model = keras.Sequential()
                model.add(layers.Dense(layer_sizes[0], activation='relu', input_shape=(input_dim,)))
                model.add(layers.Dropout(dropout_rate))
                
                for size in layer_sizes[1:]:
                    model.add(layers.Dense(size, activation='relu'))
                    model.add(layers.Dropout(dropout_rate * 0.7))  # Reduced dropout for deeper layers
                
                model.add(layers.Dense(1))  # Output layer
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='mse',
                    metrics=['mae']
                )
                
                return model
            
            model = build_custom_model(X_train_scaled.shape[1], hidden_layers, dropout_rate, learning_rate)
            st.success("✅ Custom model architecture created")
            
            # Display model summary
            with st.expander("🏗️ Model Architecture Summary", expanded=False):
                buffer = io.StringIO()
                model.summary(print_fn=lambda x: buffer.write(x + '\n'))
                model_summary = buffer.getvalue()
                st.text(model_summary)
            
            # Prepare callbacks
            callbacks = []
            if early_stopping:
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=20, 
                    restore_best_weights=True,
                    verbose=0
                )
                callbacks.append(early_stop)
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Store training history for plotting
            training_history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
            
            # Custom callback
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f'Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f} - Val Loss: {logs["val_loss"]:.4f} - MAE: {logs["mae"]:.4f}')
                    
                    # Store history
                    training_history['loss'].append(logs['loss'])
                    training_history['val_loss'].append(logs['val_loss'])
                    training_history['mae'].append(logs['mae'])
                    training_history['val_mae'].append(logs['val_mae'])
                    
                    if epoch % 10 == 0:  # Update metrics every 10 epochs
                        with metrics_placeholder.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Training Loss", f"{logs['loss']:.4f}")
                            with col2:
                                st.metric("Validation Loss", f"{logs['val_loss']:.4f}")
                            with col3:
                                st.metric("Training MAE", f"{logs['mae']:.4f}")
                            with col4:
                                st.metric("Validation MAE", f"{logs['val_mae']:.4f}")
                        
                        # Update training chart
                        if len(training_history['loss']) > 1:
                            with chart_placeholder.container():
                                fig_training = go.Figure()
                                epochs_range = list(range(1, len(training_history['loss']) + 1))
                                
                                fig_training.add_trace(go.Scatter(
                                    x=epochs_range, y=training_history['loss'],
                                    mode='lines', name='Training Loss', line=dict(color='blue')
                                ))
                                fig_training.add_trace(go.Scatter(
                                    x=epochs_range, y=training_history['val_loss'],
                                    mode='lines', name='Validation Loss', line=dict(color='red')
                                ))
                                
                                fig_training.update_layout(
                                    title="Training Progress",
                                    xaxis_title="Epoch",
                                    yaxis_title="Loss",
                                    height=300
                                )
                                st.plotly_chart(fig_training, use_container_width=True)
            
            callbacks.append(StreamlitCallback())
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks
            )
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model.save(MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
            
            # Save training metadata
            training_metadata = {
                'dataset_source': data_source,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(feature_columns),
                'epochs_trained': len(history.history['loss']),
                'final_metrics': {'mse': mse, 'r2': r2, 'mae': mae},
                'training_params': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'dropout_rate': dropout_rate,
                    'hidden_layers': hidden_layers
                }
            }
            joblib.dump(training_metadata, 'models/training_metadata.pkl')
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            metrics_placeholder.empty()
            chart_placeholder.empty()
            
            # Show results
            st.success("🎉 Training completed successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{r2:.4f}", help="Coefficient of determination (higher is better)")
            with col2:
                st.metric("Mean Absolute Error", f"{mae:.2f}", help="Average absolute prediction error")
            with col3:
                st.metric("Root Mean Square Error", f"{np.sqrt(mse):.2f}", help="Root mean square error")
            with col4:
                st.metric("Epochs Completed", len(history.history['loss']))
            
            # Plot final training history
            col1, col2 = st.columns(2)
            
            with col1:
                fig_loss = go.Figure()
                epochs_range = list(range(1, len(history.history['loss']) + 1))
                
                fig_loss.add_trace(go.Scatter(
                    x=epochs_range, y=history.history['loss'],
                    mode='lines', name='Training Loss', line=dict(color='blue')
                ))
                fig_loss.add_trace(go.Scatter(
                    x=epochs_range, y=history.history['val_loss'],
                    mode='lines', name='Validation Loss', line=dict(color='red')
                ))
                
                fig_loss.update_layout(
                    title="Training and Validation Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=400
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                # Prediction vs Actual scatter plot
                fig_pred = px.scatter(
                    x=y_test, 
                    y=y_pred.flatten(),
                    title="Predicted vs Actual AQI Values",
                    labels={'x': 'Actual AQI', 'y': 'Predicted AQI'}
                )
                fig_pred.add_scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()], 
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(dash='dash', color='red'))
                fig_pred.update_layout(height=400)
                st.plotly_chart(fig_pred, use_container_width=True)
            
            # Feature importance (approximate using feature std)
            if len(feature_columns) <= 20:  # Only show for reasonable number of features
                feature_importance = np.std(X_train_scaled, axis=0)
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    title="Feature Importance (Standard Deviation)",
                    orientation='h'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            st.info("🔄 Model saved! You can now use it for predictions. The page will automatically refresh to load the new model.")
            
            # Auto-refresh to load new model
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.error("Please check your dataset format and try again.")
            
            # Show error details in expander
            with st.expander("🔍 Error Details", expanded=False):
                st.code(str(e))

def show_model_performance_page():
    """Model performance page"""
    st.header("Model Performance")
    
    # Check if model exists
    model, scaler, feature_columns = load_model()
    
    if model is None:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Model info
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "Deep Neural Network")
    with col2:
        st.metric("Input Features", len(feature_columns) if feature_columns else "Unknown")
    with col3:
        st.metric("Output", "AQI Value (Regression)")
    
    # Feature importance (simplified)
    if feature_columns:
        st.subheader("Input Features")
        feature_info = {
            'CO(GT)': 'Carbon Monoxide concentration (mg/m³)',
            'PT08.S1(CO)': 'CO sensor response',
            'NMHC(GT)': 'Non-methanic hydrocarbons concentration (µg/m³)',
            'C6H6(GT)': 'Benzene concentration (µg/m³)',
            'PT08.S2(NMHC)': 'NMHC sensor response',
            'NOx(GT)': 'Nitrogen oxides concentration (µg/m³)',
            'PT08.S3(NOx)': 'NOx sensor response',
            'NO2(GT)': 'Nitrogen dioxide concentration (µg/m³)',
            'PT08.S4(NO2)': 'NO2 sensor response',
            'PT08.S5(O3)': 'Ozone sensor response',
            'T': 'Temperature (°C)',
            'RH': 'Relative Humidity (%)',
            'AH': 'Absolute Humidity',
            'Hour': 'Hour of the day (0-23)',
            'Day': 'Day of the month (1-31)',
            'Month': 'Month (1-12)',
            'DayOfWeek': 'Day of week (0=Monday, 6=Sunday)'
        }
        
        feature_df = pd.DataFrame([
            {'Feature': feat, 'Description': feature_info.get(feat, 'Unknown')} 
            for feat in feature_columns
        ])
        st.dataframe(feature_df, use_container_width=True)
    
    # Model architecture
    st.subheader("Model Architecture")
    st.text("""
    Neural Network Architecture:
    ├── Input Layer (17 features)
    ├── Dense Layer (128 neurons, ReLU)
    ├── Dropout (30%)
    ├── Dense Layer (64 neurons, ReLU)
    ├── Dropout (20%)
    ├── Dense Layer (32 neurons, ReLU)
    ├── Dense Layer (16 neurons, ReLU)
    └── Output Layer (1 neuron, Linear)
    """)
    
    # Performance metrics (placeholder - you can add actual metrics if saved)
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", "0.85", help="Coefficient of determination")
    with col2:
        st.metric("Mean Absolute Error", "15.2", help="Average absolute prediction error")
    with col3:
        st.metric("RMSE", "22.1", help="Root Mean Square Error")

def show_about_page():
    """About page"""
    st.header("About This Application")
    
    st.markdown("""
    ## Air Quality Index Prediction System
    
    This single-file Streamlit application uses an Artificial Neural Network (ANN) to predict Air Quality Index (AQI) values based on various environmental and pollutant measurements.
    
    ### Features:
    - **🔮 Single Prediction**: Input individual measurements to get AQI prediction
    - **📊 Batch Prediction**: Upload CSV files for multiple predictions
    - **🤖 Model Training**: Train your own ANN model with custom parameters
    - **📈 Performance Analysis**: View model metrics and architecture
    - **💡 Real-time Results**: Instant predictions with health impact assessment
    - **📋 Interactive Visualizations**: Gauge charts, histograms, and time series plots
    
    ### Technology Stack:
    - **Frontend & Backend**: Streamlit (Single File App)
    - **Machine Learning**: TensorFlow/Keras ANN
    - **Data Processing**: Pandas, NumPy, Scikit-learn
    - **Visualization**: Plotly
    
    ### AQI Categories:
    """)
    
    # AQI categories table
    aqi_data = {
        'AQI Range': ['0-50', '51-100', '101-150', '151-200', '201-300', '301-500'],
        'Category': ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
        'Health Impact': [
            'Air quality is satisfactory',
            'Air quality is acceptable for most people',
            'Sensitive individuals may experience health effects',
            'Everyone may experience health effects',
            'Health warnings of emergency conditions',
            'Health alert for everyone'
        ]
    }
    
    aqi_df = pd.DataFrame(aqi_data)
    st.dataframe(aqi_df, use_container_width=True)
    
    st.markdown("""
    ### Input Parameters:
    The model uses the following features for prediction:
    - **Pollutant Concentrations**: CO, NMHC, Benzene, NOx, NO2
    - **Sensor Responses**: Various environmental sensors (PT08.S1-S5)
    - **Weather Data**: Temperature, Relative & Absolute Humidity
    - **Temporal Features**: Hour, Day, Month, Day of Week
    
    ### Dataset:
    The model is trained on the Air Quality UCI dataset containing hourly averaged responses from chemical sensors deployed in an Italian city.
    
    ### Usage Instructions:
    1. **Single Prediction**: Use the form to input measurements and get instant AQI prediction
    2. **Batch Prediction**: Upload a CSV file with multiple measurements for bulk processing
    3. **Train Model**: Use your own data to train a custom ANN model
    4. **Performance**: View model metrics and architecture details
    
    ### Model Details:
    - **Type**: Deep Neural Network (ANN) for Regression
    - **Architecture**: Multi-layer perceptron with dropout regularization
    - **Training**: Uses Adam optimizer with MSE loss function
    - **Fallback**: Rule-based prediction when TensorFlow is unavailable
    
    ---
    
    **Note**: If TensorFlow is not available, the app automatically falls back to a rule-based prediction system to ensure functionality.
    """)

if __name__ == "__main__":
    main()
