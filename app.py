import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ETo Prediction System",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2E7D32;
        font-size: 2.5rem !important;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin: 20px 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌾 Reference Evapotranspiration (ETo) Prediction")
st.markdown("### AI-Powered ETo Calculator with Smart Missing Value Handling")
st.markdown("---")

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model, scalers, and smart imputer"""
    try:
        with open('eto_ann_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('smart_imputer.pkl', 'rb') as f:
            smart_imputer = pickle.load(f)
        return model, scaler_X, scaler_y, feature_names, smart_imputer
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found! Missing file: {e.filename}")
        st.info("Please ensure all model files are in the same directory as app.py")
        return None, None, None, None, None

# Smart prediction function
def predict_eto(input_dict, model, scaler_X, scaler_y, feature_names, smart_imputer):
    """Predict ETo with smart imputation for missing values"""
    # Create input array with NaN for missing values
    input_array = np.full(len(feature_names), np.nan)

    for i, feature in enumerate(feature_names):
        if feature in input_dict and input_dict[feature] is not None:
            input_array[i] = input_dict[feature]

    # Use smart imputer to estimate missing values
    input_imputed = smart_imputer.transform(input_array.reshape(1, -1))

    # Scale and predict
    input_scaled = scaler_X.transform(input_imputed)
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()[0]

    return prediction, input_imputed[0]

# Load model
model, scaler_X, scaler_y, feature_names, smart_imputer = load_model()

if model is not None:
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts **Reference Evapotranspiration (ETo)** 
        using a trained Artificial Neural Network with smart missing value handling.

        **Model Details:**
        - Architecture: 4-layer ANN (128-64-32-16)
        - Accuracy: R² = 0.986
        - RMSE: 0.232 mm/day
        - MAE: 0.168 mm/day
        - Imputation: MICE algorithm

        **Features:**
        - ✅ Works with 2-6 input parameters
        - ✅ Smart missing value estimation
        - ✅ Real-time predictions
        - ✅ More accurate with partial inputs
        """)

        st.markdown("---")
        st.header("📊 Training Data")
        st.markdown("""
        - Total samples: 7,665
        - Training: 70%
        - Validation: 15%
        - Test: 15%
        - Data source: Ludhiana, Punjab
        - Years: 2000-2020
        """)

        st.markdown("---")
        st.header("🎯 Parameter Guide")
        st.markdown("""
        **Minimum Requirements:**
        - At least **2 parameters** needed
        - More parameters = better accuracy
        - Missing values auto-estimated

        **Recommended Combinations:**
        - Tmax + Tmin (Basic)
        - Tmax + Tmin + RHmax (Good)
        - All 6 parameters (Best)
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Input Parameters")
        st.markdown("**Enter at least 2 parameters. Missing values will be intelligently estimated.**")

        # Create input form
        with st.form("prediction_form"):
            col_input1, col_input2 = st.columns(2)

            with col_input1:
                n = st.number_input(
                    "🌞 Sunshine Hours (n)",
                    min_value=0.0,
                    max_value=15.0,
                    value=None,
                    step=0.1,
                    help="Actual sunshine hours per day (0-15 hours)"
                )

                tmax = st.number_input(
                    "🌡️ Maximum Temperature (°C)",
                    min_value=-10.0,
                    max_value=50.0,
                    value=None,
                    step=0.1,
                    help="Maximum daily temperature in Celsius"
                )

                tmin = st.number_input(
                    "🌡️ Minimum Temperature (°C)",
                    min_value=-20.0,
                    max_value=40.0,
                    value=None,
                    step=0.1,
                    help="Minimum daily temperature in Celsius"
                )

            with col_input2:
                rhmax = st.number_input(
                    "💧 Maximum Relative Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=None,
                    step=1.0,
                    help="Maximum relative humidity percentage"
                )

                rhmin = st.number_input(
                    "💧 Minimum Relative Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=None,
                    step=1.0,
                    help="Minimum relative humidity percentage"
                )

                u = st.number_input(
                    "💨 Wind Speed (m/s)",
                    min_value=0.0,
                    max_value=10.0,
                    value=None,
                    step=0.1,
                    help="Wind speed at 2m height in meters per second"
                )

            # Submit button
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                submitted = st.form_submit_button("🚀 Predict ETo", use_container_width=True, type="primary")
            with col_btn2:
                clear = st.form_submit_button("🔄 Clear", use_container_width=True)

        if submitted:
            # Collect inputs
            input_dict = {
                'n': n,
                'Tmax (°C)': tmax,
                'Tmin (°C)': tmin,
                'RHmax': rhmax,
                'RHmin': rhmin,
                'u ': u
            }

            # Count provided inputs
            provided_params = sum(1 for v in input_dict.values() if v is not None)

            if provided_params < 2:
                st.error("⚠️ Please provide at least 2 input parameters!")
            else:
                # Make prediction
                prediction, imputed_values = predict_eto(
                    input_dict, model, scaler_X, scaler_y, feature_names, smart_imputer
                )

                # Store in session state for display
                st.session_state.prediction = prediction
                st.session_state.provided_params = provided_params
                st.session_state.input_dict = input_dict
                st.session_state.imputed_values = imputed_values
                st.session_state.feature_names = feature_names

    # Display results in col2
    with col2:
        st.subheader("📊 Results")

        if 'prediction' in st.session_state:
            # Prediction box
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #2E7D32; margin: 0;">
                    {st.session_state.prediction:.4f}
                </h2>
                <p style="text-align: center; margin: 5px 0; font-size: 1.2rem; color: #666;">
                    mm/day
                </p>
                <p style="text-align: center; margin: 0; color: #888;">
                    Reference Evapotranspiration
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Input summary
            st.markdown("**Input Summary:**")
            accuracy_level = "Excellent" if st.session_state.provided_params >= 5 else                            "Good" if st.session_state.provided_params >= 3 else "Fair"
            st.info(f"✅ Provided: {st.session_state.provided_params}/6 parameters\n\n"
                   f"Accuracy Level: **{accuracy_level}**")

            # Show which parameters were used
            st.markdown("**📥 Parameters Provided:**")
            for param, value in st.session_state.input_dict.items():
                if value is not None:
                    param_display = param.replace('u ', 'Wind Speed')
                    st.markdown(f"✓ {param_display}: **{value}**")

            # Show estimated parameters with expander
            estimated = [(param, i) for i, param in enumerate(st.session_state.feature_names) 
                        if param not in st.session_state.input_dict or 
                        st.session_state.input_dict[param] is None]

            if estimated:
                with st.expander("🔮 Auto-Estimated Values (Click to view)"):
                    st.markdown("*These values were intelligently estimated based on provided parameters:*")
                    for param, idx in estimated:
                        param_display = param.replace('u ', 'Wind Speed')
                        st.markdown(f"○ {param_display}: **{st.session_state.imputed_values[idx]:.2f}**")
        else:
            st.info("👈 Enter parameters and click 'Predict ETo' to see results")

    # Additional information
    st.markdown("---")
    st.subheader("📚 How to Use")
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.markdown("""
        **Step 1: Input Data**
        - Enter at least 2 parameters
        - More parameters = better accuracy
        - Leave blank for auto-estimation
        """)

    with col_info2:
        st.markdown("""
        **Step 2: Predict**
        - Click "Predict ETo" button
        - View results on the right
        - Check estimated values
        """)

    with col_info3:
        st.markdown("""
        **Step 3: Interpret**
        - ETo in mm/day
        - Used for irrigation planning
        - Higher values = more water needed
        """)

    # Example scenarios
    st.markdown("---")
    st.subheader("💡 Quick Test Scenarios")

    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

    with col_ex1:
        if st.button("☀️ Hot Summer Day", use_container_width=True):
            example = {
                'n': 8.0, 'Tmax (°C)': 38.0, 'Tmin (°C)': 25.0,
                'RHmax': 70.0, 'RHmin': 30.0, 'u ': 2.0
            }
            pred, imp = predict_eto(example, model, scaler_X, scaler_y, feature_names, smart_imputer)
            st.success(f"**Predicted ETo:** {pred:.4f} mm/day")
            with st.expander("View Input"):
                st.json(example)

    with col_ex2:
        if st.button("🌤️ Mild Spring Day", use_container_width=True):
            example = {
                'n': 6.0, 'Tmax (°C)': 25.0, 'Tmin (°C)': 15.0,
                'RHmax': 85.0, 'RHmin': 60.0, 'u ': 1.0
            }
            pred, imp = predict_eto(example, model, scaler_X, scaler_y, feature_names, smart_imputer)
            st.success(f"**Predicted ETo:** {pred:.4f} mm/day")
            with st.expander("View Input"):
                st.json(example)

    with col_ex3:
        if st.button("❄️ Cool Winter Day", use_container_width=True):
            example = {
                'n': 5.0, 'Tmax (°C)': 18.0, 'Tmin (°C)': 8.0,
                'RHmax': 95.0, 'RHmin': 75.0, 'u ': 0.5
            }
            pred, imp = predict_eto(example, model, scaler_X, scaler_y, feature_names, smart_imputer)
            st.success(f"**Predicted ETo:** {pred:.4f} mm/day")
            with st.expander("View Input"):
                st.json(example)

    with col_ex4:
        if st.button("🌡️ Only Temp Data", use_container_width=True):
            example = {
                'Tmax (°C)': 30.0, 'Tmin (°C)': 18.0
            }
            pred, imp = predict_eto(example, model, scaler_X, scaler_y, feature_names, smart_imputer)
            st.success(f"**Predicted ETo:** {pred:.4f} mm/day")
            st.warning("⚠️ Only 2 parameters - Less accurate")
            with st.expander("View Input + Estimated"):
                st.write("**Provided:**", example)
                st.write("**Estimated values:**")
                for i, feat in enumerate(feature_names):
                    if feat not in example:
                        st.write(f"  {feat}: {imp[i]:.2f}")

    # Interpretation guide
    st.markdown("---")
    st.subheader("📖 ETo Interpretation Guide")

    col_guide1, col_guide2 = st.columns(2)

    with col_guide1:
        st.markdown("""
        **ETo Ranges (mm/day):**
        - **0-2**: Low evapotranspiration
          - Cloudy, cool, humid conditions
          - Winter season

        - **2-4**: Moderate evapotranspiration
          - Normal spring/autumn weather
          - Mild temperatures

        - **4-6**: High evapotranspiration
          - Hot summer days
          - Low humidity

        - **6+**: Very high evapotranspiration
          - Extreme heat
          - Very dry conditions
        """)

    with col_guide2:
        st.markdown("""
        **Irrigation Recommendations:**
        - **ETo < 2**: Minimal irrigation needed
        - **ETo 2-4**: Moderate irrigation
        - **ETo 4-6**: Increased irrigation
        - **ETo > 6**: Heavy irrigation required

        **Factors Affecting ETo:**
        - Temperature (most important)
        - Humidity
        - Wind speed
        - Solar radiation (sunshine hours)
        """)

else:
    st.error("❌ Failed to load model files.")
    st.info("""
    **Required files:**
    - eto_ann_model.pkl
    - scaler_X.pkl
    - scaler_y.pkl
    - feature_names.pkl
    - smart_imputer.pkl

    Please ensure all files are in the same directory as app.py
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🌾 ETo Prediction System | Powered by Artificial Neural Network with Smart Imputation</p>
    <p>Developed for Agricultural Applications | Ludhiana, Punjab Data | 2026</p>
</div>
""", unsafe_allow_html=True)