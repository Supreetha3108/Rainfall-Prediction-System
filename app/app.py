import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="wide"
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("About Project")

st.sidebar.write("""
This project predicts
whether rainfall will occur
tomorrow using Machine Learning.

Model Used:

✓ XGBoost

Dataset:

✓ Australian Weather Dataset

Accuracy:

✓ 84.34%
""")

# =====================================
# LOAD MODEL
# =====================================

model = pickle.load(
    open(
        "models/rainfall_model.pkl",
        "rb"
    )
)

# =====================================
# TITLE
# =====================================

st.markdown(
"""
# 🌧️ Rainfall Prediction System

### Machine Learning Based Weather Forecasting
"""
)

st.markdown("---")

# =====================================
# METRICS
# =====================================

col1,col2,col3=st.columns(3)

with col1:
    st.metric(
        "Model",
        "XGBoost"
    )

with col2:
    st.metric(
        "Accuracy",
        "84.34%"
    )

with col3:
    st.metric(
        "Features",
        "8"
    )

st.markdown("---")

# =====================================
# INPUT SECTION
# =====================================

st.subheader(
    "Enter Weather Information"
)

left,right=st.columns(2)

with left:

    min_temp=st.number_input(
        "Min Temperature (°C)",
        value=10.0
    )

    max_temp=st.number_input(
        "Max Temperature (°C)",
        value=25.0
    )

    rainfall=st.number_input(
        "Rainfall (mm)",
        value=0.0
    )

    wind_gust=st.number_input(
        "Wind Gust Speed",
        value=30.0
    )

with right:

    humidity9=st.number_input(
        "Humidity 9AM (%)",
        value=60.0
    )

    humidity3=st.number_input(
        "Humidity 3PM (%)",
        value=50.0
    )

    pressure9=st.number_input(
        "Pressure 9AM (hPa)",
        value=1015.0
    )

    pressure3=st.number_input(
        "Pressure 3PM (hPa)",
        value=1013.0
    )

st.markdown("---")

# =====================================
# PREDICTION
# =====================================

if st.button(
    "🔍 Predict Rain"
):

    features=np.array([
        [
            min_temp,
            max_temp,
            rainfall,
            wind_gust,
            humidity9,
            humidity3,
            pressure9,
            pressure3
        ]
    ])

    prediction=model.predict(
        features
    )

    probability=model.predict_proba(
        features
    )[0][1]

    rain_probability=probability*100

    st.subheader(
        "Prediction Result"
    )

    if prediction[0]==1:

        st.error(
            "🌧 Rain Expected Tomorrow"
        )

    else:

        st.success(
            "☀ No Rain Expected Tomorrow"
        )

    st.write(
        f"Rain Probability: {rain_probability:.2f}%"
    )

    st.progress(
        int(rain_probability)
    )

    st.markdown("---")

    # ==========================
    # FEATURE IMPORTANCE GRAPH
    # ==========================

    st.subheader(
        "Feature Importance"
    )

    importance=model.feature_importances_

    feature_names=[
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "WindGust",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm"
    ]

    fig,ax=plt.subplots(
        figsize=(8,4)
    )

    ax.bar(
        feature_names,
        importance
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

st.markdown("---")

st.caption(
"Developed using Python, Streamlit and XGBoost"
)