# streamlit_frontend.py
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🌍 Air Quality Forecast Dashboard")
st.markdown("### 📅 Today's data + next 3 day forecasts")

# Fetch data
response = requests.get("http://127.0.0.1:8000/aqi-data")
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
else:
    st.error("❌ Failed to load data from backend.")
    st.stop()

# Clean timestamp
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"]).dt.date

# Label rows
df.reset_index(drop=True, inplace=True)
labels = ["Today", "Forecast +1", "Forecast +2", "Forecast +3"]
df.index = labels

# AQI category for 0–5 scale
def get_aqi_category(aqi):
    if aqi <= 1.0:
        return "🟢 Good", "green"
    elif aqi <= 2.0:
        return "🟡 Moderate", "gold"
    elif aqi <= 3.0:
        return "🟠 Sensitive", "orange"
    elif aqi <= 4.0:
        return "🔴 Unhealthy", "red"
    else:
        return "🟣 Very Unhealthy", "purple"

# --- Containers for individual days ---
for i in range(len(df)):
    with st.container():
        st.subheader(f"📌 {labels[i]}")
        row = df.iloc[i:i+1].copy()
        
        # Special display for Today
        if labels[i] == "Today":
            aqi_val = float(row["predicted_aqi"].values[0])
            cat_label, color = get_aqi_category(aqi_val)
            st.markdown(f"**AQI:** <span style='color:{color}; font-size:22px'>{aqi_val:.2f} {cat_label}</span>", unsafe_allow_html=True)
            
            # Rename column for Today
            row = row.rename(columns={"predicted_aqi": "AQI"})
        st.write(row)

# --- Side-by-side plots ---
features_to_plot = ["predicted_aqi", "pm2_5", "pm10", "co"]
feature_colors = {
    "predicted_aqi": "teal",
    "pm2_5": "darkorange",
    "pm10": "royalblue",
    "co": "seagreen"
}

st.markdown("## 📊 AQI Features Over Time")
cols = st.columns(len(features_to_plot))

for idx, feature in enumerate(features_to_plot):
    with cols[idx]:
        st.markdown(f"**{feature}**")
        fig, ax = plt.subplots()

        color = feature_colors.get(feature, "gray")

        # Plot today's data point (solid line)
        ax.plot(df.index[:1], df[feature].iloc[:1], marker='o', color=color, label="Today")

        # Plot forecast data points (dashed line)
        ax.plot(df.index[1:], df[feature].iloc[1:], marker='o', linestyle='dashed', color=color, label="Forecast")

        ax.set_title(feature)
        ax.set_xlabel("Day")
        ax.set_ylabel(feature)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
