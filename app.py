import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------
# 1. LOAD DATA
# ----------------------------------------
st.title("Marine Mammal Observation Dashboard & ML Model")

file_path = r"C:\Users\kalya\OneDrive\Desktop\Project\final_marine_mammal_integration.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(file_path)
    return df

df = load_data()
st.success("Dataset Loaded Successfully!")
st.write("### Dataset Preview")
st.dataframe(df.head())

# ----------------------------------------
# 2. SIDEBAR FILTERS
# ----------------------------------------
st.sidebar.header("Filters")

species = st.sidebar.multiselect(
    "Select Species",
    options=df["scientificName"].unique(),
    default=df["scientificName"].unique()
)

method = st.sidebar.multiselect(
    "Select Observation Method",
    options=df["methodObs"].unique(),
    default=df["methodObs"].unique()
)

df_filtered = df[
    (df["scientificName"].isin(species)) &
    (df["methodObs"].isin(method))
]

st.write("### Filtered Data")
st.dataframe(df_filtered.head())

# ----------------------------------------
# 3. INTERACTIVE MAP
# ----------------------------------------
st.write("## 🌍 Species Distribution Map")

map_df = df_filtered.dropna(subset=["decimalLat", "decimalLon"])

fig_map = px.scatter_mapbox(
    map_df,
    lat="decimalLat",
    lon="decimalLon",
    color="scientificName",
    zoom=2,
    height=500,
    hover_name="scientificName",
    hover_data=["date", "count"],
)

fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------------------
# 4. SPECIES COUNT BAR CHART
# ----------------------------------------
st.write("## 📊 Species Observation Counts")

species_count = df_filtered["scientificName"].value_counts().reset_index()
species_count.columns = ["Species", "Observations"]

fig_bar = px.bar(species_count, x="Species", y="Observations", color="Species")
st.plotly_chart(fig_bar)

# ----------------------------------------
# 5. ML MODEL – Predict Count
# ----------------------------------------
st.write("## 🤖 ML Model: Predict Number of Animals Observed")

# Preprocessing
ml_df = df_filtered.copy()
ml_df = ml_df.dropna(subset=["decimalLat", "decimalLon", "count"])

ml_df["date"] = pd.to_datetime(ml_df["date"])
ml_df["year"] = ml_df["date"].dt.year
ml_df["month"] = ml_df["date"].dt.month

# Encode categorical
ml_df["species_code"] = ml_df["scientificName"].astype("category").cat.codes
ml_df["method_code"] = ml_df["methodObs"].astype("category").cat.codes

X = ml_df[["decimalLat", "decimalLon", "year", "month", "species_code", "method_code"]]
y = ml_df["count"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=150)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"### Model Performance")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**R² Score:** {r2:.3f}")

# User prediction
st.write("### Predict Animal Count (Try Your Own Inputs)")
lat = st.number_input("Latitude", value=40.0)
lon = st.number_input("Longitude", value=-60.0)
year = st.number_input("Year", value=2024)
month = st.number_input("Month (1-12)", value=6)

species_list = df["scientificName"].unique()
method_list = df["methodObs"].unique()

species_input = st.selectbox("Species", species_list)
method_input = st.selectbox("Observation Method", method_list)

sp_code = df["scientificName"].astype("category").cat.categories.get_loc(species_input)
mt_code = df["methodObs"].astype("category").cat.categories.get_loc(method_input)

user_features = np.array([[lat, lon, year, month, sp_code, mt_code]])
prediction = model.predict(user_features)

st.write(f"### **Predicted Count:** {prediction[0]:.2f} animals")

st.success("App Ready! 🎉 Now run:  streamlit run app.py")
