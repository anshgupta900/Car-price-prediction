import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

data = joblib.load("cars_price_model.pkl")

model = data["model"]
scaler = data["scaler"]
columns = data["columns"]
scaled = data["scaled"]

df = pd.read_csv("cars.csv")


st.title("🚗 Car Price Prediction")
st.write("Enter the details of your car")

st.divider()

car_name = st.selectbox(
    "Car Name",
    sorted(df["Car Name"].dropna().unique())
)

year = st.number_input(
    "Year",
    min_value=int(df["Year"].min()),
    max_value=int(df["Year"].max()),
    value=int(df["Year"].max())
)

distance = st.number_input(
    "Distance (km)",
    min_value=0,
    value=50000
)

owner = st.selectbox(
    "Owner",
    sorted(df["Owner"].dropna().unique())
)

fuel = st.selectbox(
    "Fuel Type",
    sorted(df["Fuel"].dropna().unique())
)

location = st.selectbox(
    "Location",
    sorted(df["Location"].dropna().unique())
)

drive = st.selectbox(
    "Drive",
    sorted(df["Drive"].dropna().unique())
)

car_type = st.selectbox(
    "Car Type",
    sorted(df["Type"].dropna().unique())
)


st.divider()


if st.button("🚗 Predict Price", use_container_width=True):

    input_data = pd.DataFrame({
        "Car Name": [car_name],
        "Year": [year],
        "Distance": [distance],
        "Owner": [owner],
        "Fuel": [fuel],
        "Location": [location],
        "Drive": [drive],
        "Type": [car_type]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "Car Name",
            "Owner",
            "Fuel",
            "Location",
            "Drive",
            "Type"
        ],
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

    if scaled:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
    else:
        prediction = model.predict(input_data)

    st.success(
        f"💰 Estimated Car Price: ₹{prediction[0]:,.2f}"
    )